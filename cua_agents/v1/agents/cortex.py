"""
Cortex — LangGraph Orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Central orchestrator that manages the flow between:
  • Orchestrator Node  — GPT-5-mini (Azure) brain that decides what to do next
  • GUI Worker Node    — EvoCUA model for screen interactions (click, type, etc.)
  • MCP Worker Node    — LLM + MCP tools (Slack, Notion) via ToolManager
  • Code Worker Node   — Local Python / Bash execution for file processing

The graph is a loop:
    orchestrator → (conditional edge) → worker → orchestrator → …
until the orchestrator emits next_node="__end__".
"""

import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TypedDict

import pyautogui
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langfuse import observe, get_client

from cua_agents.v1.agents.code_agent import CodeAgent
from cua_agents.v1.agents.evocua_agent import EvoCUAAgent
from cua_agents.v1.agents.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    MCP_WORKER_SYSTEM_PROMPT,
)
from cua_agents.v1.tools.tool_manager import ToolManager
from cua_agents.v1.utils.local_env import LocalController

load_dotenv()

# Central Workspace for all file operations
CORTEX_WORKSPACE = os.path.join(os.path.expanduser("~"), "Desktop", "cortex_workspace")
if not os.path.exists(CORTEX_WORKSPACE):
    os.makedirs(CORTEX_WORKSPACE, exist_ok=True)

logger = logging.getLogger("cortex.orchestrator")


# ── State ────────────────────────────────────────────────────────────────────


class CortexState(TypedDict):
    """Shared state flowing through every node in the graph."""

    messages: List[Dict[str, Any]]
    screenshot: Optional[str]
    working_files: List[str]
    next_node: str
    task: str
    step: int
    max_steps: int
    last_worker_result: str
    mcp_tool_descriptions: str
    orchestrator_reasoning: str
    _instruction: str


# ── Module-level references (set by Cortex.__init__) ─────────────────────────

_llm: Optional[AzureChatOpenAI] = None
_tool_manager: Optional[ToolManager] = None
_tool_manager_connected: bool = False
_evocua_agent: Optional[EvoCUAAgent] = None
_code_agent: Optional[CodeAgent] = None
_local_controller: Optional[LocalController] = None

# Persistent background event loop for MCP async operations.
# MCP stdio connections are bound to a single event loop; all async MCP work
# must run in this loop regardless of which thread calls it.
_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_thread: Optional[threading.Thread] = None


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """Return (or start) the persistent background event loop."""
    global _bg_loop, _bg_thread
    if _bg_loop is None or not _bg_loop.is_running():
        _bg_loop = asyncio.new_event_loop()
        _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True, name="cortex-bg-loop")
        _bg_thread.start()
    return _bg_loop


def _run_in_bg_loop(coro) -> Any:
    """Submit *coro* to the background loop and block until it completes."""
    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()

# Callbacks injected by ws_server (or default no-ops)
_hide_ui_fn: Callable[[], None] = lambda: None
_show_ui_fn: Callable[[], None] = lambda: None
_stop_flag: Optional[threading.Event] = None
_log_fn: Callable[..., None] = lambda msg, *a, **kw: None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _capture_screenshot_b64() -> str:
    """Capture the screen (with UI hidden) and return base64 PNG."""
    _hide_ui_fn()
    shot = pyautogui.screenshot()
    _show_ui_fn()
    buf = io.BytesIO()
    shot.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _capture_screenshot_bytes() -> bytes:
    """Capture the screen (with UI hidden) and return raw PNG bytes."""
    _hide_ui_fn()
    shot = pyautogui.screenshot()
    _show_ui_fn()
    buf = io.BytesIO()
    shot.save(buf, format="PNG")
    return buf.getvalue()


def _is_stopped() -> bool:
    return _stop_flag is not None and _stop_flag.is_set()


# ── Orchestrator Node (Brain) ────────────────────────────────────────────────





def _build_orchestrator_messages(state: CortexState) -> list:
    system_msg = ORCHESTRATOR_SYSTEM_PROMPT.format(
        mcp_tools=state.get("mcp_tool_descriptions", "No MCP tools available."),
        workspace=CORTEX_WORKSPACE
    )
    messages = [{"role": "system", "content": system_msg}]
    messages.append({
        "role": "user",
        "content": f"Task: {state['task']}\n\nStep: {state['step']}/{state['max_steps']}"
    })
    for msg in state["messages"][-10:]:
        messages.append(msg)
    if state.get("screenshot"):
        # Encode bytes to B64 only at the point of message construction
        b64_img = base64.b64encode(state["screenshot"]).decode("utf-8")
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Current screenshot of the desktop:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
            ]
        })
    if state.get("last_worker_result"):
        messages.append({
            "role": "user",
            "content": f"Last worker result:\n{state['last_worker_result']}"
        })
    return messages


@observe(name="orchestrator_node")
def orchestrator_node(state: CortexState) -> dict:
    """The brain — decides what worker to call next."""
    step = state["step"]
    max_s = state["max_steps"]
    logger.info("🧠 Orchestrator (step %d/%d)", step, max_s)
    _log_fn(f"Step {step + 1}/{max_s}: Thinking…", "step", "🧠")

    if _is_stopped():
        _log_fn("Stopped by user.", "warning", "⏹️")
        return {"next_node": "__end__", "orchestrator_reasoning": "Stopped by user.", "step": step + 1}

    if step >= max_s:
        _log_fn("Step budget exhausted.", "warning", "⏰")
        return {"next_node": "__end__", "orchestrator_reasoning": "Budget exhausted.", "step": step + 1}

    _log_fn("Capturing screen…", "step", "📷")
    screenshot_bytes = _capture_screenshot_bytes()

    updated_state = {**state, "screenshot": screenshot_bytes}
    messages = _build_orchestrator_messages(updated_state)

    try:
        response = _llm.invoke(messages)
        raw_text = response.content
        print(f"[{time.strftime('%H:%M:%S')}] 🧠 raw: {raw_text}", flush=True)

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            cleaned = cleaned.rsplit("```", 1)[0]
        decision = json.loads(cleaned)

    except json.JSONDecodeError as e:
        logger.error("Parse error: %s — %s", e, raw_text[:500])
        decision = {"reasoning": f"Parse error: {raw_text[:200]}", "next_node": "__end__", "instruction": ""}
    except Exception as e:
        logger.error("Orchestrator LLM error: %s", e)
        decision = {"reasoning": f"LLM error: {e}", "next_node": "__end__", "instruction": ""}

    next_node = decision.get("next_node", "__end__")
    instruction = decision.get("instruction", "")
    reasoning = decision.get("reasoning", "")

    str_instruction = str(instruction)
    _log_fn(f"→ {next_node}: {str_instruction}", "step", "🧠")
    print(f"[{time.strftime('%H:%M:%S')}] 🧠 → {next_node}: {str_instruction}", flush=True)

    langfuse = get_client()
    # Encode for logging purposes
    log_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    langfuse.update_current_span(
        metadata={"step": step, "next_node": next_node},
        input={"task": state["task"], "screenshot": f"data:image/png;base64,{log_b64}"},
        output=decision,
    )

    new_msg = {"role": "assistant", "content": f"[Orchestrator → {next_node}] {instruction}"}
    return {
        "screenshot": screenshot_bytes,
        "next_node": next_node,
        "orchestrator_reasoning": reasoning,
        "step": step + 1,
        "messages": state["messages"] + [new_msg],
        "_instruction": instruction,
    }


# ── GUI Worker Node (EvoCUA) ─────────────────────────────────────────────────


def _get_evocua_agent() -> EvoCUAAgent:
    global _evocua_agent
    if _evocua_agent is None:
        _evocua_agent = EvoCUAAgent(
            base_url=os.getenv("GROUND_URL", ""),
            model=os.getenv("GROUND_MODEL", "meituan/EvoCUA-8B-20260105"),
            api_key=os.getenv("GROUND_API_KEY", "any-value"),
            workspace=CORTEX_WORKSPACE,
            screen_size=(
                int(os.getenv("GROUNDING_WIDTH", "1920")),
                int(os.getenv("GROUNDING_HEIGHT", "1080")),
            ),
            max_history=4, temperature=0.01, resize_factor=32,
        )
    return _evocua_agent


@observe(name="gui_worker_node")
def gui_worker_node(state: CortexState) -> dict:
    if _is_stopped():
        return {"last_worker_result": "Stopped.", "messages": state["messages"], "next_node": "orchestrator"}

    instruction = str(state.get("_instruction", state.get("task", "")))
    _log_fn(f"GUI Task: {instruction}", "step", "🖱️")
    print(f"[{time.strftime('%H:%M:%S')}] 🖱️ GUI: {instruction}", flush=True)

    agent = _get_evocua_agent()
    agent.reset() # Start fresh history for this delegation

    step_count = 0
    max_sub_steps = 30 # Internal budget to prevent infinite loops
    last_result = "Delegated to GUI worker."
    current_messages = list(state["messages"])

    while step_count < max_sub_steps:
        if _is_stopped():
            last_result = "Stopped by user."
            break

        _log_fn(f"GUI ({step_count+1}/{max_sub_steps}): Thinking…", "step", "🧠")
        screenshot_bytes = _capture_screenshot_bytes()
        
        info, codes = agent.predict(instruction=instruction, observation={"screenshot": screenshot_bytes})
        action = codes[0] if codes else "WAIT"
        
        desc = str(info.get("action_description", action))
        _log_fn(f"Action: {desc}", "info", "▶️")
        print(f"[{time.strftime('%H:%M:%S')}] 🖱️ GUI Step {step_count+1}: {desc}", flush=True)

        if action == "DONE":
            last_result = f"GUI Task Completed: {desc}"
            _log_fn("GUI: Done.", "success", "🏁")
            break
        if action == "FAIL":
            last_result = f"GUI Task Failed: {desc}"
            _log_fn("GUI: Failed.", "error", "❌")
            break

        # Execute
        try:
            if action == "WAIT":
                time.sleep(3)
                result_text = "Waited 3s."
            else:
                exec(action)
                result_text = f"Executed: {action}"
                time.sleep(1)
        except Exception as e:
            result_text = f"GUI Execution error: {e}"
            logger.error("GUI exec error: %s", e)
            last_result = result_text
            break # Exit loop on hard crash

        last_result = result_text
        step_count += 1

    if step_count >= max_sub_steps:
        _log_fn("Budget hit. Summarizing progress…", "info", "📝")
        summary = agent.summarize_progress(instruction)
        last_result = f"GUI budget reached. Status: {summary}"
        _log_fn("GUI: Detailed status reported.", "warning", "⏰")

    new_msg = {"role": "user", "content": f"[GUI Worker] {last_result}"}
    return {
        "last_worker_result": last_result,
        "messages": current_messages + [new_msg],
        "next_node": "orchestrator",
    }


# ── MCP Worker Node (LLM + MCP Tools) ───────────────────────────────────────





def _get_mcp_config() -> dict:
    config = {}
    npm_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    
    # Configure Notion
    notion_token = os.getenv("NOTION_TOKEN")
    if notion_token:
        config["notion"] = {
            "command": npm_cmd,
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "env": {"NOTION_TOKEN": notion_token},
        }
        
    # Configure Slack
    slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
    slack_app_token = os.getenv("SLACK_APP_TOKEN")
    # For some implementations xoxp/user token works too, but we prioritize the bot config we know we have
    if slack_bot_token:
        slack_env = {
            "SLACK_BOT_TOKEN": slack_bot_token,
            "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID", "T0000000") # Dummy to prevent instant crash
        }
        if slack_app_token:
            slack_env["SLACK_APP_TOKEN"] = slack_app_token
            
        config["slack"] = {
            "command": npm_cmd,
            # The community server that's still supported:
            "args": ["-y", "@modelcontextprotocol/server-slack"],
            "env": slack_env,
        }
    return config


async def _ensure_tool_manager() -> ToolManager:
    global _tool_manager, _tool_manager_connected
    if _tool_manager is None:
        _tool_manager = ToolManager(_get_mcp_config())
    if not _tool_manager_connected:
        await _tool_manager.connect()
        _tool_manager_connected = True
    return _tool_manager


def _sanitize_schema(schema: Any, root_schema: Any = None) -> Any:
    """Recursively sanitize JSON schema to appease strict Pydantic v2 parsers (like Gemini SDK)."""
    if root_schema is None:
        root_schema = schema
        
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_sanitize_schema(item, root_schema) for item in schema]
        return schema
        
    s = dict(schema)
    
    # Resolve $ref
    if "$ref" in s:
        ref_path = s["$ref"]
        if ref_path.startswith("#/"):
            parts = ref_path.split("/")[1:]
            resolved = root_schema
            for p in parts:
                if p in resolved:
                    resolved = resolved[p]
                else:
                    break
            else:
                return _sanitize_schema(resolved, root_schema)
        return s

    # Simplify oneOf/anyOf
    if "oneOf" in s and isinstance(s["oneOf"], list) and len(s["oneOf"]) > 0:
        return _sanitize_schema(s["oneOf"][0], root_schema)
    if "anyOf" in s and isinstance(s["anyOf"], list) and len(s["anyOf"]) > 0:
        return _sanitize_schema(s["anyOf"][0], root_schema)

    # Handle const
    if "const" in s:
        val = s["const"]
        t = "string" if isinstance(val, str) else "integer" if isinstance(val, int) else "boolean" if isinstance(val, bool) else "object"
        return {"type": t, "description": f"Must be {val}"}
    
    # If the schema has no type but has properties, it's an object
    if "properties" in s and "type" not in s:
        s["type"] = "object"
        
    # If schema has basically nothing, default to string to satisfy type requirements
    if not s:
        return {"type": "string"}
        
    # Remove None values
    keys_to_remove = [k for k, v in s.items() if v is None]
    for k in keys_to_remove:
        del s[k]
        
    # If type is a list (e.g., ["string", "null"]), grab the first non-null type
    if "type" in s and isinstance(s["type"], list):
        types = [t for t in s["type"] if t != "null"]
        s["type"] = types[0] if types else "string"
        
    # Special fix for Notion MCP API schemas
    # Notion schemas often omit "type": "object" on nested properties
    if "type" not in s and ("properties" in s or "additionalProperties" in s):
        s["type"] = "object"
        
    for k, v in list(s.items()):
        s[k] = _sanitize_schema(v, root_schema)
        
    # Gemini SDK hates $defs, remove it from the root at the end of resolution
    if "$defs" in s:
        del s["$defs"]
            
    return s


def _mcp_tools_to_openai_functions(tool_manager: ToolManager) -> list:
    """Convert MCP tools → OpenAI function-calling schema for llm.bind(tools=...)."""
    functions = []
    for qualified_name, info in tool_manager._tool_map.items():
        tool = info["tool"]
        server = info["server"]
        name = tool.name if hasattr(tool, "name") else tool.get("name", "")
        desc = tool.description if hasattr(tool, "description") else tool.get("description", "")
        input_schema = {}
        if hasattr(tool, "inputSchema"):
            input_schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
        elif hasattr(tool, "input_schema"):
            input_schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
            
        input_schema = _sanitize_schema(input_schema, input_schema)
        
        functions.append({
            "type": "function",
            "function": {
                "name": f"{server}__{name}",
                "description": f"[{server.upper()}] {desc}",
                "parameters": input_schema or {"type": "object", "properties": {}},
            },
        })
    return functions


@observe(name="mcp_worker_node")
def mcp_worker_node(state: CortexState) -> dict:
    if _is_stopped():
        return {"last_worker_result": "Stopped.", "messages": state["messages"], "next_node": "orchestrator"}

    instruction = str(state.get("_instruction", ""))
    _log_fn(f"MCP: {instruction}", "step", "🔧")
    print(f"[{time.strftime('%H:%M:%S')}] 🔧 MCP: {instruction}", flush=True)

    try:
        result_text = _run_in_bg_loop(_mcp_worker_async(instruction))
    except Exception as e:
        result_text = f"MCP error: {e}"
        logger.error("MCP error: %s", e)

    _log_fn(f"MCP result: {result_text}", "info", "🔧")
    print(f"[{time.strftime('%H:%M:%S')}] 🔧 Result: {result_text}", flush=True)

    new_msg = {"role": "user", "content": f"[MCP Worker] {result_text}"}
    return {
        "last_worker_result": result_text,
        "messages": state["messages"] + [new_msg],
        "next_node": "orchestrator",
    }


async def _mcp_worker_async(instruction: str) -> str:
    """LLM picks a tool → ToolManager executes it."""
    tm = await _ensure_tool_manager()
    openai_tools = _mcp_tools_to_openai_functions(tm)
    if not openai_tools:
        return "No MCP tools available."

    llm_with_tools = _llm.bind(tools=openai_tools)
    response = llm_with_tools.invoke([
        {"role": "system", "content": MCP_WORKER_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ])

    if hasattr(response, "tool_calls") and response.tool_calls:
        results = []
        for tc in response.tool_calls:
            func_name = tc["name"]
            func_args = tc["args"]
            if "__" in func_name:
                server, tool_name = func_name.split("__", 1)
            else:
                results.append(f"Invalid tool format: {func_name}")
                continue
            logger.info("🔧 Calling %s.%s(%s)", server, tool_name, json.dumps(func_args, default=str)[:200])
            try:
                result = await tm.call_tool(server, tool_name, func_args)
                results.append(f"{server}.{tool_name}: {str(result)[:2000]}")
            except Exception as e:
                results.append(f"{server}.{tool_name} FAILED: {e}")
        return "\n".join(results)
    else:
        return f"LLM text (no tool call): {response.content[:1000]}"


# ── Code Worker Node (CodeAgent wrapper) ─────────────────────────────────────


def _get_code_agent() -> CodeAgent:
    """Lazy-init the CodeAgent using Azure engine params."""
    global _code_agent, _local_controller
    if _code_agent is None:
        _code_agent = CodeAgent(
            engine_params={
                "engine_type": "azure",
                "model": os.getenv("CODE_MODEL", os.getenv("MODEL", "gpt-4o")),
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "api_version": os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            },
            workspace=CORTEX_WORKSPACE,
            budget=20
        )
        _local_controller = LocalController()
        logger.info("CodeAgent initialized (budget=20)")
    return _code_agent


@observe(name="code_worker_node")
def code_worker_node(state: CortexState) -> dict:
    """Delegates to the existing CodeAgent which has its own multi-step iterative loop."""
    if _is_stopped():
        return {"last_worker_result": "Stopped.", "messages": state["messages"], "next_node": "orchestrator"}

    instruction = str(state.get("_instruction", ""))
    _log_fn(f"Code Agent: {instruction}", "step", "💻")
    print(f"[{time.strftime('%H:%M:%S')}] 💻 Code Agent: {instruction}", flush=True)

    agent = _get_code_agent()
    screenshot_bytes = state.get("screenshot") or _capture_screenshot_bytes()

    # Construct a comprehensive instruction for the code worker
    user_goal = state.get("task", "")
    delegated_instruction = instruction
    
    full_instruction = f"""### USER'S ULTIMATE GOAL:
{user_goal}

### YOUR CURRENT ASSIGNMENT & TIPS:
{delegated_instruction}"""

    try:
        result = agent.execute(
            task_instruction=full_instruction,
            screenshot=screenshot_bytes,
            env_controller=_local_controller,
            stop_flag=_stop_flag,
        )

        completion = result.get("completion_reason", "UNKNOWN")
        summary = result.get("summary", "No summary.")
        steps = result.get("steps_executed", 0)

        result_text = (
            f"Code Agent finished: {completion} ({steps} steps)\n"
            f"Summary:\n{summary}"
        )
    except Exception as e:
        result_text = f"Code Agent error: {e}"
        logger.error("Code Agent error: %s", e)

    _log_fn(f"Code done: {result_text}", "info", "💻")
    print(f"[{time.strftime('%H:%M:%S')}] 💻 Done: {result_text}", flush=True)

    new_msg = {"role": "user", "content": f"[Code Worker] {result_text}"}
    return {
        "last_worker_result": result_text,
        "messages": state["messages"] + [new_msg],
        "next_node": "orchestrator",
    }


# ── Graph Construction ───────────────────────────────────────────────────────


def _route(state: CortexState) -> str:
    n = state.get("next_node", "__end__")
    return n if n in ("gui_worker", "mcp_worker", "code_worker") else "__end__"


def build_graph():
    g = StateGraph(CortexState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("gui_worker", gui_worker_node)
    g.add_node("mcp_worker", mcp_worker_node)
    g.add_node("code_worker", code_worker_node)
    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", _route, {
        "gui_worker": "gui_worker",
        "mcp_worker": "mcp_worker",
        "code_worker": "code_worker",
        "__end__": END,
    })
    g.add_edge("gui_worker", "orchestrator")
    g.add_edge("mcp_worker", "orchestrator")
    g.add_edge("code_worker", "orchestrator")
    return g.compile()


# ── Public API ───────────────────────────────────────────────────────────────


class Cortex:
    """
    High-level orchestrator wrapping the LangGraph pipeline.

    Uses a SINGLE AzureChatOpenAI instance (self.llm) shared by:
      - Orchestrator (brain / routing)
      - MCP worker   (LLM + bound MCP tools)
      - Code worker  (code generation)
    The GUI worker uses its own EvoCUA model.
    """

    def __init__(
        self,
        max_steps: int = 30,
        hide_ui: Callable[[], None] = lambda: None,
        show_ui: Callable[[], None] = lambda: None,
        stop_flag: Optional[threading.Event] = None,
        log_fn: Callable[..., None] = lambda msg, *a, **kw: None,
    ):
        global _llm, _hide_ui_fn, _show_ui_fn, _stop_flag, _log_fn

        self.max_steps = max_steps

        # Wire callbacks for ws_server integration
        _hide_ui_fn = hide_ui
        _show_ui_fn = show_ui
        _stop_flag = stop_flag
        _log_fn = log_fn

        # Single shared LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=1,
            max_tokens=4096,
        )
        _llm = self.llm

        self.graph = build_graph()
        self._mcp_tool_desc = ""

        # Eagerly initialize worker agents
        log_fn("Initializing worker agents…", "info", "⚙️")
        _get_code_agent()
        _get_evocua_agent()
        
        log_fn("Cortex initialized.", "success")
        print(f"[{time.strftime('%H:%M:%S')}] Cortex initialized (LLM: {os.getenv('MODEL', 'gpt-5-mini')})", flush=True)

    async def warm_up(self):
        """Pre-initialize everything: agents, MCP tools, and connections."""
        _get_code_agent()
        _get_evocua_agent()
        await self.initialize_mcp()

    async def initialize_mcp(self):
        """Pre-connect MCP servers and cache tool descriptions."""
        try:
            # Run in the persistent bg loop so the connections stay there.
            # asyncio.to_thread prevents blocking the caller's event loop.
            tm = await asyncio.to_thread(_run_in_bg_loop, _ensure_tool_manager())
            self._mcp_tool_desc = tm.get_tools_description()
            _log_fn("MCP tools loaded.", "success", "🔧")
            print(f"[{time.strftime('%H:%M:%S')}] ✅ MCP:\n{self._mcp_tool_desc}", flush=True)
        except Exception as e:
            logger.warning("MCP init failed: %s", e)
            self._mcp_tool_desc = "MCP tools unavailable."

    @observe(name="cortex_run")
    def run(self, task: str) -> dict:
        """Run the full orchestrator loop. Returns the final state."""
        _log_fn(f"Task: {task}", "info", "📋")
        print(f"\n{'='*60}\n[{time.strftime('%H:%M:%S')}] 🚀 CORTEX: {task}\n{'='*60}", flush=True)

        initial_state: CortexState = {
            "messages": [],
            "screenshot": None,
            "working_files": [],
            "next_node": "orchestrator",
            "task": task,
            "step": 0,
            "max_steps": self.max_steps,
            "last_worker_result": "",
            "mcp_tool_descriptions": self._mcp_tool_desc,
            "orchestrator_reasoning": "",
            "_instruction": "",
        }

        final_state = self.graph.invoke(initial_state)

        _log_fn(f"Finished ({final_state.get('step', 0)} steps).", "success", "🎉")
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Done in {final_state.get('step', 0)} steps", flush=True)
        return final_state

    def shutdown(self):
        global _tool_manager, _tool_manager_connected, _bg_loop
        if _tool_manager is not None:
            try:
                _run_in_bg_loop(_tool_manager.disconnect())
            except Exception:
                pass
            _tool_manager = None
            _tool_manager_connected = False
        if _bg_loop is not None and _bg_loop.is_running():
            _bg_loop.call_soon_threadsafe(_bg_loop.stop)
