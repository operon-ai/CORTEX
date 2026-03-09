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
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

from cua_agents.v1.agents.code_agent import CodeAgent
from cua_agents.v1.agents.evocua_agent import EvoCUAAgent
from cua_agents.v1.agents.infra_agent import INFRA_TOOLS, handle_infra_tool
from cua_agents.v1.agents.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    MCP_WORKER_SYSTEM_PROMPT,
    INFRA_WORKER_SYSTEM_PROMPT,
)
from cua_agents.v1.tools.tool_manager import ToolManager
from cua_agents.v1.utils.local_env import LocalController

load_dotenv()

# Central Workspace for all file operations
CORTEX_WORKSPACE = os.path.join(os.path.expanduser("~"), "Desktop")
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
    # For VSCode AI coding tasks: the orchestrator crafts the AI prompt separately
    # from the GUI navigation instruction. gui_worker will type this into the chatbox.
    vscode_prompt: str
    # Slack context extracted from messages (channel, requester) for reporting back
    slack_reply_context: str
    # Todo tracker — live checklist of sub-steps
    todo_list: List[Dict[str, Any]]  # [{"id": 0, "text": "...", "status": "pending"|"in_progress"|"done"|"failed"}]
    current_todo_index: int


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
_todo_fn: Callable[..., None] = lambda items: None


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

    # Build context block with task + step + todo state
    context_parts = [f"Task: {state['task']}", f"Step: {state['step']}/{state['max_steps']}"]
    todo_list = state.get("todo_list", [])
    if todo_list:
        todo_text = "\n\nCurrent TODO checklist:"
        for item in todo_list:
            icon = {"done": "✅", "in_progress": "🔄", "failed": "❌"}.get(item["status"], "⬜")
            todo_text += f"\n  {icon} {item['id']+1}. {item['text']} [{item['status']}]"
        context_parts.append(todo_text)
    else:
        context_parts.append("\n\nThis is your FIRST step. You MUST include a \"todo_list\" in your response — decompose the user's task into 3-7 concrete sub-steps.")

    messages.append({"role": "user", "content": "\n".join(context_parts)})

    for msg in state["messages"][-10:]:
        messages.append(msg)
    if state.get("screenshot"):
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
            "content": f"Last worker result:\n{state['last_worker_result']}\n\nEvaluate: did the previous step succeed? Set todo_evaluation to 'pass', 'fail', or 'skip'."
        })
    return messages


@observe(name="orchestrator_node")
def orchestrator_node(state: CortexState) -> dict:
    """The brain — decides what worker to call next."""
    step = state["step"]
    max_s = state["max_steps"]
    logger.info(" Orchestrator (step %d/%d)", step, max_s)
    _log_fn(f"Step {step + 1}/{max_s}: Thinking…", "step", "")

    if _is_stopped():
        _log_fn("Stopped by user.", "warning", "")
        return {"next_node": "__end__", "orchestrator_reasoning": "Stopped by user.", "step": step + 1}

    if step >= max_s:
        _log_fn("Step budget exhausted.", "warning", "")
        return {"next_node": "__end__", "orchestrator_reasoning": "Budget exhausted.", "step": step + 1}

    _log_fn("Capturing screen…", "step", "")
    screenshot_bytes = _capture_screenshot_bytes()

    updated_state = {**state, "screenshot": screenshot_bytes}
    messages = _build_orchestrator_messages(updated_state)

    try:
        response = _llm.invoke(messages)
        raw_text = response.content
        print(f"[{time.strftime('%H:%M:%S')}]  raw: {raw_text}", flush=True)

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
    vscode_prompt = decision.get("vscode_prompt", "")
    slack_reply_context = decision.get("slack_reply_context", state.get("slack_reply_context", ""))

    # ── Todo list management ──────────────────────────────────────────────
    todo_list = list(state.get("todo_list", []))
    current_idx = state.get("current_todo_index", 0)

    # First step or whenever LLM returns a full todo list structure
    if "todo_list" in decision:
        raw_todos = decision["todo_list"]
        
        # If this is the initial creation and we just have strings, convert to dicts
        # BUT if LLM provides full structured dicts with subtasks, preserve them.
        new_list = []
        for i, t in enumerate(raw_todos):
            if isinstance(t, dict):
                item = {
                    "id": t.get("id", i), 
                    "text": str(t.get("text", "")), 
                    "status": t.get("status", "pending")
                }
                if "subtasks" in t:
                    item["subtasks"] = t["subtasks"]
                new_list.append(item)
            else:
                new_list.append({"id": i, "text": str(t), "status": "pending"})
                
        if step == 0 and new_list:
            new_list[0]["status"] = "in_progress"
            
        todo_list = new_list
        current_idx = decision.get("current_todo_index", 0)
        
        if step == 0:
            _log_fn(f" Plan: {len(todo_list)} steps", "info", "")
            
        # Send flattened/top-level only to UI so JS doesn't break
        _todo_fn(todo_list)

    # Subsequent steps: evaluate previous step and advance IF todo_list wasn't just completely rewritten
    elif todo_list and step > 0:
        evaluation = decision.get("todo_evaluation", "pass")
        if current_idx < len(todo_list):
            if evaluation == "fail":
                todo_list[current_idx]["status"] = "failed"
                _log_fn(f"❌ Step {current_idx+1} failed: {todo_list[current_idx]['text']}", "warning", "")
            else:  # pass or skip
                todo_list[current_idx]["status"] = "done"
                _log_fn(f"✅ Step {current_idx+1} done: {todo_list[current_idx]['text']}", "success", "")

        # Advance to next pending todo
        new_idx = decision.get("current_todo_index", current_idx + 1)
        if isinstance(new_idx, int) and 0 <= new_idx < len(todo_list):
            current_idx = new_idx
        else:
            current_idx = min(current_idx + 1, len(todo_list) - 1)

        if current_idx < len(todo_list) and todo_list[current_idx]["status"] == "pending":
            todo_list[current_idx]["status"] = "in_progress"

        _todo_fn(todo_list)

    # If ending, mark all top-level remaining as done (if no failures)
    if next_node == "__end__" and todo_list:
        for item in todo_list:
            if item["status"] in ("pending", "in_progress"):
                item["status"] = "done"
        _todo_fn(todo_list)

    str_instruction = str(instruction)
    _log_fn(f"→ {next_node}: {str_instruction}", "step", "")
    print(f"[{time.strftime('%H:%M:%S')}]  → {next_node}: {str_instruction}", flush=True)
    if vscode_prompt:
        print(f"[{time.strftime('%H:%M:%S')}]  vscode_prompt crafted ({len(vscode_prompt)} chars)", flush=True)

    langfuse = get_client()
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
        "vscode_prompt": vscode_prompt,
        "slack_reply_context": slack_reply_context,
        "todo_list": todo_list,
        "current_todo_index": current_idx,
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


def _is_vscode_chatbox_task(instruction: str) -> bool:
    """Detect if the GUI instruction is to navigate to a VSCode AI chatbox."""
    keywords = ["vscode", "vs code", "copilot chat", "ai chat", "chatbox", "chat panel",
                 "chat input", "open chat", "agent chat", "chat box"]
    lower = instruction.lower()
    return any(kw in lower for kw in keywords)


@observe(name="gui_worker_node")
def gui_worker_node(state: CortexState) -> dict:
    if _is_stopped():
        return {"last_worker_result": "Stopped.", "messages": state["messages"], "next_node": "orchestrator"}

    instruction = str(state.get("_instruction", state.get("task", "")))
    vscode_prompt = state.get("vscode_prompt", "")
    _log_fn(f"GUI Task: {instruction}", "step", "")
    print(f"[{time.strftime('%H:%M:%S')}]  GUI: {instruction}", flush=True)

    # If this is a VSCode chatbox task and the orchestrator supplied a vscode_prompt,
    # embed it into the navigation instruction so the GUI agent knows exactly what to type.
    # The GUI agent's sole job: navigate to the chatbox and submit this prompt.
    if vscode_prompt and _is_vscode_chatbox_task(instruction):
        _log_fn("VSCode chatbox task detected — injecting orchestrator prompt.", "info", "")
        print(f"[{time.strftime('%H:%M:%S')}]  Injecting vscode_prompt into GUI instruction", flush=True)
        instruction = (
            f"{instruction}\n\n"
            f"IMPORTANT: Once the AI chat input box is focused and ready, type the following "
            f"prompt EXACTLY as written (do not paraphrase or abbreviate), then press Enter to submit:\n\n"
            f"--- BEGIN PROMPT ---\n"
            f"{vscode_prompt}\n"
            f"--- END PROMPT ---\n\n"
            f"After submitting, wait for the AI to finish responding (watch for the response to stop "
            f"streaming), then take a final screenshot capturing the full response."
        )

    agent = _get_evocua_agent()
    agent.reset()  # Start fresh history for this delegation

    step_count = 0
    max_sub_steps = 30  # Internal budget to prevent infinite loops
    last_result = "Delegated to GUI worker."
    current_messages = list(state["messages"])
    original_instruction = str(state.get("_instruction", ""))

    while step_count < max_sub_steps:
        if _is_stopped():
            last_result = "Stopped by user."
            break

        _log_fn(f"GUI ({step_count+1}/{max_sub_steps}): Thinking…", "step", "")
        screenshot_bytes = _capture_screenshot_bytes()

        info, codes = agent.predict(instruction=instruction, observation={"screenshot": screenshot_bytes})
        action = codes[0] if codes else "WAIT"

        desc = str(info.get("action_description", action))
        _log_fn(f"Action: {desc}", "info", "")
        print(f"[{time.strftime('%H:%M:%S')}]  GUI Step {step_count+1}: {desc}", flush=True)

        if action == "DONE":
            # Capture a final screenshot to report back (useful for Slack reporting)
            final_screenshot_bytes = _capture_screenshot_bytes()
            final_screenshot_b64 = base64.b64encode(final_screenshot_bytes).decode("utf-8")
            last_result = f"GUI Task Completed: {desc}"
            if vscode_prompt and _is_vscode_chatbox_task(original_instruction):
                last_result += f"\n[SCREENSHOT_B64:{final_screenshot_b64}]"
            _log_fn("GUI: Done.", "success", "")
            break
        if action == "FAIL":
            last_result = f"GUI Task Failed: {desc}"
            _log_fn("GUI: Failed.", "error", "")
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
            break  # Exit loop on hard crash

        last_result = result_text
        step_count += 1

    if step_count >= max_sub_steps:
        _log_fn("Budget hit. Summarizing progress…", "info", "")
        summary = agent.summarize_progress(instruction)
        # Capture screenshot even on budget exhaustion for VSCode tasks
        if vscode_prompt and _is_vscode_chatbox_task(original_instruction):
            final_screenshot_bytes = _capture_screenshot_bytes()
            final_screenshot_b64 = base64.b64encode(final_screenshot_bytes).decode("utf-8")
            last_result = f"GUI budget reached. Status: {summary}\n[SCREENSHOT_B64:{final_screenshot_b64}]"
        else:
            last_result = f"GUI budget reached. Status: {summary}"
        _log_fn("GUI: Detailed status reported.", "warning", "")

    new_msg = {"role": "user", "content": f"[GUI Worker] {last_result}"}
    return {
        "last_worker_result": last_result,
        "messages": current_messages + [new_msg],
        "next_node": "orchestrator",
        # Clear vscode_prompt after use so it isn't re-injected on the next GUI call
        "vscode_prompt": "",
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
    slack_team_id = os.getenv("SLACK_TEAM_ID")
    slack_app_token = os.getenv("SLACK_APP_TOKEN")
    if slack_bot_token and slack_team_id:
        slack_env = {
            "SLACK_BOT_TOKEN": slack_bot_token,
            "SLACK_TEAM_ID": slack_team_id,
        }
        if slack_app_token:
            slack_env["SLACK_APP_TOKEN"] = slack_app_token
        # Pass through optional xoxp token and add-message flag
        xoxp = os.getenv("SLACK_MCP_XOXP_TOKEN")
        if xoxp:
            slack_env["SLACK_MCP_XOXP_TOKEN"] = xoxp
        if os.getenv("SLACK_MCP_ADD_MESSAGE_TOOL"):
            slack_env["SLACK_MCP_ADD_MESSAGE_TOOL"] = os.getenv("SLACK_MCP_ADD_MESSAGE_TOOL")

        config["slack"] = {
            "command": npm_cmd,
            "args": ["-y", "@modelcontextprotocol/server-slack"],
            "env": slack_env,
        }
    elif slack_bot_token and not slack_team_id:
        logger.warning("SLACK_BOT_TOKEN set but SLACK_TEAM_ID missing — Slack MCP disabled")
    return config


async def _ensure_tool_manager() -> ToolManager:
    global _tool_manager, _tool_manager_connected
    if _tool_manager is None:
        _tool_manager = ToolManager(_get_mcp_config())
    if not _tool_manager_connected:
        await _tool_manager.connect()
        _tool_manager_connected = True
    return _tool_manager


_GEMINI_UNSUPPORTED_KEYS = {
    "$defs", "definitions", "$schema", "$comment", "$id",
    "additionalProperties", "default", "examples", "title",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "if", "then", "else", "not", "allOf",
}
_GEMINI_VALID_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def _sanitize_schema(schema: Any, root_schema: Any = None) -> Any:
    """Recursively sanitize JSON schema for Gemini's strict function-calling requirements."""
    import copy
    if root_schema is None:
        root_schema = schema

    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_sanitize_schema(item, root_schema) for item in schema]
        return schema

    s = copy.deepcopy(schema)

    # Resolve $ref inline
    if "$ref" in s:
        ref_path = s["$ref"]
        if ref_path.startswith("#/"):
            parts = ref_path.split("/")[1:]
            resolved = root_schema
            for p in parts:
                if isinstance(resolved, dict) and p in resolved:
                    resolved = resolved[p]
                else:
                    break
            else:
                return _sanitize_schema(resolved, root_schema)
        return {"type": "string"}  # unresolvable ref → fallback

    # Collapse oneOf/anyOf → pick first non-null concrete candidate
    for kw in ("oneOf", "anyOf"):
        if kw in s and isinstance(s[kw], list) and s[kw]:
            candidates = [c for c in s[kw] if not (isinstance(c, dict) and c.get("type") == "null")]
            chosen = candidates[0] if candidates else s[kw][0]
            return _sanitize_schema(chosen, root_schema)

    # Handle const → minimal typed schema
    if "const" in s:
        val = s["const"]
        t = "string" if isinstance(val, str) else "integer" if isinstance(val, int) else "boolean" if isinstance(val, bool) else "string"
        return {"type": t, "description": f"Must be {val}"}

    # Remove None values and all unsupported keys
    for k in list(s.keys()):
        if s[k] is None or k in _GEMINI_UNSUPPORTED_KEYS:
            del s[k]

    # Fix type if it's a list → pick first non-null
    if "type" in s and isinstance(s["type"], list):
        types = [t for t in s["type"] if t != "null"]
        s["type"] = types[0] if types else "string"

    # Ensure type is a valid Gemini string
    if "type" in s:
        st = s["type"]
        if not isinstance(st, str) or st not in _GEMINI_VALID_TYPES:
            del s["type"]

    # Infer object type from properties
    if "properties" in s and "type" not in s:
        s["type"] = "object"

    # Convert open-ended object types (type=object with no inner properties)
    # to type=string.  This prevents Pydantic crashes in langchain-google-genai
    # when a tool parameter is literally named "properties" — the Gemini SDK's
    # Schema model confuses user parameter names with JSON Schema keywords.
    if s.get("type") == "object" and "properties" not in s:
        desc = s.get("description", "")
        s["type"] = "string"
        s["description"] = f"{desc} (JSON string)".strip()

    # Recurse into properties values (each value is a sub-schema)
    if "properties" in s and isinstance(s["properties"], dict):
        new_props = {}
        for pk, pv in s["properties"].items():
            if isinstance(pv, str):
                new_props[pk] = {"type": pv} if pv in _GEMINI_VALID_TYPES else {"type": "string"}
            elif isinstance(pv, dict):
                new_props[pk] = _sanitize_schema(pv, root_schema)
            else:
                new_props[pk] = {"type": "string"}
        s["properties"] = new_props

    # Recurse into items (array item schema)
    if "items" in s and isinstance(s["items"], dict):
        s["items"] = _sanitize_schema(s["items"], root_schema)
    elif "items" in s and isinstance(s["items"], list):
        s["items"] = [_sanitize_schema(item, root_schema) for item in s["items"]]

    # Empty schema → default to string
    if not s:
        return {"type": "string"}

    # Strip required entries whose properties don't exist
    if "required" in s and isinstance(s["required"], list):
        props = set(s.get("properties", {}).keys())
        s["required"] = [r for r in s["required"] if isinstance(r, str) and r in props]
        if not s["required"]:
            del s["required"]

    return s


def _deep_clean_required(obj: Any) -> None:
    """Post-pass: recursively remove required entries that don't exist in properties."""
    if isinstance(obj, dict):
        if "required" in obj and isinstance(obj["required"], list):
            props = set(obj.get("properties", {}).keys())
            obj["required"] = [r for r in obj["required"] if r in props]
            if not obj["required"]:
                del obj["required"]
        for v in list(obj.values()):
            _deep_clean_required(v)
    elif isinstance(obj, list):
        for item in obj:
            _deep_clean_required(item)


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
        _deep_clean_required(input_schema)

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
    _log_fn(f"MCP: {instruction}", "step", "")
    print(f"[{time.strftime('%H:%M:%S')}]  MCP: {instruction}", flush=True)

    langfuse_handler = CallbackHandler()

    try:
        result_text = _run_in_bg_loop(_mcp_worker_async(instruction, callbacks=[langfuse_handler]))
    except Exception as e:
        result_text = f"MCP error: {e}"
        logger.error("MCP error: %s", e)

    _log_fn(f"MCP result: {result_text}", "info", "")
    print(f"[{time.strftime('%H:%M:%S')}]  Result: {result_text}", flush=True)

    new_msg = {"role": "user", "content": f"[MCP Worker] {result_text}"}
    return {
        "last_worker_result": result_text,
        "messages": state["messages"] + [new_msg],
        "next_node": "orchestrator",
    }

async def _mcp_worker_async(instruction: str, callbacks: list = None) -> str:
    """LLM picks tools → ToolManager executes them in a JIT loop."""
    tm = await _ensure_tool_manager()
    openai_tools = _mcp_tools_to_openai_functions(tm)
    if not openai_tools:
        return "No MCP tools available."

    llm_with_tools = _llm.bind(tools=openai_tools)
    config = {"callbacks": callbacks} if callbacks else {}
    messages = [
        {"role": "system", "content": MCP_WORKER_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    for step in range(15):  # Safety limit for JIT loop
        print(f"JIT Loop Step {step}: Invoking LLM with {len(messages)} messages...", flush=True)
        response = llm_with_tools.invoke(messages, config=config)
        print(f"JIT Loop Step {step}: LLM returned.", flush=True)
        messages.append(response)

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            print(f"JIT Loop Step {step}: No tool calls, finishing.", flush=True)
            # The LLM decided it has finished its task and responded with text
            if isinstance(response.content, list):
                text_parts = []
                for part in response.content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "\n".join(text_parts).strip()
            return str(response.content)

        # Execute tools
        print(f"JIT Loop Step {step}: Executing {len(response.tool_calls)} tools...", flush=True)
        for tc in response.tool_calls:
            func_name = tc["name"]
            func_args = tc["args"]
            tool_call_id = tc["id"]
            
            if "__" in func_name:
                server, tool_name = func_name.split("__", 1)
            else:
                messages.append(ToolMessage(
                    tool_call_id=tool_call_id,
                    name=func_name,
                    content="Error: Invalid tool format. Expected server__tool_name."
                ))
                continue
                
            logger.info(" Calling %s.%s(%s)", server, tool_name, json.dumps(func_args, default=str)[:200])
            try:
                result = await tm.call_tool(server, tool_name, func_args)
                result_str = str(result)
            except Exception as e:
                logger.error(" Tool Failed: %s", e)
                result_str = f"Error executing tool: {e}"
                
            messages.append(ToolMessage(
                tool_call_id=tool_call_id,
                name=func_name,
                content=result_str
            ))
        print(f"JIT Loop Step {step}: Finished executing tools, looping...", flush=True)

    return "MCP Worker JIT Loop hit maximum steps without finishing."


# ── Infra Worker Node (Terminal + UI Tools) ──────────────────────────────────


@observe(name="infra_worker_node")
def infra_worker_node(state: CortexState) -> dict:
    """Executes infrastructure tasks: terminal commands, UI tree inspection, direct UI interaction."""
    if _is_stopped():
        return {"last_worker_result": "Stopped.", "messages": state["messages"], "next_node": "orchestrator"}

    instruction = str(state.get("_instruction", ""))
    vscode_prompt = state.get("vscode_prompt", "")
    _log_fn(f"Infra: {instruction}", "step", "")
    print(f"[{time.strftime('%H:%M:%S')}]  Infra: {instruction}", flush=True)

    # Detect VSCode chatbox tasks and inject the prompt for the infra worker
    if vscode_prompt and _is_vscode_chatbox_task(instruction):
        _log_fn("VSCode chatbox task detected (Infra) — injecting orchestrator prompt.", "info", "")
        print(f"[{time.strftime('%H:%M:%S')}]  Injecting vscode_prompt into Infra instruction", flush=True)
        instruction = (
            f"{instruction}\n\n"
            f"IMPORTANT: Use the 'Message input' element. Type the following "
            f"prompt EXACTLY as written, then press Enter:\n\n"
            f"--- BEGIN PROMPT ---\n"
            f"{vscode_prompt}\n"
            f"--- END PROMPT ---\n\n"
            f"After typing, verify that the AI in VS Code has started responding."
        )

    llm_with_tools = _llm.bind(tools=INFRA_TOOLS)
    langfuse_handler = CallbackHandler()
    config = {"callbacks": [langfuse_handler]}
    messages = [
        {"role": "system", "content": INFRA_WORKER_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    result_text = "Infra worker completed."
    try:
        for step in range(15):
            response = llm_with_tools.invoke(messages, config=config)
            messages.append(response)

            if not hasattr(response, "tool_calls") or not response.tool_calls:
                if isinstance(response.content, list):
                    text_parts = []
                    for part in response.content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    result_text = "\n".join(text_parts).strip()
                else:
                    result_text = str(response.content)
                break

            for tc in response.tool_calls:
                func_name = tc["name"]
                func_args = tc["args"]
                tool_call_id = tc["id"]
                print(f"[{time.strftime('%H:%M:%S')}]  Tool: {func_name}", flush=True)

                try:
                    tool_result = handle_infra_tool(func_name, func_args)
                except Exception as e:
                    tool_result = json.dumps({"error": str(e)})

                messages.append(ToolMessage(
                    tool_call_id=tool_call_id,
                    name=func_name,
                    content=tool_result,
                ))
        else:
            result_text = "Infra worker hit maximum steps."
    except Exception as e:
        result_text = f"Infra error: {e}"
        logger.error("Infra error: %s", e)

    _log_fn(f"Infra result: {result_text}", "info", "")
    print(f"[{time.strftime('%H:%M:%S')}]  Done: {result_text}", flush=True)

    new_msg = {"role": "user", "content": f"[Infra Worker] {result_text}"}
    return {
        "last_worker_result": result_text,
        "messages": state["messages"] + [new_msg],
        "next_node": "orchestrator",
        "vscode_prompt": "",
    }


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
    _log_fn(f"Code Agent: {instruction}", "step", "")
    print(f"[{time.strftime('%H:%M:%S')}]  Code Agent: {instruction}", flush=True)

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

    _log_fn(f"Code done: {result_text}", "info", "")
    print(f"[{time.strftime('%H:%M:%S')}]  Done: {result_text}", flush=True)

    new_msg = {"role": "user", "content": f"[Code Worker] {result_text}"}
    return {
        "last_worker_result": result_text,
        "messages": state["messages"] + [new_msg],
        "next_node": "orchestrator",
    }


# ── Graph Construction ───────────────────────────────────────────────────────


def _route(state: CortexState) -> str:
    n = state.get("next_node", "__end__")
    return n if n in ("gui_worker", "mcp_worker", "code_worker", "infra_worker") else "__end__"


def build_graph():
    g = StateGraph(CortexState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("gui_worker", gui_worker_node)
    g.add_node("mcp_worker", mcp_worker_node)
    g.add_node("code_worker", code_worker_node)
    g.add_node("infra_worker", infra_worker_node)
    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", _route, {
        "gui_worker": "gui_worker",
        "mcp_worker": "mcp_worker",
        "code_worker": "code_worker",
        "infra_worker": "infra_worker",
        "__end__": END,
    })
    g.add_edge("gui_worker", "orchestrator")
    g.add_edge("mcp_worker", "orchestrator")
    g.add_edge("code_worker", "orchestrator")
    g.add_edge("infra_worker", "orchestrator")
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
        todo_fn: Callable[..., None] = lambda items: None,
    ):
        global _llm, _hide_ui_fn, _show_ui_fn, _stop_flag, _log_fn, _todo_fn

        self.max_steps = max_steps

        # Wire callbacks for ws_server integration
        _hide_ui_fn = hide_ui
        _show_ui_fn = show_ui
        _stop_flag = stop_flag
        _log_fn = log_fn
        _todo_fn = todo_fn

        # Single shared LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
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
            _log_fn("MCP tools loaded.", "success", "")
            print(f"[{time.strftime('%H:%M:%S')}] ✅ MCP:\n{self._mcp_tool_desc}", flush=True)
        except Exception as e:
            logger.warning("MCP init failed: %s", e)
            self._mcp_tool_desc = "MCP tools unavailable."

    @observe(name="cortex_run")
    def run(self, task: str) -> dict:
        """Run the full orchestrator loop. Returns the final state."""
        _log_fn(f"Task: {task}", "info", "")
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
            "vscode_prompt": "",
            "slack_reply_context": "",
            "todo_list": [],
            "current_todo_index": 0,
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
