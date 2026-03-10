"""
EvoCUA End-to-End Agent
~~~~~~~~~~~~~~~~~~~~~~~
Wraps the EvoCUA model as an end-to-end CUA: screenshot + instruction in,
pyautogui action code out.  Uses the S2 (tool-call) prompt style from the
official meituan/EvoCUA repo.

Coordinates are output in a 0-999 normalised grid and mapped to real screen
pixels at execution time.
"""

import base64
import json
import logging
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from PIL import Image
from langfuse import observe, get_client
from openai import OpenAI

from cua_agents.v1.agents.prompts import (
    EVOCUA_ACTION_DESCRIPTION,
    EVOCUA_DESCRIPTION_TEMPLATE,
    COMPUTER_USE_GUIDELINES,
    EVOCUA_SYSTEM_PROMPT,
    GUI_SUMMARY_PROMPT,
)

logger = logging.getLogger("cortex.evocua")


# ─── S2 prompt fragments (adapted from meituan/EvoCUA) ──────────────────────




def _build_tools_def(description_prompt: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": EVOCUA_ACTION_DESCRIPTION,
                        "enum": [
                            "key", "type", "mouse_move",
                            "left_click", "left_click_drag",
                            "right_click", "middle_click",
                            "double_click", "triple_click",
                            "scroll", "wait", "terminate",
                            "key_down", "key_up",
                        ],
                        "type": "string",
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
                "type": "object",
            },
        },
    }





# ─── Image processing ───────────────────────────────────────────────────────

def _smart_resize(height: int, width: int, factor: int = 32,
                  min_pixels: int = 3136, max_pixels: int = 12845056) -> Tuple[int, int]:
    """Resize dims so both are divisible by *factor* while staying in pixel budget.
    Mirrors qwen_vl_utils.smart_resize used by EvoCUA."""
    if height < factor or width < factor:
        raise ValueError(f"Image too small: {width}x{height}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        ratio = (max_pixels / (h_bar * w_bar)) ** 0.5
        h_bar = int(round(height * ratio / factor)) * factor
        w_bar = int(round(width * ratio / factor)) * factor
    if h_bar * w_bar < min_pixels:
        ratio = (min_pixels / (h_bar * w_bar)) ** 0.5
        h_bar = int(round(height * ratio / factor)) * factor
        w_bar = int(round(width * ratio / factor)) * factor
    return h_bar, w_bar


def _process_image(image_bytes: bytes, factor: int = 32) -> Tuple[str, int, int]:
    """Resize image for the VL model and return (base64, width, height)."""
    img = Image.open(BytesIO(image_bytes))
    w, h = img.size
    new_h, new_w = _smart_resize(h, w, factor=factor)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), new_w, new_h


# ─── Agent ───────────────────────────────────────────────────────────────────

class EvoCUAAgent:
    """
    End-to-end Computer Use Agent backed by meituan/EvoCUA.

    Unlike the Worker + Grounding split, this agent handles both planning
    AND coordinate generation in a single model call.

    Args:
        base_url:       vLLM OpenAI-compatible endpoint (e.g. http://…/v1)
        model:          Model name served by vLLM
        api_key:        API key for the endpoint
        screen_size:    (width, height) of the real screen
        max_history:    Number of past turns to include in context
        temperature:    Sampling temperature
        resize_factor:  Image resize factor (32 for S2 mode)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        workspace: str,
        api_key: str = "any-value",
        screen_size: Tuple[int, int] = (1920, 1080),
        max_history: int = 4,
        temperature: float = 0.01,
        resize_factor: int = 32,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.screen_width, self.screen_height = screen_size
        self.max_history = max_history
        self.temperature = temperature
        self.resize_factor = resize_factor

        # Build the system prompt once (resolution is always 1000×1000 for relative coords)
        resolution_info = "* The screen's resolution is 1000x1000."
        desc = EVOCUA_DESCRIPTION_TEMPLATE.format(resolution_info=resolution_info)
        tools_def = _build_tools_def(desc)
        self.system_prompt = EVOCUA_SYSTEM_PROMPT.format(
            tools_xml=json.dumps(tools_def),
            computer_guidelines=COMPUTER_USE_GUIDELINES.strip(),
            workspace=workspace
        )

        # History
        self._actions: List[str] = []       # action descriptions
        self._action_codes: List[str] = []  # raw pyautogui codes
        self._responses: List[str] = []
        self._screenshots: List[str] = []   # base64
        self._loop_warning: str = ""        # injected into next prompt if loop detected

    def reset(self):
        self._actions.clear()
        self._action_codes.clear()
        self._responses.clear()
        self._screenshots.clear()
        self._loop_warning = ""

    # ── Main entry point ─────────────────────────────────────────────────

    @observe(name="evocua_predict")
    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """
        Takes a screenshot observation and returns (info_dict, [pyautogui_code]).
        Compatible with the Cortex.predict() interface.
        """
        screenshot_bytes = observation["screenshot"]

        # Resize for the VL model
        processed_b64, p_width, p_height = _process_image(
            screenshot_bytes, factor=self.resize_factor
        )
        self._screenshots.append(processed_b64)

        # Build messages
        messages = self._build_messages(instruction, processed_b64)

        # Call the model
        logger.info("Calling EvoCUA model: %s", self.model)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=self.temperature,
                top_p=0.9,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            logger.error("EvoCUA call failed: %s", e)
            return {"error": str(e)}, ["FAIL"]

        logger.info("EvoCUA raw response: %s", response_text[:500])
        print(f"[DEBUG] RAW RESPONSE:\n{response_text}\n[/DEBUG]", flush=True)
        self._responses.append(response_text)

        # Parse tool call → pyautogui code
        action_desc, pyautogui_codes = self._parse_response(response_text)
        print(f"[DEBUG] PARSED: action_desc={action_desc!r}, codes={pyautogui_codes}", flush=True)
        self._actions.append(action_desc)
        self._action_codes.append(pyautogui_codes[0] if pyautogui_codes else "")

        # ── Loop detection ────────────────────────────────────────────
        loop_info = self._detect_loop()
        if loop_info["hard_bail"]:
            print(f"[LOOP] HARD BAIL after {loop_info['repeat_count']} identical actions", flush=True)
            return {"action_description": "Loop detected — auto-terminating", "raw_response": response_text}, ["FAIL"]
        if loop_info["warning"]:
            self._loop_warning = loop_info["warning"]
            print(f"[LOOP] Warning injected: {self._loop_warning[:80]}…", flush=True)
        else:
            self._loop_warning = ""

        # Attach screenshot + action to Langfuse span
        langfuse = get_client()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        langfuse.update_current_span(
            metadata={
                "step": len(self._actions),
                "action": pyautogui_codes[0] if pyautogui_codes else "",
                "loop_warning": loop_info["warning"] or None,
            },
            input={
                "instruction": instruction,
                "screenshot": f"data:image/png;base64,{screenshot_b64}",
            },
            output=response_text,
        )

        info = {
            "action_description": action_desc,
            "raw_response": response_text,
        }
        return info, pyautogui_codes

    @observe(name="evocua_summarize")
    def summarize_progress(self, instruction: str) -> str:
        """
        Analyze the history of actions and screenshots to tell the orchestrator
        exactly what was done and what is left.
        """
        if not self._actions:
            return "No actions were performed yet."

        history_lines = []
        for i, (action, response) in enumerate(zip(self._actions, self._responses)):
            history_lines.append(f"Step {i+1}:\n- Action: {action}\n- Reasoning: {response[:300]}...")

        history_text = "\n\n".join(history_lines)
        
        prompt = f"""Original Instruction: {instruction}

Recent Action History:
{history_text}

Analyze the history above and provide a detailed status for the orchestrator."""

        try:
            # Use a slightly more capable model or the same one for summary
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GUI_SUMMARY_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            summary = completion.choices[0].message.content or "Failed to generate summary."
            return summary
        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            return f"Error generating summary: {e}. Last action: {self._actions[-1] if self._actions else 'None'}"

    # ── Loop detection ────────────────────────────────────────────────────

    def _detect_loop(self) -> dict:
        """
        Analyse recent action history for loops.
        Returns {"warning": str | "", "hard_bail": bool, "repeat_count": int}
        """
        codes = self._action_codes
        result = {"warning": "", "hard_bail": False, "repeat_count": 0}

        if len(codes) < 2:
            return result

        # --- Check 1: Exact same action code repeated N times ---
        last = codes[-1]
        streak = 1
        for c in reversed(codes[:-1]):
            if c == last:
                streak += 1
            else:
                break
        result["repeat_count"] = streak

        if streak >= 5:
            result["hard_bail"] = True
            result["warning"] = f"HARD BAIL: Same action repeated {streak} times."
            return result

        if streak >= 3:
            banned_action = self._describe_action(last)
            result["warning"] = (
                f"⚠️ LOOP DETECTED: You have repeated the same action "
                f"{streak} times in a row: \"{banned_action}\".\n"
                f"This approach is NOT working. You MUST try a completely "
                f"different action. Do NOT repeat \"{banned_action}\" again.\n"
                f"Look at the screenshot carefully and find a different "
                f"element to interact with."
            )
            return result

        # --- Check 2: Similar click coordinates (within 30px) ---
        if len(codes) >= 3:
            recent_coords = []
            for c in codes[-3:]:
                m = re.search(r'pyautogui\.(?:click|doubleClick|rightClick)\((\d+),\s*(\d+)\)', c)
                if m:
                    recent_coords.append((int(m.group(1)), int(m.group(2))))

            if len(recent_coords) == 3:
                xs = [c[0] for c in recent_coords]
                ys = [c[1] for c in recent_coords]
                x_spread = max(xs) - min(xs)
                y_spread = max(ys) - min(ys)
                if x_spread <= 30 and y_spread <= 30:
                    result["warning"] = (
                        f"⚠️ LOOP DETECTED: You have clicked on nearly the same "
                        f"spot ({recent_coords[-1]}) for the last 3 steps.\n"
                        f"The click target may be wrong. Carefully re-examine the "
                        f"screenshot and click on a DIFFERENT element."
                    )
                    return result

        # --- Check 3: Alternating pattern (A-B-A-B) ---
        if len(codes) >= 4:
            if codes[-1] == codes[-3] and codes[-2] == codes[-4] and codes[-1] != codes[-2]:
                result["warning"] = (
                    f"⚠️ LOOP DETECTED: You are alternating between two actions "
                    f"without making progress. Break the cycle by trying a "
                    f"completely different approach to accomplish the task."
                )
                return result

        return result

    @staticmethod
    def _describe_action(code: str) -> str:
        """Human-readable description of a pyautogui action code."""
        if code == "WAIT":
            return "wait"
        if code in ("DONE", "FAIL"):
            return code.lower()
        m = re.search(r'pyautogui\.(\w+)\(', code)
        if m:
            func = m.group(1)
            coord = re.search(r'\((\d+),\s*(\d+)\)', code)
            if coord:
                return f"{func} at ({coord.group(1)}, {coord.group(2)})"
            return func
        return code[:60]

    # ── Message building ─────────────────────────────────────────────────

    def _build_messages(self, instruction: str, current_img_b64: str) -> List[dict]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        ]

        step = len(self._actions)
        history_n = min(self.max_history, len(self._responses))

        # Summarise actions before the history window
        history_start = max(0, step - history_n)
        prev_actions = []
        for i in range(history_start):
            if i < len(self._actions):
                prev_actions.append(f"Step {i + 1}: {self._actions[i]}")
        prev_str = "\n".join(prev_actions) if prev_actions else "None"

        # Build loop warning suffix
        loop_suffix = ""
        if self._loop_warning:
            loop_suffix = f"\n\n{self._loop_warning}"

        # Add historical turns as text-only (model supports only 1 image per prompt)
        if history_n > 0:
            hist_responses = self._responses[-history_n:]

            for i in range(history_n):
                if i == 0:
                    prompt = (
                        f"\nPlease generate the next move according to the "
                        f"UI screenshot, instruction and previous actions.\n\n"
                        f"Instruction: {instruction}\n\n"
                        f"Previous actions:\n{prev_str}"
                    )
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "(screenshot omitted)"}],
                    })

                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": hist_responses[i]}],
                })

        # Current turn — only image in the prompt
        img_url = f"data:image/png;base64,{current_img_b64}"
        prompt_parts = []
        if history_n == 0:
            prompt_parts.append(
                f"\nPlease generate the next move according to the "
                f"UI screenshot, instruction and previous actions.\n\n"
                f"Instruction: {instruction}\n\n"
                f"Previous actions:\n{prev_str}"
            )
        if loop_suffix:
            prompt_parts.append(loop_suffix.strip())

        content: list = [{"type": "image_url", "image_url": {"url": img_url}}]
        if prompt_parts:
            content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

        messages.append({"role": "user", "content": content})

        return messages

    # ── Response parsing ─────────────────────────────────────────────────

    def _adjust_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Map from 0-999 normalised grid to real screen pixels."""
        return (
            int(x * self.screen_width / 999),
            int(y * self.screen_height / 999),
        )

    def _parse_response(self, response: str) -> Tuple[str, List[str]]:
        """Parse EvoCUA S2 response into (action_description, [pyautogui_code])."""
        codes: List[str] = []
        action_desc = ""

        if not response or not response.strip():
            return action_desc, codes

        # Strip <think>...</think> chain-of-thought blocks
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Also handle case where opening <think> is missing but </think> is present
        response = re.sub(r"^.*?</think>", "", response, flags=re.DOTALL).strip()

        # Extract action description (text before <tool_call>)
        tc_match = re.search(r"<tool_call>", response)
        if tc_match:
            action_desc = response[:tc_match.start()].strip()
            # Remove "Action:" prefix if present
            if action_desc.lower().startswith("action:"):
                action_desc = action_desc[7:].strip()

        # Find all tool_call blocks
        tool_calls = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL
        )

        for tc_json in tool_calls:
            try:
                tc = json.loads(tc_json)
            except json.JSONDecodeError:
                # Try to fix common issues (single quotes, trailing commas)
                try:
                    fixed = tc_json.replace("'", '"')
                    fixed = re.sub(r",\s*}", "}", fixed)
                    fixed = re.sub(r",\s*]", "]", fixed)
                    tc = json.loads(fixed)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool_call JSON: %s", tc_json[:200])
                    continue

            if tc.get("name") != "computer_use":
                continue

            args = tc.get("arguments", {})
            action = args.get("action", "")

            if action in ("left_click", "click"):
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    ax, ay = self._adjust_coords(x, y)
                    codes.append(f"import pyautogui; pyautogui.click({ax}, {ay})")
                else:
                    codes.append("import pyautogui; pyautogui.click()")

            elif action == "right_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    ax, ay = self._adjust_coords(x, y)
                    codes.append(f"import pyautogui; pyautogui.rightClick({ax}, {ay})")
                else:
                    codes.append("import pyautogui; pyautogui.rightClick()")

            elif action == "double_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    ax, ay = self._adjust_coords(x, y)
                    codes.append(f"import pyautogui; pyautogui.doubleClick({ax}, {ay})")
                else:
                    codes.append("import pyautogui; pyautogui.doubleClick()")

            elif action == "triple_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    ax, ay = self._adjust_coords(x, y)
                    codes.append(f"import pyautogui; pyautogui.tripleClick({ax}, {ay})")
                else:
                    codes.append("import pyautogui; pyautogui.tripleClick()")

            elif action == "middle_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    ax, ay = self._adjust_coords(x, y)
                    codes.append(f"import pyautogui; pyautogui.middleClick({ax}, {ay})")
                else:
                    codes.append("import pyautogui; pyautogui.middleClick()")

            elif action == "mouse_move":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    ax, ay = self._adjust_coords(x, y)
                    codes.append(f"import pyautogui; pyautogui.moveTo({ax}, {ay})")

            elif action == "left_click_drag":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    ax, ay = self._adjust_coords(x, y)
                    codes.append(f"import pyautogui; pyautogui.dragTo({ax}, {ay})")

            elif action == "type":
                text = args.get("text", "")
                if text:
                    # Safely escape text using json.dumps for the exec() string
                    # pyautogui.write handles newlines by pressing enter automatically
                    codes.append(f"import pyautogui; pyautogui.write({json.dumps(text)})")

            elif action == "key":
                keys = args.get("keys", [])
                if isinstance(keys, str):
                    keys = [keys]
                keys = [k.strip() for k in keys if isinstance(k, str)]
                if len(keys) > 1:
                    keys_str = ", ".join(f"'{k}'" for k in keys)
                    codes.append(f"import pyautogui; pyautogui.hotkey({keys_str})")
                elif keys:
                    codes.append(f"import pyautogui; pyautogui.press('{keys[0]}')")

            elif action == "key_down":
                keys = args.get("keys", [])
                if isinstance(keys, str):
                    keys = [keys]
                for k in keys:
                    codes.append(f"import pyautogui; pyautogui.keyDown('{k}')")

            elif action == "key_up":
                keys = args.get("keys", [])
                if isinstance(keys, str):
                    keys = [keys]
                for k in reversed(keys):
                    codes.append(f"import pyautogui; pyautogui.keyUp('{k}')")

            elif action == "scroll":
                pixels = args.get("pixels", 0)
                codes.append(f"import pyautogui; pyautogui.scroll({pixels})")

            elif action == "wait":
                codes.append("WAIT")

            elif action == "terminate":
                status = args.get("status", "success")
                codes.append("DONE" if str(status).lower() != "failure" else "FAIL")

        # Fallback if nothing parsed
        if not codes:
            logger.warning("No actions parsed from response, returning FAIL")
            codes = ["FAIL"]

        return action_desc, codes
