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

logger = logging.getLogger("cortex.evocua")


# ─── S2 prompt fragments (adapted from meituan/EvoCUA) ──────────────────────

_S2_ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and HOLD the specified key(s) down in order (no release). Use this for stateful holds like holding Shift while clicking.
* `key_up`: Release the specified key(s) in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
"""

_S2_DESCRIPTION_TEMPLATE = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
{resolution_info}
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""


def _build_tools_def(description_prompt: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": _S2_ACTION_DESCRIPTION,
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


_S2_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_xml}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {{"name": <function-name>, "arguments": <args-json-object>}}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""


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
        desc = _S2_DESCRIPTION_TEMPLATE.format(resolution_info=resolution_info)
        tools_def = _build_tools_def(desc)
        self.system_prompt = _S2_SYSTEM_PROMPT.format(tools_xml=json.dumps(tools_def))

        # History
        self._actions: List[str] = []
        self._responses: List[str] = []
        self._screenshots: List[str] = []  # base64

    def reset(self):
        self._actions.clear()
        self._responses.clear()
        self._screenshots.clear()

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

        # Attach screenshot + action to Langfuse span
        langfuse = get_client()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        langfuse.update_current_span(
            metadata={
                "step": len(self._actions),
                "action": pyautogui_codes[0] if pyautogui_codes else "",
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

        # Add historical turns (screenshot + response)
        if history_n > 0:
            hist_responses = self._responses[-history_n:]
            hist_imgs = self._screenshots[-history_n - 1:-1]

            for i in range(history_n):
                if i < len(hist_imgs):
                    img_url = f"data:image/png;base64,{hist_imgs[i]}"
                    if i == 0:
                        prompt = (
                            f"\nPlease generate the next move according to the "
                            f"UI screenshot, instruction and previous actions.\n\n"
                            f"Instruction: {instruction}\n\n"
                            f"Previous actions:\n{prev_str}"
                        )
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img_url}},
                                {"type": "text", "text": prompt},
                            ],
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img_url}},
                            ],
                        })

                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": hist_responses[i]}],
                })

        # Current turn
        img_url = f"data:image/png;base64,{current_img_b64}"
        if history_n == 0:
            prompt = (
                f"\nPlease generate the next move according to the "
                f"UI screenshot, instruction and previous actions.\n\n"
                f"Instruction: {instruction}\n\n"
                f"Previous actions:\n{prev_str}"
            )
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": prompt},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            })

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
                # Convert to per-character presses (EvoCUA style, avoids encoding issues)
                press_cmds = []
                for ch in text:
                    if ch == "\n":
                        press_cmds.append("pyautogui.press('enter')")
                    elif ch == "'":
                        press_cmds.append('pyautogui.press("\\\'")')
                    elif ch == "\\":
                        press_cmds.append("pyautogui.press('\\\\\\\\')")
                    else:
                        press_cmds.append(f"pyautogui.press('{ch}')")
                if press_cmds:
                    codes.append("import pyautogui; " + "; ".join(press_cmds))

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
