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
* `scroll`: Performs a scroll of the mouse scroll wheel. Use positive values to scroll up and negative values to scroll down.
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

# General computer-use knowledge injected into the system prompt
_COMPUTER_USE_GUIDELINES = """
# Computer Use Guidelines

## Desktop & App Launching
- On Windows, desktop icons require DOUBLE-CLICK to open. Single-clicking only selects them.
- Taskbar icons (bottom bar) require a single left-click to open or switch to an app.
- To open an app not on the desktop: click the Windows Start button (bottom-left) or press the Windows key, then type the app name to search for it.
- If an app is minimized, click its icon in the taskbar to restore it.
- If an app is behind another window, click its taskbar icon to bring it to the front.
- After launching an application, WAIT at least 2-3 seconds for it to fully load before interacting with it.

## Windows Search
- Press the Windows key to open Start Menu, then type to search for apps, files, or settings.
- The Windows search bar may also be visible in the taskbar — click it and type.
- After typing a search query, wait briefly for results to appear, then click the appropriate result.

## Keyboard Shortcuts (Windows)
- Alt+Tab: Switch between open windows.
- Alt+F4: Close the current window.
- Ctrl+C / Ctrl+V: Copy / Paste.
- Ctrl+A: Select all text.
- Ctrl+Z / Ctrl+Y: Undo / Redo.
- Ctrl+W: Close the current tab (in browsers and many apps).
- Ctrl+T: Open a new tab (in browsers).
- Ctrl+L or F6: Focus the address/URL bar in browsers.
- Win+D: Show desktop (minimize all windows).
- Win+E: Open File Explorer.
- Enter: Confirm/submit the current action (press a button, submit a form, open a selected item).
- Escape: Cancel/close the current dialog or popup.

## Browser Usage
- To navigate to a URL: click the address bar (or press Ctrl+L), clear it, type the URL, and press Enter.
- To search: click the address bar, type your query, and press Enter.
- Use Ctrl+T for a new tab, Ctrl+W to close the current tab.
- To go back: click the back arrow or press Alt+Left.
- Bookmarks and pinned sites may appear on the new tab page.
- Web pages may take a few seconds to load — wait before interacting with page elements.
- If a page element isn't responding to clicks, try scrolling to make it fully visible first.

## Text Input
- Before typing, you MUST click on the text field to focus it (the cursor should be blinking in the field).
- To clear an existing text field: triple-click to select all text, then type the new text.
- Alternatively, use Ctrl+A to select all text in a focused field, then type to replace.
- For search bars, usually clicking and typing directly works; the previous text gets replaced.

## Scroll Behavior
- Use positive scroll values to scroll UP (content moves down) and negative values to scroll DOWN (content moves up).
- Many apps and web pages require scrolling to see all content.
- If a button or element is not visible, try scrolling down to find it.

## File Management
- Right-click on the desktop or in File Explorer for context menus (New, Open, Properties, etc.).
- Double-click folders to open them in File Explorer.
- File paths on Windows use backslashes: C:\\Users\\...

## Common Pitfalls to Avoid
- Do NOT try to interact with elements that are behind other windows — bring the target window to the front first.
- Do NOT keep waiting (action=wait) if nothing is changing — try a different approach.
- Do NOT click on the same unresponsive element repeatedly — try a different way to accomplish the goal.
- If a dialog/popup appears, handle it first before continuing with the main task.
- If you see an error message, read it carefully before deciding the next action.
- Never assume a task is done without visual confirmation on the screen.
- If you are unsure of the name of any app, either hover over it and check the name or use the search bar in the start menu to search for it.
"""


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

{computer_guidelines}

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {{"name": <function-name>, "arguments": <args-json-object>}}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call.
- Verify each action visually before moving to the next step.
- If an action did not produce the expected result, try an alternative approach."""


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
        self.system_prompt = _S2_SYSTEM_PROMPT.format(
            tools_xml=json.dumps(tools_def),
            computer_guidelines=_COMPUTER_USE_GUIDELINES.strip(),
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

        # Current turn — inject loop warning if present
        img_url = f"data:image/png;base64,{current_img_b64}"
        if history_n == 0:
            prompt = (
                f"\nPlease generate the next move according to the "
                f"UI screenshot, instruction and previous actions.\n\n"
                f"Instruction: {instruction}\n\n"
                f"Previous actions:\n{prev_str}"
                f"{loop_suffix}"
            )
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": prompt},
                ],
            })
        else:
            if loop_suffix:
                # Need to add text alongside the image
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": loop_suffix.strip()},
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
