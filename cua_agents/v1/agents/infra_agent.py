"""
Infra Agent — Terminal + Desktop UI Interaction Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides the orchestrator with direct access to:
  • Terminal commands (subprocess execution on the host)
  • UI tree inspection (pywinauto UIA backend)
  • Type into UI element (inject text into named inputs)
  • Click UI element (click named buttons / controls)
"""

import json
import os
import subprocess
import sys
import time

# ── Tool Schemas (OpenAI function-calling format) ────────────────────────────

INFRA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "host_terminal__run_command",
            "description": (
                "Execute a shell command on the host machine. "
                "The working directory defaults to the user's Desktop. "
                "Returns stdout, stderr, and exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Working directory for the command. "
                            "Defaults to the user's Desktop folder."
                        )
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60)."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "host_ui_controller__get_ui_tree",
            "description": (
                "Walk the Windows UI Automation element tree and return interactive "
                "elements (buttons, inputs, text fields, etc.) as JSON. "
                "Use this to discover element names for type_into_element / click_element."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window_title": {
                        "type": "string",
                        "description": (
                            "Substring of the target window title. "
                            "If omitted, inspects the foreground window."
                        )
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to walk the UI tree (default: 50)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "host_ui_controller__type_into_element",
            "description": (
                "Type text directly into a named UI element (e.g. an input box, "
                "terminal, or chat field) without needing vision. "
                "Use get_ui_tree first to discover the element name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to type into the element."
                    },
                    "element_name": {
                        "type": "string",
                        "description": (
                            "The name (or substring) of the UI element to type into. "
                            "Matched case-insensitively against the element's Name property."
                        )
                    },
                    "window_title": {
                        "type": "string",
                        "description": "Substring of the target window title."
                    },
                    "press_enter": {
                        "type": "boolean",
                        "description": "Whether to press Enter after typing (default: true)."
                    },
                    "clear_first": {
                        "type": "boolean",
                        "description": "Whether to clear the field before typing (default: true)."
                    }
                },
                "required": ["text", "element_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "host_ui_controller__click_element",
            "description": (
                "Click a named UI element (button, link, menu item, etc.). "
                "Use get_ui_tree first to discover the element name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "element_name": {
                        "type": "string",
                        "description": (
                            "The name (or substring) of the UI element to click. "
                            "Matched case-insensitively."
                        )
                    },
                    "window_title": {
                        "type": "string",
                        "description": "Substring of the target window title."
                    }
                },
                "required": ["element_name"]
            }
        }
    },
]


# ── Tool Dispatcher ──────────────────────────────────────────────────────────

def handle_infra_tool(name: str, args: dict) -> str:
    """Dispatch an infra tool call by name. Returns a JSON string result."""

    if name == "host_terminal__run_command":
        return _run_terminal_command(
            command=args.get("command", ""),
            cwd=args.get("cwd"),
            timeout=args.get("timeout", 60),
        )
    elif name == "host_ui_controller__get_ui_tree":
        return _get_ui_tree(
            window_title=args.get("window_title"),
            max_depth=args.get("max_depth", 50),
        )
    elif name == "host_ui_controller__type_into_element":
        return _type_into_element(
            text=args.get("text", ""),
            element_name=args.get("element_name", ""),
            window_title=args.get("window_title"),
            press_enter=args.get("press_enter", True),
            clear_first=args.get("clear_first", True),
        )
    elif name == "host_ui_controller__click_element":
        return _click_element(
            element_name=args.get("element_name", ""),
            window_title=args.get("window_title"),
        )
    else:
        return json.dumps({"error": f"Unknown infra tool: {name}"})


# ── Private Helpers ──────────────────────────────────────────────────────────

_DEFAULT_CWD = os.path.join(os.path.expanduser("~"), "Desktop")

# Interactive control types we care about when building the UI tree
_INTERACTIVE_TYPES = {
    "Button", "Edit", "ComboBox", "CheckBox", "RadioButton",
    "Hyperlink", "MenuItem", "TabItem", "ListItem", "TreeItem",
    "Slider", "Spinner", "Document", "Text", "Pane", "Group",
    "DataItem", "Custom", "List",
}


def _run_terminal_command(command: str, cwd: str = None, timeout: int = 60) -> str:
    """Execute a shell command and return its output."""
    work_dir = cwd or _DEFAULT_CWD
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return json.dumps({
            "status": "success",
            "exit_code": result.returncode,
            "stdout": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
            "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
            "cwd": work_dir,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "error": f"Command timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _walk_tree(element, elements: list, depth: int, max_depth: int, max_elements: int):
    """Recursively walk the UIA tree, collecting interactive elements."""
    if depth > max_depth or len(elements) >= max_elements:
        return

    try:
        ctrl_type = element.element_info.control_type or ""
    except Exception:
        ctrl_type = ""

    try:
        name = element.element_info.name or ""
    except Exception:
        name = ""

    try:
        auto_id = element.element_info.automation_id or ""
    except Exception:
        auto_id = ""

    if ctrl_type in _INTERACTIVE_TYPES and (name.strip() or auto_id.strip()):
        elements.append({
            "type": ctrl_type,
            "name": name,
            "automation_id": auto_id,
            "depth": depth,
        })

    try:
        children = element.children()
    except Exception:
        children = []

    for child in children:
        if len(elements) >= max_elements:
            break
        _walk_tree(child, elements, depth + 1, max_depth, max_elements)


def _get_ui_tree(window_title: str = None, max_depth: int = 50) -> str:
    """Walk the UIA element tree and return a compact JSON of interactive elements."""
    if sys.platform != "win32":
        return json.dumps({"error": "UI tree inspection is only supported on Windows."})

    try:
        from pywinauto import Desktop as Dsk

        if window_title:
            try:
                root = Dsk(backend="uia").window(title_re=f".*{window_title}.*")
                root.wait("exists", timeout=3)
            except Exception:
                root = Dsk(backend="uia").window(active_only=True, found_index=0)
        else:
            root = Dsk(backend="uia").window(active_only=True, found_index=0)

        elements = []
        _walk_tree(root, elements, depth=0, max_depth=max_depth, max_elements=1000)

        return json.dumps({
            "window": root.window_text(),
            "element_count": len(elements),
            "elements": elements,
        })
    except Exception as e:
        return json.dumps({"error": f"UI tree error: {e}"})


def _find_element(element_name: str, window_title: str = None):
    """Find a UI element by name substring. Returns the pywinauto wrapper or None."""
    from pywinauto import Desktop as Dsk

    if window_title:
        try:
            root = Dsk(backend="uia").window(title_re=f".*{window_title}.*")
            root.wait("exists", timeout=3)
        except Exception:
            root = Dsk(backend="uia").window(active_only=True, found_index=0)
    else:
        root = Dsk(backend="uia").window(active_only=True, found_index=0)

    search_name = element_name.lower()

    def _search(el, depth=0, max_depth=50):
        if depth > max_depth:
            return None
        try:
            name = (el.element_info.name or "").lower()
            if search_name in name:
                return el
        except Exception:
            pass
        try:
            for child in el.children():
                found = _search(child, depth + 1, max_depth)
                if found:
                    return found
        except Exception:
            pass
        return None

    return _search(root)


def _type_into_element(
    text: str,
    element_name: str,
    window_title: str = None,
    press_enter: bool = True,
    clear_first: bool = True,
) -> str:
    """Type text into a named UI element."""
    if sys.platform != "win32":
        return json.dumps({"error": "Only supported on Windows."})

    try:
        el = _find_element(element_name, window_title)
        if el is None:
            return json.dumps({
                "status": "error",
                "error": f"Element '{element_name}' not found.",
            })

        try:
            el.set_focus()
            time.sleep(0.3)
        except Exception:
            pass

        if clear_first:
            try:
                from pywinauto import keyboard
                keyboard.send_keys("^a")
                time.sleep(0.1)
            except Exception:
                pass

        try:
            el.type_keys(text, with_spaces=True, with_newlines=True)
        except Exception:
            from pywinauto import keyboard
            keyboard.send_keys(text, with_spaces=True, with_newlines=True)

        if press_enter:
            time.sleep(0.2)
            from pywinauto import keyboard
            keyboard.send_keys("{ENTER}")

        return json.dumps({
            "status": "success",
            "typed": text[:100],
            "enter_pressed": press_enter,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _click_element(element_name: str, window_title: str = None) -> str:
    """Click a named UI element."""
    if sys.platform != "win32":
        return json.dumps({"error": "Only supported on Windows."})

    try:
        el = _find_element(element_name, window_title)
        if el is None:
            return json.dumps({
                "status": "error",
                "error": f"Element '{element_name}' not found.",
            })

        try:
            el.click_input()
        except Exception:
            try:
                el.invoke()
            except Exception:
                el.set_focus()

        return json.dumps({
            "status": "success",
            "clicked": element_name,
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
