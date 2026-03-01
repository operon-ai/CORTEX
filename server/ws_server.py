"""
CORTEX WebSocket Server
Bridges the Electron UI to the CORTEX agent brain via WebSocket on port 7577.
"""

import asyncio
import json
import os
import sys
import threading
import time

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# Load .env from the CORTEX root directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# Add root to path so we can import cua_agents
sys.path.insert(0, _ROOT)

app = FastAPI(title="CORTEX Server")

# ── State ─────────────────────────────────────────────────────────────────────

_event_loop: asyncio.AbstractEventLoop | None = None
_clients: list[WebSocket] = []
_agent_running = False
_stop_flag = threading.Event()


@app.on_event("startup")
async def _capture_loop() -> None:
    global _event_loop
    _event_loop = asyncio.get_running_loop()


# ── Broadcast helpers (thread-safe) ──────────────────────────────────────────

async def _broadcast(message: dict) -> None:
    data = json.dumps(message)
    gone: list[WebSocket] = []
    for ws in _clients:
        try:
            await ws.send_text(data)
        except Exception:
            gone.append(ws)
    for ws in gone:
        _clients.remove(ws)


def _push(message: dict) -> None:
    """Push a message from any thread into the asyncio event loop."""
    if _event_loop and _event_loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast(message), _event_loop)


def log_to_ui(message: str, level: str = "info", icon: str = "") -> None:
    ts = time.strftime("%H:%M:%S")
    prefix = {"info": "INFO", "step": "STEP", "success": " OK ", "warning": "WARN", "error": " ERR"}.get(level, level.upper())
    print(f"[{ts}] [{prefix}] {icon}  {message}", flush=True)
    _push({
        "type": "log",
        "level": level,
        "message": message,
        "time": ts,
        "icon": icon,
    })


def set_status(status: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [STATUS] → {status}", flush=True)
    _push({"type": "status", "status": status})


def _hide_ui() -> None:
    """Tell the Electron window to go transparent so it doesn't appear in screenshots."""
    _push({"type": "hide"})
    time.sleep(0.1)  # opacity change is instant, small buffer for IPC


def _show_ui() -> None:
    """Restore the Electron window."""
    _push({"type": "show"})


# ── Agent runner (background thread) ─────────────────────────────────────────

def _run_agent(task: str) -> None:
    global _agent_running
    try:
        log_to_ui(f"Task: {task}", "info", "📋")
        set_status("running")

        try:
            log_to_ui("Initializing agent…", "info", "⚙️")
            print(f"[{time.strftime('%H:%M:%S')}] Importing dependencies...", flush=True)
            import io
            import platform as plat

            import pyautogui
            from PIL import Image
            from cua_agents.v1.agents.cortex import Cortex
            from cua_agents.v1.agents.grounding import OSWorldACI
            print(f"[{time.strftime('%H:%M:%S')}] All imports OK.", flush=True)

            current_platform = plat.system().lower()  # 'windows', 'darwin', 'linux'
            print(f"[{time.strftime('%H:%M:%S')}] Platform: {current_platform}", flush=True)

            # Screen dimensions
            screen_width, screen_height = pyautogui.size()
            max_dim = 2400
            scale = min(max_dim / screen_width, max_dim / screen_height, 1)
            scaled_w = int(screen_width * scale)
            scaled_h = int(screen_height * scale)
            log_to_ui(f"Screen: {screen_width}x{screen_height} → {scaled_w}x{scaled_h}", "info", "🖥️")

            # ── Engine params ─────────────────────────────────────────────
            provider = os.environ["PROVIDER"]
            model = os.environ["MODEL"]
            engine_params: dict = {
                "engine_type": provider,
                "model": model,
                "base_url": os.environ.get("MODEL_URL", ""),
                "api_key": os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
                "temperature": None,  # use model default (avoids temperature=0.0 rejection)
            }
            # Azure-specific extras
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            if azure_endpoint:
                engine_params["azure_endpoint"] = azure_endpoint
            api_version = os.environ.get("OPENAI_API_VERSION", "")
            if api_version:
                engine_params["api_version"] = api_version

            print(f"[{time.strftime('%H:%M:%S')}] Engine: provider={provider}, model={model}", flush=True)

            engine_params_for_grounding = {
                "engine_type": os.environ["GROUND_PROVIDER"],
                "model": os.environ["GROUND_MODEL"],
                "base_url": os.environ["GROUND_URL"],
                "api_key": os.environ.get("GROUND_API_KEY", "any-value"),
                "grounding_width": int(os.environ.get("GROUNDING_WIDTH", "1920")),
                "grounding_height": int(os.environ.get("GROUNDING_HEIGHT", "1080")),
            }
            print(f"[{time.strftime('%H:%M:%S')}] Grounding: {engine_params_for_grounding['engine_type']} / {engine_params_for_grounding['model']}", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] Grounding URL: {engine_params_for_grounding['base_url']}", flush=True)

            # ── Initialize agents ─────────────────────────────────────────
            print(f"[{time.strftime('%H:%M:%S')}] Creating OSWorldACI...", flush=True)
            grounding_agent = OSWorldACI(
                env=None,
                platform=current_platform,
                engine_params_for_generation=engine_params,
                engine_params_for_grounding=engine_params_for_grounding,
                width=screen_width,
                height=screen_height,
            )
            print(f"[{time.strftime('%H:%M:%S')}] Grounding agent ready.", flush=True)

            print(f"[{time.strftime('%H:%M:%S')}] Creating Cortex agent...", flush=True)
            agent = Cortex(
                engine_params,
                grounding_agent,
                platform=current_platform,
            )
            agent.reset()
            print(f"[{time.strftime('%H:%M:%S')}] Cortex agent ready.", flush=True)

            log_to_ui("Agent initialized. Starting task.", "success", "✅")

            # ── Run loop ──────────────────────────────────────────────────
            for step in range(15):
                if _stop_flag.is_set():
                    log_to_ui("Stopped by user.", "warning", "⏹️")
                    break

                log_to_ui(f"Step {step + 1}/15: Capturing screen…", "step", "📷")
                _hide_ui()
                screenshot = pyautogui.screenshot()
                _show_ui()
                screenshot = screenshot.resize((scaled_w, scaled_h), Image.LANCZOS)
                print(f"[{time.strftime('%H:%M:%S')}] Screenshot captured & resized to {scaled_w}x{scaled_h}", flush=True)

                buffered = io.BytesIO()
                screenshot.save(buffered, format="PNG")
                img_size_kb = len(buffered.getvalue()) / 1024
                obs = {"screenshot": buffered.getvalue()}
                print(f"[{time.strftime('%H:%M:%S')}] Image encoded: {img_size_kb:.0f} KB", flush=True)

                log_to_ui("Agent is thinking…", "step", "🧠")
                print(f"[{time.strftime('%H:%M:%S')}] Calling agent.predict()...", flush=True)
                info, code = agent.predict(instruction=task, observation=obs)
                print(f"[{time.strftime('%H:%M:%S')}] agent.predict() returned. code={code}", flush=True)

                action = code[0]
                print(f"[{time.strftime('%H:%M:%S')}] Action: {action}", flush=True)

                if "done" in action.lower() or "fail" in action.lower():
                    log_to_ui(f"Agent finished: {action}", "success", "🎉")
                    break

                if "next" in action.lower():
                    log_to_ui("Skipping to next step.", "info", "⏭️")
                    continue

                if "wait" in action.lower():
                    log_to_ui("Agent waiting 5 s…", "info", "⏳")
                    time.sleep(5)
                    continue

                log_to_ui(f"Executing: {action}", "step", "🖱️")
                print(f"[{time.strftime('%H:%M:%S')}] exec() → {action}", flush=True)
                exec(action)
                print(f"[{time.strftime('%H:%M:%S')}] exec() done, sleeping 1s", flush=True)
                time.sleep(1)

            set_status("done")

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[{time.strftime('%H:%M:%S')}] [CRITICAL] {exc}\n{tb}", flush=True)
            log_to_ui(f"Error: {exc}", "error", "❌")
            set_status("error")

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] {exc}\n{tb}", flush=True)
        log_to_ui(f"Error: {exc}", "error", "❌")
        set_status("error")
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] Agent thread finished.", flush=True)
        _agent_running = False
        _stop_flag.clear()


# ── HTTP + WebSocket endpoints ────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "agent_running": _agent_running})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _agent_running
    await ws.accept()
    _clients.append(ws)

    await ws.send_text(json.dumps({
        "type": "status",
        "status": "running" if _agent_running else "idle",
    }))
    await ws.send_text(json.dumps({
        "type": "log", "level": "success",
        "message": "Connected to CORTEX server.",
        "time": time.strftime("%H:%M:%S"), "icon": "🔗",
    }))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg["type"] == "start_task":
                if _agent_running:
                    await ws.send_text(json.dumps({
                        "type": "log", "level": "warning",
                        "message": "Agent is already running.",
                        "time": time.strftime("%H:%M:%S"), "icon": "⚠️",
                    }))
                else:
                    _agent_running = True
                    _stop_flag.clear()
                    threading.Thread(
                        target=_run_agent, args=(msg["task"],), daemon=True,
                    ).start()

            elif msg["type"] == "stop_task":
                _stop_flag.set()

            elif msg["type"] == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7577, log_level="info")
