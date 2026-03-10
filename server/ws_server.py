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
from langfuse import get_client, observe
import uvicorn

# Move imports to top for better visibility and earlier error detection
from cua_agents.v1.agents.cortex import Cortex
from cua_agents.v1.utils.azure_audio import AzureAudio


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
_cortex = None
_audio = None


@app.on_event("startup")
async def _on_startup() -> None:
    global _event_loop, _cortex, _audio
    _event_loop = asyncio.get_running_loop()

    # Pre-initialize the Cortex orchestrator early
    print(f"[{time.strftime('%H:%M:%S')}] [INIT] Pre-initializing Cortex orchestrator and agents...", flush=True)
    
    _cortex = Cortex(
        max_steps=30,
        hide_ui=_hide_ui,
        show_ui=_show_ui,
        stop_flag=_stop_flag,
        log_fn=log_to_ui,
        todo_fn=push_todo,
    )
    
    print(f"[{time.strftime('%H:%M:%S')}] [INIT] Initializing Azure Audio...", flush=True)
    _audio = AzureAudio()
    
    # Perform warm-up (connect MCP, start agents)
    await _cortex.warm_up()
    print(f"[{time.strftime('%H:%M:%S')}] [INIT] Startup initialization complete.", flush=True)


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
    
    # Optional: Speak the log message if it's an important update
    if level in ["success", "error"] and _audio:
        _audio.speak(message)


def set_status(status: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [STATUS] → {status}", flush=True)
    _push({"type": "status", "status": status})


def push_todo(items: list) -> None:
    """Push an updated todo list to the UI."""
    _push({
        "type": "todo_update",
        "items": items
    })


def _hide_ui() -> None:
    """Tell the Electron window to go transparent so it doesn't appear in screenshots."""
    _push({"type": "hide"})
    time.sleep(0.1)  # opacity change is instant, small buffer for IPC


def _show_ui() -> None:
    """Restore the Electron window."""
    _push({"type": "show"})


# ── Agent runner (background thread) ─────────────────────────────────────────

@observe(name="cortex_run")
def _run_agent(task: str) -> None:
    global _agent_running
    try:
        # Set the Langfuse trace name
        langfuse = get_client()
        langfuse.update_current_trace(name=task)

        log_to_ui(f"Task: {task}", "info")
        set_status("running")

        try:
            # Run the orchestrator graph using the pre-initialized global instance
            if _cortex is None:
                log_to_ui("Error: Cortex not initialized.", "error")
                return

            log_to_ui("Starting orchestrator loop…", "success")
            final_state = _cortex.run(task)

            # Clean up
            set_status("done")

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[{time.strftime('%H:%M:%S')}] [CRITICAL] {exc}\n{tb}", flush=True)
            log_to_ui(f"Error: {exc}", "error")
            set_status("error")

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] {exc}\n{tb}", flush=True)
        log_to_ui(f"Error: {exc}", "error")
        set_status("error")
    finally:
        try:
            get_client().flush()
        except Exception:
            pass
        print(f"[{time.strftime('%H:%M:%S')}] Agent thread finished.", flush=True)
        _agent_running = False
        _stop_flag.clear()


# ── HTTP + WebSocket endpoints ────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "agent_running": _agent_running})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _agent_running, _audio
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
                        "time": time.strftime("%H:%M:%S"),
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
                
            elif msg["type"] == "stt_audio":
                # Handle audio transcription via Azure OpenAI
                audio_data = msg.get("data") # base64 (matches client send)
                if _audio is None:
                    log_to_ui("Azure Audio is NOT initialized yet. Check server logs.", "error")
                elif not audio_data:
                    log_to_ui("Audio data missing from WebSocket message.", "warning")
                else:
                    log_to_ui("Transcribing audio...", "info")
                    import base64
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                        tmp.write(base64.b64decode(audio_data))
                        tmp_path = tmp.name
                    
                    try:
                        text = await _audio.transcribe(tmp_path)
                        await ws.send_text(json.dumps({
                            "type": "transcript",
                            "text": text
                        }))
                        log_to_ui(f"Heard: {text}", "info")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    except WebSocketDisconnect:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7577, log_level="info")
