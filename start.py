"""
start.py — Launch both CORTEX processes with a single command.

Usage:
    python start.py

Starts:
  1. Python WebSocket server  (port 7577)
  2. Electron GUI application  (connects to ws://localhost:7577/ws)

Press Ctrl+C to stop both.
"""

import os
import signal
import subprocess
import sys
import time

ROOT    = os.path.dirname(os.path.abspath(__file__))
SERVER  = os.path.join(ROOT, "server", "ws_server.py")
GUI_DIR = os.path.join(ROOT, "gui")

procs: list[subprocess.Popen] = []


def cleanup(*_args):
    for p in procs:
        try:
            p.terminate()
        except OSError:
            pass
    for p in procs:
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def main():
    print("\n  ⚡ CORTEX Launcher")
    print("  ──────────────────\n")

    # 1. Start the Python server
    print("  [1/2] Starting WebSocket server on port 7577 …")
    server_proc = subprocess.Popen(
        [sys.executable, SERVER],
        cwd=ROOT,
    )
    procs.append(server_proc)

    time.sleep(1.5)
    if server_proc.poll() is not None:
        print("  ❌  Server failed to start. Check server/ws_server.py.")
        sys.exit(1)
    print("  ✅  Server running (PID {})".format(server_proc.pid))

    # 2. Start the Electron GUI
    print("  [2/2] Starting Electron GUI …")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    gui_proc = subprocess.Popen(
        [npm, "start"],
        cwd=GUI_DIR,
        shell=False,
    )
    procs.append(gui_proc)
    print("  ✅  GUI starting (PID {})".format(gui_proc.pid))

    print("\n  Press Ctrl+C to stop both processes.\n")

    while True:
        time.sleep(1)
        if server_proc.poll() is not None:
            print("  ⚠️  Server process exited unexpectedly.")
            cleanup()
        if gui_proc.poll() is not None:
            print("  ⚠️  GUI process exited.")
            cleanup()


if __name__ == "__main__":
    main()
