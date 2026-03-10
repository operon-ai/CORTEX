import argparse
import logging
import os
import signal
import sys
import threading
from dotenv import load_dotenv

# Path hack to ensure we can import from parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cua_agents.v1.agents.cortex import Cortex

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli_app")

# Global stop event for signal handling
stop_event = threading.Event()

def signal_handler(signum, frame):
    """Handle Ctrl+C to stop the orchestrator gracefully."""
    if not stop_event.is_set():
        print("\n\n🛑 STOP REQUESTED (Ctrl+C). Finishing current step then exiting...")
        stop_event.set()
    else:
        print("\n\n🛑 EMERGENCY EXIT.")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def console_log(msg: str, level: str = "info", icon: str = ""):
    """Callback for Cortex to log to the console."""
    icon_prefix = f"{icon} " if icon else ""
    print(f"[{level.upper()}] {icon_prefix}{msg}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Cortex CLI Orchestrator")
    parser.add_argument("task", type=str, nargs="?", help="The task for the agent to perform")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps to allow")
    
    args = parser.parse_args()

    # Initialize Cortex
    # Note: CLI doesn't need hide/show UI logic for itself usually, but we could add it if desired.
    orchestrator = Cortex(
        max_steps=args.max_steps,
        stop_flag=stop_event,
        log_fn=console_log
    )

    task = args.task
    if not task:
        task = input("\n📝 Enter task description: ").strip()
    
    if not task:
        print("Empty task. Exiting.")
        return

    print("\n" + "="*80)
    print(f"🚀 STARTING CORTEX: {task}")
    print("="*80 + "\n")

    try:
        final_state = orchestrator.run(task)
        print("\n" + "="*80)
        print("✅ WORKFLOW COMPLETE")
        print(f"Final Outcome: {final_state.get('last_worker_result', 'No result.')}")
        print("="*80)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        logger.exception("Cortex crashed")

if __name__ == "__main__":
    main()
