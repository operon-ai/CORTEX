# CORTEX: Multimodal Desktop Automation Agent

CORTEX is a high-performance, multimodal desktop automation agent designed to orchestrate complex tasks across various applications using a combination of a large-scale reasoning brain and specialized visual grounding models.

## Watch CORTEX in Action

[![CORTEX Demo](https://img.youtube.com/vi/yyZtm4mEG04/0.jpg)](https://youtu.be/yyZtm4mEG04?si=Nb_8RFAJDVjuDNvb)

## Architecture: "Brain & Hands"

CORTEX operates on a modular architecture that separates reasoning from execution:

- **The Brain (Orchestrator)**: Powered by **Azure OpenAI (gpt-5-mini)**. It manages high-level reasoning, task decomposition into TODOs, and routing between specialized worker nodes.
- **The Hands (GUI Grounding)**: Powered by **EvoCUA-8B**. It specializes in visual grounding, translating natural language instructions into precise screen coordinates and UI interactions (click, type, scroll, etc.).
- **Worker Nodes**:
    - **GUI Worker**: Screen-based interaction via EvoCUA.
    - **MCP Worker**: External service integration (Slack, Notion, etc.) via Model Context Protocol.
    - **Code Worker**: Local Python and Bash execution for complex file processing.
    - **Infra Worker**: Direct Windows UI Automation (UIA) and terminal commands.

## Codebase Structure

### Backend (Python / FastAPI)
- `server/ws_server.py`: The central hub that manages WebSocket connections, initializes the Cortex brain, and handles audio (STT/TTS).
- `cua_agents/v1/agents/cortex.py`: The LangGraph-powered orchestrator that defines the "state loop" of the agent.
- `cua_agents/v1/agents/evocua_agent.py`: Specialized agent for the EvoCUA visual model.
- `cua_agents/v1/utils/azure_audio.py`: Azure Cognitive Services integration for high-quality speech-to-text and text-to-speech.

### Frontend (Electron / Vite)
- `gui/main.js`: Electron main process. Manages the transparent, glassmorphic "island" window and IPC communications.
- `gui/renderer/app.js`: Vue-like reactive logic (vanilla JS) that handles the WebSocket protocol, task input, feed rendering, and real-time TODO checklist updates.
- `gui/renderer/style.css`: Modern, premium design system with dark mode, animations, and responsive layouts.

## WebSocket Architecture

The frontend and backend communicate via a real-time WebSocket protocol (port 7577):

1. **Task Flow**: User enters a task → `start_task` (WS) → Cortex Brain starts → Status/Logs/TODOs streamed back → `done` (WS).
2. **Audio Flow**: User speaks → Audio captured in browser → `stt_audio` (WS) → Azure Transcription → Transcribed text sent back to input field.
3. **Feedback Loop**: Every action taken by a worker node is reported back to the Orchestrator and the UI simultaneously for full transparency.

## Installation & Setup

### Prerequisites
- **Python 3.10+** (Recommend using a virtual environment).
- **Node.js 18+** (For the Electron GUI).
- **Tesseract OCR**: Required for text-based UI inspection. [Download for Windows](https://github.com/UB-Mannheim/tesseract/wiki).

### 1. Backend Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .
pip install azure-cognitiveservices-speech langchain-openai httpx
```

### 2. Frontend Setup
```powershell
cd gui
npm install
```

### 3. Configuration
1. Copy `.env.example` to `.env`.
2. Provide your **Azure OpenAI** credentials (Endpoint, API Key, Deployment Names).
3. Provide your **Azure Speech** credentials (Key and Region).

## Running CORTEX

CORTEX provides a unified launcher to start both the server and the GUI in one command:

```powershell
python start.py
```

- Follow the real-time logs in the Electron island.
- Use the Microphone icon for voice-driven tasks.
- Monitor the TODO checklist to track progress on long-running tasks.

---
© 2026 Operon AI. All rights reserved.
