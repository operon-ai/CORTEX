# CORTEX: Multimodal Desktop Automation Agent

CORTEX is a high-performance, multimodal desktop automation agent designed to orchestrate complex tasks across various applications using a combination of a large-scale reasoning brain and specialized visual grounding models.

## 🚀 Architecture

CORTEX operates on a modular "Brain & Hands" architecture:

- **The Brain (Orchestrator)**: Powered by **Azure OpenAI (gpt-5-mini)**. It manages high-level reasoning, task decomposition, and tool orchestration.
- **The Hands (Grounding)**: Powered by **EvoCUA-8B**. It specializes in visual grounding, translating natural language instructions into precise screen coordinates and UI interactions.
- **The Interface**: A modern **Electron-based UI** that communicates with the CORTEX brain via a **WebSocket server** (port 7577).
- **The Infrastructure**: Supports **MCP (Model Context Protocol)** for external tool integration (Slack, Notion, etc.) and a **Local Coding Agent** for direct file and system manipulation.

## 🛠️ Installation

CORTEX uses [**uv**](https://github.com/astral-sh/uv) for fast and reproducible dependency management.

### 1. Install uv
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup
```bash
git clone https://github.com/operon-ai/CORTEX.git
cd CORTEX
uv sync
```

### 3. External Dependencies
- **Tesseract OCR**: Required for text grounding. [Download for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
- **Node.js**: Required for the Electron frontend.

## ⚙️ Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Fill in your **Azure OpenAI** credentials and **EvoCUA** endpoint details.

## 🖥️ Running CORTEX

### Starting the WebSocket Server
The backend server bridges the brain to the UI:
```bash
uv run python server/ws_server.py
```

### Starting the UI
(Assuming node is installed and you are in the frontend directory)
```bash
npm install
npm run dev
```

## 🔌 Core Components

### WebSocket Architecture
The Electron frontend sends user tasks to the Python server via WebSockets. The server then:
1. Captures system screenshots.
2. Forwards them to the Orchestrator.
3. Decides on the next action (GUI, MCP, or Code).
4. Streams logs and status updates back to the UI.

### Model Configuration
- **Orchestrator**: `gpt-5-mini` (Azure)
- **Grounding**: `meituan/EvoCUA-8B-20260105`

## ⚠️ Security
The **Local Coding Agent** executes arbitrary Python and Bash code. Use with caution in trusted environments.

---
© 2026 Operon AI. All rights reserved.
