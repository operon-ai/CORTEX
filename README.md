
### Installation

This project uses [**uv**](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

#### Install uv (one-time setup)
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Clone and install for development
```bash
git clone https://github.com/your-org/cortex.git
cd CORTEX
uv sync          # creates .venv and installs all core dependencies
```


Don't forget to also `brew install tesseract` (macOS/Linux) or install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)! Pytesseract requires this extra installation to work.

### API Configuration

#### Option 1: Environment Variables
Add to your `.bashrc` (Linux) or `.zshrc` (MacOS):
```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

#### Option 2: Python Script
```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
```

### CLI

Note, this is running Cortex, our improved agent, without bBoN. 

Run Cortex with the required parameters:

```bash
uv run cortex \
    --provider openai \
    --model gpt-5-2025-08-07 \
    --ground_provider huggingface \
    --ground_url http://localhost:8080 \
    --ground_model meituan/EvoCUA-8B-20260105 \
    --grounding_width 1920 \
    --grounding_height 1080
```

#### Local Coding Environment (Optional)
For tasks that require code execution (e.g., data processing, file manipulation, system automation), you can enable the local coding environment:

```bash
uv run cortex \
    --provider openai \
    --model gpt-5-2025-08-07 \
    --ground_provider huggingface \
    --ground_url http://localhost:8080 \
    --ground_model meituan/EvoCUA-8B-20260105 \
    --grounding_width 1920 \
    --grounding_height 1080 \
    --enable_local_env
```

⚠️ **WARNING**: The local coding environment executes arbitrary Python and Bash code locally on your machine. Only use this feature in trusted environments and with trusted inputs.

#### Required Parameters
- **`--provider`**: Main generation model provider (e.g., openai, anthropic, etc.) - Default: "openai"
- **`--model`**: Main generation model name (e.g., gpt-5-2025-08-07) - Default: "gpt-5-2025-08-07"
- **`--ground_provider`**: The provider for the grounding model - **Required**
- **`--ground_url`**: The URL of the grounding model - **Required**
- **`--ground_model`**: The model name for the grounding model - **Required**
- **`--grounding_width`**: Width of the output coordinate resolution from the grounding model - **Required**
- **`--grounding_height`**: Height of the output coordinate resolution from the grounding model - **Required**

#### Optional Parameters
- **`--model_temperature`**: The temperature to fix all model calls to (necessary to set to 1.0 for models like o3 but can be left blank for other models)

#### Grounding Model Dimensions
The grounding width and height should match the output coordinate resolution of your grounding model:
- **meituan/EvoCUA-8B-20260105**: Use `--grounding_width 1920 --grounding_height 1080`

#### Optional Parameters
- **`--model_url`**: Custom API URL for main generation model - Default: ""
- **`--model_api_key`**: API key for main generation model - Default: ""
- **`--ground_api_key`**: API key for grounding model endpoint - Default: ""
- **`--max_trajectory_length`**: Maximum number of image turns to keep in trajectory - Default: 8
- **`--enable_reflection`**: Enable reflection agent to assist the worker agent - Default: True
- **`--enable_local_env`**: Enable local coding environment for code execution (WARNING: Executes arbitrary code locally) - Default: False

#### Local Coding Environment Details
The local coding environment enables Cortex to execute Python and Bash code directly on your machine. This is particularly useful for:

- **Data Processing**: Manipulating spreadsheets, CSV files, or databases
- **File Operations**: Bulk file processing, content extraction, or file organization
- **System Automation**: Configuration changes, system setup, or automation scripts
- **Code Development**: Writing, editing, or executing code files
- **Text Processing**: Document manipulation, content editing, or formatting

When enabled, the agent can use the `call_code_agent` action to execute code blocks for tasks that can be completed through programming rather than GUI interaction.

**Requirements:**
- **Python**: The same Python interpreter used to run Cortex (automatically detected)
- **Bash**: Available at `/bin/bash` (standard on macOS and Linux)
- **System Permissions**: The agent runs with the same permissions as the user executing it

**Security Considerations:**
- The local environment executes arbitrary code with the same permissions as the user running the agent
- Only enable this feature in trusted environments
- Be cautious when the agent generates code for system-level operations
- Consider running in a sandboxed environment for untrusted tasks
- Bash scripts are executed with a 30-second timeout to prevent hanging processes

### `cua_agents` SDK

First, we import the necessary modules. `Cortex` is the main agent class for Cortex. `OSWorldACI` is our grounding agent that translates agent actions into executable python code.
```python
import pyautogui
import io
from cua_agents.v1.agents.cortex import Cortex
from cua_agents.v1.agents.grounding import OSWorldACI
from cua_agents.v1.utils.local_env import LocalEnv  # Optional: for local coding environment

# Load in your API keys.
from dotenv import load_dotenv
load_dotenv()

current_platform = "linux"  # "darwin", "windows"
```

Next, we define our engine parameters. `engine_params` is used for the main agent, and `engine_params_for_grounding` is for grounding. For `engine_params_for_grounding`, we support custom endpoints like HuggingFace TGI, vLLM, and Open Router.

```python
engine_params = {
  "engine_type": provider,
  "model": model,
  "base_url": model_url,           # Optional
  "api_key": model_api_key,        # Optional
  "temperature": model_temperature # Optional
}

# Load the grounding engine from a custom endpoint
ground_provider = "<your_ground_provider>"
ground_url = "<your_ground_url>"
ground_model = "<your_ground_model>"
ground_api_key = "<your_ground_api_key>"

grounding_width = 1920  # Width of output coordinate resolution
grounding_height = 1080  # Height of output coordinate resolution

engine_params_for_grounding = {
  "engine_type": ground_provider,
  "model": ground_model,
  "base_url": ground_url,
  "api_key": ground_api_key,  # Optional
  "grounding_width": grounding_width,
  "grounding_height": grounding_height,
}
```

Then, we define our grounding agent and Cortex.

```python
# Optional: Enable local coding environment
enable_local_env = False  # Set to True to enable local code execution
local_env = LocalEnv() if enable_local_env else None

grounding_agent = OSWorldACI(
    env=local_env,  # Pass local_env for code execution capability
    platform=current_platform,
    engine_params_for_generation=engine_params,
    engine_params_for_grounding=engine_params_for_grounding,
    width=1920,  # Optional: screen width
    height=1080  # Optional: screen height
)

agent = Cortex(
    engine_params,
    grounding_agent,
    platform=current_platform,
    max_trajectory_length=8,  # Optional: maximum image turns to keep
    enable_reflection=True     # Optional: enable reflection agent
)
```

Finally, let's query the agent!

```python
# Get screenshot.
screenshot = pyautogui.screenshot()
buffered = io.BytesIO() 
screenshot.save(buffered, format="PNG")
screenshot_bytes = buffered.getvalue()

obs = {
  "screenshot": screenshot_bytes,
}

instruction = "Close VS Code"
info, action = agent.predict(instruction=instruction, observation=obs)

exec(action[0])
```

Refer to `cua_agents/v1/cli_app.py` for more details on how the inference loop works.
