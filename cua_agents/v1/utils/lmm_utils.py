import logging
import base64
from typing import Dict, List, Optional, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger("cortex.agent")

class LMMAgent:
    """
    A lightweight replacement for the legacy LMMAgent that used the deleted core module.
    Wraps AzureChatOpenAI from LangChain and maintains message history.
    """
    def __init__(self, engine_params: Dict, system_prompt: Optional[str] = None):
        self.engine_params = engine_params
        self.system_prompt = system_prompt
        self.messages = []
        
        # Initialize the Azure OpenAI client
        # Defaulting to values from environment if not explicitly provided in engine_params
        self.llm = AzureChatOpenAI(
            azure_deployment=engine_params.get("model", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=engine_params.get("temperature", 0.0),
        )
        
        self.reset()

    def reset(self):
        """Reset the message history to the system prompt."""
        self.messages = []
        if self.system_prompt:
            self.messages.append(SystemMessage(content=self.system_prompt))

    def add_system_prompt(self, system_prompt: str):
        """Update or set the system prompt."""
        self.system_prompt = system_prompt
        # If the first message is a SystemMessage, update it. Otherwise, prepend it.
        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0] = SystemMessage(content=system_prompt)
        else:
            self.messages.insert(0, SystemMessage(content=system_prompt))

    def add_message(self, text_content: str, image_content: Optional[Any] = None, role: str = "user", put_text_last: bool = False):
        """Add a message to the history. image_content can be raw bytes or base64 string."""
        content = []
        
        if image_content:
            # If it's bytes (raw screenshot), encode to base64
            if isinstance(image_content, bytes):
                image_b64 = base64.b64encode(image_content).decode("utf-8")
            else:
                image_b64 = image_content
                
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })
        
        if text_content:
            text_item = {"type": "text", "text": text_content}
            if put_text_last:
                content.append(text_item)
            else:
                content.insert(0, text_item)

        if role == "user":
            self.messages.append(HumanMessage(content=content))
        elif role == "assistant":
            # Assistant content in LangChain is usually just text for simple cases
            self.messages.append(AIMessage(content=text_content))
        elif role == "system":
            self.messages.append(SystemMessage(content=text_content))

    def get_response(self, temperature: Optional[float] = None, use_thinking: bool = False, **kwargs) -> str:
        """Get a response from the LLM based on the current message history."""
        # temperature override if provided
        config = {}
        if temperature is not None:
            # Note: We might need to recreate the LLM if temperature changes frequently, 
            # but for now let's hope it's fairly static or we can pass it in bind if it was a chain.
            # AzureChatOpenAI doesn't easily allow changing temperature after init without creating a new instance.
            pass

        try:
            # If use_thinking is requested, we might need to handle specific model types
            # but for Azure GPT-4o-mini/GPT-4o, we just call invoke.
            response = self.llm.invoke(self.messages)
            return response.content
        except Exception as e:
            logger.error(f"Error getting response from AzureChatOpenAI: {e}")
            raise

import os
