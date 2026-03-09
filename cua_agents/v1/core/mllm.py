import base64

import numpy as np

from cua_agents.v1.core.engine import (
    LMMEngineAzureOpenAI,
    LMMEngineOpenAI,
    LMMEnginevLLM,
)

# All supported engine types — used for isinstance checks
_OPENAI_COMPAT_ENGINES = (LMMEngineOpenAI, LMMEngineAzureOpenAI, LMMEnginevLLM)


class LMMAgent:
    """Multimodal LLM agent that wraps a provider engine with message-history management."""

    def __init__(self, engine_params=None, system_prompt=None, engine=None):
        if engine is None:
            if engine_params is not None:
                engine_type = engine_params.get("engine_type")
                if engine_type == "openai":
                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "azure":
                    self.engine = LMMEngineAzureOpenAI(**engine_params)
                elif engine_type == "vllm":
                    self.engine = LMMEnginevLLM(**engine_params)
                else:
                    raise ValueError(
                        f"engine_type '{engine_type}' is not supported. "
                        f"Supported: 'openai', 'azure', 'vllm'"
                    )
            else:
                raise ValueError("engine_params must be provided")
        else:
            self.engine = engine

        self.messages = []

        if system_prompt:
            self.add_system_prompt(system_prompt)
        else:
            self.add_system_prompt("You are a helpful assistant.")

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode_image(self, image_content):
        """Base64-encode an image from a file path or raw bytes."""
        if isinstance(image_content, str):
            with open(image_content, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return base64.b64encode(image_content).decode("utf-8")

    # ── Message management ───────────────────────────────────────────────────

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        if self.messages:
            self.messages[0] = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        else:
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

    def remove_message_at(self, index):
        if index < len(self.messages):
            self.messages.pop(index)

    def replace_message_at(self, index, text_content, image_content=None, image_detail="high"):
        if index < len(self.messages):
            self.messages[index] = {
                "role": self.messages[index]["role"],
                "content": [{"type": "text", "text": text_content}],
            }
            if image_content:
                b64 = self.encode_image(image_content)
                self.messages[index]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": image_detail,
                        },
                    }
                )

    def add_message(
        self,
        text_content,
        image_content=None,
        role=None,
        image_detail="high",
        put_text_last=False,
    ):
        """Add a message to the conversation history.

        All three supported engines (OpenAI, Azure, vLLM) use the same
        OpenAI-compatible message format, so there is one unified path.
        """
        if not isinstance(self.engine, _OPENAI_COMPAT_ENGINES):
            raise ValueError(f"Unsupported engine type: {type(self.engine)}")

        # Infer role from previous message if not explicitly set
        if role != "user":
            last_role = self.messages[-1]["role"]
            if last_role in ("system", "assistant"):
                role = "user"
            else:
                role = "assistant"

        message = {
            "role": role,
            "content": [{"type": "text", "text": text_content}],
        }

        if isinstance(image_content, np.ndarray) or image_content:
            images = image_content if isinstance(image_content, list) else [image_content]
            # vLLM uses bare data URI; OpenAI/Azure use image_url with detail
            use_detail = not isinstance(self.engine, LMMEnginevLLM)
            # vLLM models typically support only 1 image per prompt — keep only the latest
            if not use_detail and len(images) > 1:
                images = images[-1:]
            for img in images:
                b64 = self.encode_image(img)
                if use_detail:
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": image_detail,
                            },
                        }
                    )
                else:
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image;base64,{b64}"},
                        }
                    )

        # Rotate text to be last if requested (some grounding models prefer this)
        if put_text_last:
            text_block = message["content"].pop(0)
            message["content"].append(text_block)

        self.messages.append(message)

    # ── Generation ───────────────────────────────────────────────────────────

    def get_response(
        self,
        user_message=None,
        messages=None,
        temperature=0.0,
        max_new_tokens=None,
        use_thinking=False,
        **kwargs,
    ):
        """Generate the next response based on conversation history."""
        if messages is None:
            messages = self.messages
        if user_message:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_message}]}
            )
        return self.engine.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
