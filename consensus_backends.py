"""consensus_backends.py

Modular backend definitions for use with consensus_agent.py.

Each backend implements the LLMBackend interface, providing
an async `complete(prompt: str) -> str` method for uniform multi-model inference.
Production implementations should subclass LLMBackend and override
the `complete` method with real model or API calls.

Included mock classes: LlamaCppBackend, OllamaBackend, OpenAIBackend, AnthropicBackend.
"""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
import aiohttp
import os

class LLMBackend(ABC):
    """
    Abstract base class for all large language model backends used
    in the consensus engine.

    Subclasses must implement the async 'complete' method.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """
        Issue a prompt to this backend and return a response string.

        Args:
            prompt: The input prompt to provide to the model.

        Returns:
            str: The backend/model's raw text response.
        """
        pass

class LlamaCppBackend(LLMBackend):
    """
    Backend for local Llama.cpp-based models.

    For production, integrate llama-cpp-python or subprocess to the local Llama model server.
    """
    def __init__(self, name: str, path: Path):
        super().__init__(name)
        self.path = path

    async def complete(self, prompt: str) -> str:
        # Mock response; replace with llama-cpp model invocation
        await asyncio.sleep(0.05)
        return f"(Mock LlamaCpp: {self.name}, model file: {self.path}) => {prompt}"

class OllamaBackend(LLMBackend):
    """
    Backend for models served by Ollama.

    Args:
        name: Internal identifier.
        model: Ollama model name/tag.

    For real use, connects to the Ollama HTTP API (default: http://localhost:11434).
    Falls back to a mock response if the API is not reachable.
    """
    def __init__(self, name: str, model: str):
        super().__init__(name)
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

    async def complete(self, prompt: str) -> str:
        """Send the prompt to local Ollama API or fallback to mock if it fails."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False  # get entire output at once
                }
                async with session.post(self.api_url, json=payload, timeout=60) as resp:
                    if resp.status != 200:
                        msg = await resp.text()
                        return f"[Ollama API error {resp.status}]: {msg}"
                    data = await resp.json()
                    # Ollama returns {"response": "..."} or list of stream chunks if stream True
                    return data.get("response", "").strip() or "[Ollama API empty response]"
        except Exception as e:
            # If running in offline/test mode, fallback to mock
            return f"(Mock Ollama: {self.name}, model: {self.model}) [API err: {str(e)}] => {prompt}"

class OpenAIBackend(LLMBackend):
    """
    Backend for OpenAI-compatible APIs (e.g., OpenAI, Azure, Meta).

    Args:
        name: Internal identifier.
        model: Model ID.
        endpoint: URL for the API endpoint.
        api_key: User's API key.
    """
    def __init__(self, name: str, model: str, endpoint: str, api_key: str):
        super().__init__(name)
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key

    async def complete(self, prompt: str) -> str:
        # Mock response; replace with actual OpenAI API request
        await asyncio.sleep(0.05)
        return f"(Mock OpenAI: {self.name}, model: {self.model}, endpoint: {self.endpoint}) => {prompt}"

class AnthropicBackend(LLMBackend):
    """
    Backend for Anthropic API (Claude and compatible).

    Args:
        name: Internal identifier.
        model: Model ID.
        api_key: User's Anthropic API Key.
    """
    def __init__(self, name: str, model: str, api_key: str):
        super().__init__(name)
        self.model = model
        self.api_key = api_key

    async def complete(self, prompt: str) -> str:
        # Mock response; replace with actual Anthropic API call
        await asyncio.sleep(0.05)
        return f"(Mock Anthropic: {self.name}, model: {self.model}) => {prompt}"

# -------------------------------------------------------------------------
class LlamaAPIBackend(LLMBackend):
    """
    Backend for the Llama API (e.g., llama-api.com or similar).

    Args:
        name: Internal identifier.
        model: Model name/id (e.g. "Llama-4-Maverick-17B-128E-Instruct-FP8").
        endpoint: API base URL, e.g. "https://api.llama.com/v1/chat/completions".
        api_key: API key (recommended to pass from environment: os.getenv('LLAMA_API_KEY')).

    This backend performs a real POST request to the Llama API endpoint, with
    Bearer token authentication and error handling. Falls back to a mock result on error.

    How to use with ConsensusEngine (in consensus_agent.py):

        LlamaAPIBackend(
            "llama-api",
            "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "https://api.llama.com/v1/chat/completions",
            os.getenv("LLAMA_API_KEY")
        )

    Replace endpoint/model/args as instructed by your Llama API provider.

    For streaming support, extend the implementation using 'stream': True and
    parse the response accordingly.
    """
    def __init__(self, name: str, model: str, endpoint: str, api_key: str):
        super().__init__(name)
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key

    async def complete(self, prompt: str) -> str:
        """
        Send a prompt to the Llama API for completion and return the model output.
        Falls back to a mock result if any error occurs.
        """
        if not self.api_key:
            return f"[LlamaAPIBackend error]: No API key provided for {self.endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, headers=headers, json=payload, timeout=60) as resp:
                    if resp.status != 200:
                        msg = await resp.text()
                        return f"[Llama API error {resp.status}]: {msg}"
                    data = await resp.json()
                    # Robust response parsing:
                    # Llama API final format: check "completion_message.content.text" first
                    if (
                        "completion_message" in data
                        and "content" in data["completion_message"]
                        and "text" in data["completion_message"]["content"]
                    ):
                        return data["completion_message"]["content"]["text"].strip()
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        # OpenAI Chat style: {"message": {"role": "assistant", "content": "text"}}
                        if "message" in choice and "content" in choice["message"]:
                            return choice["message"]["content"].strip()
                        # OpenAI (non-chat) style: {"text": "..."}
                        if "text" in choice:
                            return choice["text"].strip()
                    if "result" in data:
                        return data["result"]
                    # Fallback: return the full response JSON for visibility
                    import json
                    return f"[Llama API: {json.dumps(data)}]"
        except Exception as e:
            return f"(Mock LlamaAPIBackend: {self.name}, model: {self.model}) [API err: {str(e)}] => {prompt}"

if __name__ == "__main__":
    # Example test demonstrating construction and invocation of all backends
    async def main():
        print("Testing consensus_backends.py mock classes...\n")
        llama = LlamaCppBackend("llama-local", Path("/models/llama.bin"))
        ollama = OllamaBackend("gemma", "gemma:latest")
        openai = OpenAIBackend("gpt4o", "gpt-4o", "https://api.openai.com/v1", "sk-demo")
        anthropic = AnthropicBackend("claude3", "claude-3-opus-20240229", "test-key")
        llama_api_key = os.getenv("LLAMA_API_KEY", "sk-demo")
        llama_api = LlamaAPIBackend(
            "llama-api",
            "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "https://api.llama.com/v1/chat/completions",
            llama_api_key
        )

        for backend in [llama, ollama, openai, anthropic, llama_api]:
            response = await backend.complete("What is the capital of France?")
            print(response)

    asyncio.run(main())

