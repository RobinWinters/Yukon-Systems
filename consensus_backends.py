"""consensus_backends.py

Modular backend definitions for use with consensus_agent.py.

Each backend implements the LLMBackend interface, providing
an async `complete(prompt: str) -> dict` method for uniform multi-model inference.

Return value:
    Dict[str, Any] with required keys:
        "answer": str,
        "reasoning": str (may be empty for mock/data-free models),
        "confidence": float (0.0 for mocks/defaults)
For backward compatibility, if a string is returned, it will be wrapped as dict.
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
    async def complete(self, prompt: str) -> dict:
        """
        Issue a prompt to this backend and return a response dict for consensus flows.

        Args:
            prompt: The input prompt to provide to the model.

        Returns:
            dict: {
                "answer": str,        # The main response from the model
                "reasoning": str,     # Model's explanation, may be empty for mocks
                "confidence": float   # 0.0 by default; real models may estimate/probabilistically derive this
            }
        """
        pass

# If you add other backends here (such as your llama-api backend), make sure to use the same timing/wrap pattern as above!

class LlamaCppBackend(LLMBackend):
    """
    Backend for local Llama.cpp-based models.

    For production, integrate llama-cpp-python or subprocess to the local Llama model server.
    """
    def __init__(self, name: str, path: Path):
        super().__init__(name)
        self.path = path

    async def complete(self, prompt: str) -> dict:
        import time
        start = time.monotonic()
        # Mock response; replace with llama-cpp model invocation
        await asyncio.sleep(0.05)
        elapsed = time.monotonic() - start
        return {
            "answer": f"(Mock LlamaCpp: {self.name})",
            "reasoning": f"Given the prompt: {prompt}",
            "confidence": 0.0,
            "time": elapsed
        }
# End LlamaCppBackend.

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

    async def complete(self, prompt: str) -> dict:
        import time
        start = time.monotonic()
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
                        elapsed = time.monotonic() - start
                        return {
                            "answer": f"[Ollama API error {resp.status}]",
                            "reasoning": msg,
                            "confidence": 0.0,
                            "time": elapsed
                        }
                    data = await resp.json()
                    response = data.get("response", "").strip() or "[Ollama API empty response]"
                    elapsed = time.monotonic() - start
                    return {
                        "answer": response,
                        "reasoning": f"Ollama ({self.name}): returned response string",
                        "confidence": 0.0,
                        "time": elapsed
                    }
        except Exception as e:
            # If running in offline/test mode, fallback to mock
            elapsed = time.monotonic() - start
            return {
                "answer": f"(Mock Ollama: {self.name})",
                "reasoning": f"[API error: {str(e)}], prompt: {prompt}",
                "confidence": 0.0,
                "time": elapsed
            }

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

    async def complete(self, prompt: str) -> dict:
        # Mock response; replace with actual OpenAI API request
        await asyncio.sleep(0.05)
        return {
            "answer": f"(Mock OpenAI: {self.name})",
            "reasoning": f"Prompted with: {prompt}",
            "confidence": 0.0
        }

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

    async def complete(self, prompt: str) -> dict:
        # Mock response; replace with actual Anthropic API call
        await asyncio.sleep(0.05)
        return {
            "answer": f"(Mock Anthropic: {self.name})",
            "reasoning": f"Model '{self.model}' prompt: {prompt}",
            "confidence": 0.0
        }

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

    async def complete(self, prompt: str) -> dict:
        """
        Send a prompt to the Llama API for completion and return the model output as a dict.
        Falls back to a mock result if any error occurs.
        """
        import time
        start = time.monotonic()
        try:
            if not self.api_key:
                elapsed = time.monotonic() - start
                return {
                    "answer": f"[LlamaAPIBackend error]: No API key for {self.endpoint}",
                    "reasoning": "",
                    "confidence": 0.0,
                    "time": elapsed
                }
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
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, headers=headers, json=payload, timeout=60) as resp:
                    if resp.status != 200:
                        msg = await resp.text()
                        elapsed = time.monotonic() - start
                        return {
                            "answer": f"[Llama API error {resp.status}]",
                            "reasoning": msg,
                            "confidence": 0.0,
                            "time": elapsed
                        }
                    data = await resp.json()
                    elapsed = time.monotonic() - start
                    # Robust response parsing:
                    # Llama API final format: check "completion_message.content.text" first
                    if (
                        "completion_message" in data
                        and "content" in data["completion_message"]
                        and "text" in data["completion_message"]["content"]
                    ):
                        ans = data["completion_message"]["content"]["text"].strip()
                        return {"answer": ans, "reasoning": ans, "confidence": 0.0, "time": elapsed}
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        # OpenAI Chat style
                        if "message" in choice and "content" in choice["message"]:
                            ans = choice["message"]["content"].strip()
                            return {"answer": ans, "reasoning": ans, "confidence": 0.0, "time": elapsed}
                        if "text" in choice:
                            ans = choice["text"].strip()
                            return {"answer": ans, "reasoning": ans, "confidence": 0.0, "time": elapsed}
                    if "result" in data:
                        ans = data["result"]
                        return {"answer": ans, "reasoning": ans, "confidence": 0.0, "time": elapsed}
                    # Fallback: return the full response JSON for visibility
                    import json
                    return {
                        "answer": f"[Llama API: {json.dumps(data)}]",
                        "reasoning": f"[Llama API: {json.dumps(data)}]",
                        "confidence": 0.0,
                        "time": elapsed
                    }
        except Exception as e:
            elapsed = time.monotonic() - start
            return {
                "answer": f"(Mock LlamaAPIBackend: {self.name})",
                "reasoning": f"[API err: {str(e)}] => {prompt}",
                "confidence": 0.0,
                "time": elapsed
            }

class ReasoningBackend(LLMBackend):
    """
    A wrapper backend that enhances any existing backend with structured reasoning capabilities.
    
    This backend wraps another LLMBackend and modifies prompts to explicitly elicit
    structured, step-by-step reasoning based on domain-specific templates. It also 
    attempts to parse structured components from the model's response.
    
    Args:
        base_backend: The underlying LLMBackend to enhance with reasoning capabilities
        template_name: Optional specific template name to use (default: auto-detect)
        reasoning_prefix: Optional custom prefix to request structured reasoning (overrides template)
    """
    
    def __init__(self, base_backend: LLMBackend, template_name: str = None, reasoning_prefix: str = None):
        super().__init__(f"reasoning-{base_backend.name}")
        self.base_backend = base_backend
        self.template_name = template_name
        self.custom_prefix = reasoning_prefix
        
    def _get_template(self, question: str):
        """Get the appropriate template for the question."""
        from consensus_templates import detect_template, get_template_by_name
        
        # If a specific template name was provided, use that
        if self.template_name:
            return get_template_by_name(self.template_name)
        
        # Otherwise, auto-detect the most appropriate template
        return detect_template(question)
        
    def _get_prompt_prefix(self, question: str) -> str:
        """Get the appropriate prompt prefix for the question."""
        # If a custom prefix was provided, use that
        if self.custom_prefix:
            return self.custom_prefix
        
        # Otherwise, use the template's format_prompt method
        template = self._get_template(question)
        # Extract just the prefix part (everything before the question)
        prefix = template.format_prompt("").strip()
        return prefix
    
    async def complete(self, prompt: str) -> dict:
        """
        Enhance the prompt with domain-specific reasoning instructions, then send to base backend.
        Uses specialized templates to format prompts and parse responses based on the question domain.
        """
        # Get the appropriate template for this question
        template = self._get_template(prompt)
        
        # Format the prompt using the template
        if self.custom_prefix:
            # If a custom prefix was provided, use that
            template_prompt = f"{self.custom_prefix}{prompt}"
        else:
            # Otherwise, use the template's format_prompt method
            template_prompt = template.format_prompt(prompt)
        
        # Get response from base backend
        import time
        start = time.monotonic()
        result = await self.base_backend.complete(template_prompt)
        elapsed = time.monotonic() - start
        
        # Ensure time is tracked
        if "time" not in result:
            result["time"] = elapsed
            
        # Extract structured components if possible
        answer = result.get("answer", "")
        reasoning = result.get("reasoning", "")
        
        # Parse the response using the template's parser
        combined_text = answer + "\n" + reasoning
        parsed_response = template.parse_response(combined_text)
        
        # Evaluate the reasoning quality using the template's evaluator
        reasoning_quality = template.evaluate_reasoning(parsed_response)
        
        # Set reasonable confidence based on reasoning quality
        confidence = result.get("confidence", 0.0)
        # Blend existing confidence with reasoning quality if confidence is not provided
        if confidence == 0.0:
            confidence = reasoning_quality * 0.8  # Default to 80% of reasoning quality
        
        # Extract conclusion for the answer (if available)
        conclusion = ""
        for key in ["conclusion", "answer", "result"]:
            if key in parsed_response and parsed_response[key]:
                conclusion = parsed_response[key]
                break
        
        # Build enhanced response
        enhanced_result = {
            "answer": conclusion or answer,
            "reasoning": reasoning,
            "confidence": confidence,
            "time": result.get("time", elapsed),
            "template_name": template.name,
            "reasoning_quality": reasoning_quality,
        }
        
        # Add all parsed components to the result
        for key, value in parsed_response.items():
            if key != "full_response" and value:  # Skip the full response to avoid duplication
                enhanced_result[key] = value
                
        return enhanced_result
        
        # Start of next function or code block
        current_section = None
        section_content = {key: [] for key in section_markers}
        
        for line in lines:
            # Check if this line is a section header
            for section, markers in section_markers.items():
                if any(marker.lower() in line.lower() for marker in markers):
                    current_section = section
                    break
            
            # If we're in a section and this isn't a header, add the content
            if current_section and not any(
                marker.lower() in line.lower() 
                for markers in section_markers.values() 
                for marker in markers
            ):
                section_content[current_section].append(line)
        
        # Second approach: If explicit sections weren't found, look for numbered lists
        if not any(content for content in section_content.values()):
            numbered_sections = []
            current_section = []
            
            for line in lines:
                # Check for new numbered section (1., Step 1, etc.)
                if (line.strip().startswith("1.") or 
                    line.strip().startswith("1)") or
                    "step 1" in line.lower() or
                    "first" in line.lower() and len(line) < 50):
                    
                    if current_section:  # Save previous section if it exists
                        numbered_sections.append(current_section)
                    current_section = [line]
                else:
                    current_section.append(line)
            
            # Add the last section
            if current_section:
                numbered_sections.append(current_section)
            
            # If we found 2-4 sections, map them to our structure
            if 2 <= len(numbered_sections) <= 4:
                if len(numbered_sections) >= 1:
                    section_content["premises"] = numbered_sections[0]
                if len(numbered_sections) >= 2:
                    section_content["steps"] = numbered_sections[1]
                if len(numbered_sections) >= 3:
                    section_content["conclusion"] = numbered_sections[-1]
        
        # Third approach: If still no structure found, use position-based heuristic
        if not any(content for content in section_content.values()):
            total_lines = len(lines)
            if total_lines > 5:  # Only apply for sufficiently long responses
                # Approximately: first 20% = premises, last 20% = conclusion, middle = steps
                premise_end = max(1, int(total_lines * 0.2))
                conclusion_start = max(premise_end + 1, int(total_lines * 0.8))
                
                section_content["premises"] = lines[:premise_end]
                section_content["steps"] = lines[premise_end:conclusion_start]
                section_content["conclusion"] = lines[conclusion_start:]
        
        # Fourth approach: Extract conclusion directly by searching for conclusion markers
        if not section_content["conclusion"]:
            text_lower = text.lower()
            for marker in ["therefore", "thus", "in conclusion", "the answer is"]:
                if marker in text_lower:
                    marker_pos = text_lower.find(marker)
                    end_pos = text[marker_pos:].find("\n\n")
                    if end_pos == -1:
                        end_pos = len(text[marker_pos:])
                    conclusion_text = text[marker_pos:marker_pos+end_pos].strip()
                    section_content["conclusion"] = [conclusion_text]
                    break
        
        # Join the content for each section
        for section, content in section_content.items():
            if content:
                result[section] = "\n".join(content).strip()
        
        # Final fallback for conclusion: use the last paragraph if it's short
        if not result["conclusion"] and lines:
            last_paragraph = ""
            for line in reversed(lines):
                if not line.strip():
                    break
                last_paragraph = line + "\n" + last_paragraph
            
            if last_paragraph and len(last_paragraph) < 300:  # Reasonable size for conclusion
                result["conclusion"] = last_paragraph.strip()
        
        return result
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

