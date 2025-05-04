# Yukon Systems: Multi-Model LLM Consensus Engine 🏔️

A **provider-agnostic, multi-model consensus engine** for Large Language Models (LLMs) with both a rich web UI and command-line interface (CLI). Easily compare, combine, and reach consensus across multiple LLMs for more robust, reliable answers.

---

## Features

- **Web UI:** Chat interface at [http://localhost:8000](http://localhost:8000) (Tailwind + htmx).
- **API:** JSON/SSE API endpoints (`/api/ask`, `/api/stream`).
- **CLI:** Query multiple LLMs and get consensus directly from your terminal.
- **Provider Agnostic:** Supports local and remote LLMs via pluggable backends.
- **Consensus & Safety:** Flexible voting (e.g. majority) and safety gates baked in.

---

## Installation & Setup

First, clone this repository, then install dependencies (Python 3.9+ recommended):

```bash
pip install fastapi uvicorn htmx-tailwind aiohttp sse-starlette \
            openai anthropic llama-cpp-python pyfiglet rich
```

Some features require optional dependencies for specific providers.

---

## Usage

### Run the Command-Line Interface (CLI)

Minimal use:
```bash
python consensus_agent.py -q "What is the capital of France?"
```
Or launch interactive mode:
```bash
python consensus_agent.py
```

Sample startup (see the classic banner!):
```
             YUKON SYSTEMS: TOROS
```
_Sample output:_
```
[llama-local]: (Mock response...)
[gemma]: (Mock response...)
Consensus: Paris
```

### Start the Web UI

```bash
python consensus_agent.py --web
# Then open http://localhost:8000 in your browser
```

### API Endpoints

- `POST /api/ask` — JSON endpoint for prompt/response.
- `GET /api/stream` — Server-sent events (SSE) for live output.

---

## Backends & Extensibility

Backends are implemented in [`consensus_backends.py`](./consensus_backends.py). Each backend subclasses a shared `LLMBackend` interface, exposing `async def complete(prompt: str) -> str`.

Included mock backends:

- **LlamaCppBackend:** Stub for local [llama.cpp](https://github.com/ggerganov/llama.cpp) models.
- **OllamaBackend:** Talks to [Ollama](https://ollama.com/) or mocks responses.
- **OpenAIBackend:** (Mock/stub) for OpenAI, Azure, Meta, etc.
- **AnthropicBackend:** (Mock/stub) for Claude API.
- **LlamaAPIBackend:** Real/remote API for Llama, supports real HTTP calls.

For a real environment, plug in your API keys and endpoints by configuring environment variables.

You can easily add support for other LLMs by extending `LLMBackend` and implementing the `complete()` method.

---

## Consensus Logic

- **Voting:** The default is majority vote (`majority_vote`), but this is easily swappable.
- **Safety:** Basic content filter (`basic_safety`) can be adapted for your needs.
- Both are configurable via `ConsensusEngine` in [`consensus_agent.py`](./consensus_agent.py):

```python
ConsensusEngine(
    models=[...], 
    vote_fn=majority_vote, 
    safety_fn=basic_safety
)
```

---

## Configuration

Set these environment variables to enable specific model providers:

- `LLAMA_MODEL`: Path to a local llama.cpp GGUF file.
- `OPENAI_API_KEY`: OpenAI API key for remote GPT models.
- `ANTHROPIC_API_KEY`: Claude/Anthropic API key.
- `LLAMA_API_KEY`: Key for remote Llama API access.
- `META_API_KEY`: Meta Llama-3 API key.

You can set them per session, for example:
```bash
export LLAMA_MODEL=/models/llama.bin
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

---

## Extending with New Backends

To support a new LLM or API:

1. Subclass `LLMBackend` in `consensus_backends.py`.
2. Implement the `async complete(prompt: str)` method.
3. Add your backend to `build_engine()` in `consensus_agent.py`.

---

## License

[MIT](./LICENSE) &copy; Robin Winters

---

Enjoy robust LLM consensus with Yukon Systems! ❄️

