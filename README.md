# Yukon Systems: Toros — Multi-Model, Multi-Round Consensus LLM Engine

![Toros ASCII Banner](https://raw.githubusercontent.com/<your-org>/<your-repo>/main/assets/toros_banner.png) <!-- (optional: replace with your own banner or text) -->

## Overview

Yukon Systems Toros is an extensible, provider-agnostic **multi-model LLM consensus engine** supporting:

- Parallel, multi-round debate cycles among any number of models (OpenAI, Anthropic, Ollama, Llama.cpp, and more).
- CLI and beautiful, color-formatted output for stepwise reasoning and aggregated consensus.
- Flexible backend system for plugging in additional LLMs or inference engines.
- (Optional) Web UI and API endpoints via FastAPI.

The engine enables richer, more robust LLM evaluation by orchestrating logical "debates" among separate agents, each presenting their answers, reasoning, and confidence scores. Consensus is reached through majority and confidence-weighted voting, displayed round-by-round in a professional, easy-to-digest visual summary.

---

## Features

- Async multi-round debate engine
- Majority and confidence-weighted voting
- Pluggable backend system (Ollama, Llama.cpp, OpenAI, Anthropic, etc.)
- Rich CLI with colorized output and progress bars
- Optional FastAPI web API

---

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```
To use Ollama-served models, install [Ollama](https://ollama.com/) and make sure `ollama` daemon is running locally.

### 2. (Optional) Set Environment Variables

Configure API keys or model files in your environment:

```bash
export LLAMA_API_KEY=sk-your-llama-key
export ANTHROPIC_API_KEY=sk-your-anthropic-key
export OPENAI_API_KEY=sk-your-openai-key
export LLAMA_MODEL=/models/llama.bin      # for Llama.cpp local
```

### 3. Run in CLI Mode

```bash
python consensus_agent.py -q "Is Python better than Rust?"
```

Or start interactive shell:

```bash
python consensus_agent.py
```

You’ll see an ASCII art title and a prompt (`>>`). Enter your question and see round-by-round debate, rationales, and a highlighted consensus!

### 4. Multi-Model Example (Ollama + Llama API + More)

By default, the CLI will use all of: `Ollama (gemma)`, `Ollama (llama3-abliterated)`, and any API-backed models you have configured.

### 5. (Optional) Start as Web Server

```bash
python consensus_agent.py --web
```
Browse to [http://localhost:8000](http://localhost:8000). Use `POST /api/ask` to query the engine programmatically.

---

## Example CLI Output

```
───────────────── 🗣️ Debate History (Round-by-Round) ─────────────────
─────────────── Round 1 ───────────────
 Model          Answer                    Conf.    Reasoning
 gemma          Paris is the capital...   0.00     Ollama (gemma): ...
 llama3-ablit.  Paris, France             0.00     Ollama (llama3-ablit.)...
 llama-api      Paris                     0.00   

─────────────── Round 2 ───────────────
 Model              Answer          Conf.  Reasoning
 gemma              ...
 llama3-ablit.      ...
 llama-api          ...

╭──────────── Final Output ────────────╮
│ Model Responses (Final Round)        │
╰──────────────────────────────────────╯
 Model          Answer                     Conf.   Reasoning
 gemma          Paris is ...               0.00    ...
 llama3-ablit.  ...                                ...
 llama-api      ...                                ...

Consensus by Majority: Paris
Consensus by Confidence-Weighted: Paris

──────────────────────────── Rationales ────────────────────────────
 gemma: Ollama (gemma): returned response string
 llama3-ablit.: Ollama (llama3-ablit.): returned response string

──────────────────────────── Total consensus time: 9.32s ──────────
```

---

## How to Add More Models

- Enable additional API keys in your environment; supported out of the box: OpenAI, Anthropic, Llama.cpp, Llama API, Ollama.
- To add more Ollama local models, just duplicate/extend the lines in `build_engine()` in `consensus_agent.py`:
    ```python
    OllamaBackend("gemma", "gemma:latest"),
    OllamaBackend("llama3-abliterated", "superdrew100/llama3-abliterated:latest"),
    # Add more as desired
    ```

---

## Project Structure

| File                   | Description                              |
|------------------------|------------------------------------------|
| consensus_agent.py     | Main CLI, web, and engine orchestrator   |
| consensus_backends.py  | Model backend adapters                   |
| consensus_debate.py    | Debate, aggregation and prompt logic     |
| README.md              | This file                                |

---

## Attribution

- Project inspired in part by [dinobby/ReConcile](https://github.com/dinobby/ReConcile)
- Uses `rich` and `pyfiglet` for CLI enhancements.

---

## License

MIT License (c) Yukon Systems

