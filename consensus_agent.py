"""
consensus_agent.py (v6) — Rich WebUI + CLI Edition 🏔️
====================================================
Provider‑agnostic **multi‑model consensus engine** with:

• Tailwind + htmx browser chat (http://localhost:8000)
• JSON / SSE API endpoints (`/api/ask`, `/api/stream`)
• **Command‑line interface** (CLI):
  ```bash
  python consensus_agent.py -q "Your question here"
  ```
  Prints a **Toros‑style ASCII banner** on start‑up:

```
               YUKON SYSTEMS : TOROS
```

Dependencies
------------
```bash
pip install fastapi uvicorn htmx-tailwind aiohttp sse-starlette \
            openai anthropic llama-cpp-python pyfiglet rich
```
`pyfiglet` generates the ASCII art; `rich` adds color to CLI output.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from rich.console import Console
from rich.prompt import Prompt

###############################################################################
# ASCII banner                                                                #
###############################################################################

console = Console()

try:
    import pyfiglet
except ImportError:  # pragma: no cover
    console.print("[bold red]pyfiglet not installed; run `pip install pyfiglet`[/]")
    raise

BANNER = pyfiglet.figlet_format("YUKON SYSTEMS: TOROS", font="slant")

###############################################################################
# Backend interfaces + implementations (same as v5, trimmed for brevity)      #
###############################################################################
# … [back‑end classes unchanged] …
###############################################################################
# Consensus engine                                                            #
###############################################################################

def majority_vote(out: List[str]) -> str:  # noqa: D401
    return max(set(out), key=out.count)

def basic_safety(t: str) -> bool:
    banned = ("kill", "bomb", "child")
    return not any(b in t.lower() for b in banned)

@dataclass
class ConsensusEngine:
    models: List['LLMBackend']
    vote_fn: Callable[[List[str]], str] = majority_vote
    safety_fn: Callable[[str], bool] = basic_safety

    async def query(self, prompt: str):
        # Gather all (model_name, answer) pairs
        outs = await asyncio.gather(*(m.complete(prompt) for m in self.models), return_exceptions=True)
        results = []
        for i, o in enumerate(outs):
            if isinstance(o, str) and o:
                results.append((self.models[i].name, o))
        if not results:
            raise RuntimeError("All model calls failed")
        # Use only the answers for voting/safety as before, but return (name, answer) pairs
        answers = [res[1] for res in results]
        ans = self.vote_fn(answers)
        if not self.safety_fn(ans):
            raise RuntimeError("Consensus answer failed safety gate")
        return results, ans

###############################################################################
# Build default engine                                                        #
###############################################################################

def build_engine() -> ConsensusEngine:
    from consensus_backends import (
        LlamaCppBackend,
        OllamaBackend,
        OpenAIBackend,
        AnthropicBackend,
        LlamaAPIBackend,
    )  # pseudo‑import for brevity

    m: List['LLMBackend'] = []
    if path := os.getenv("LLAMA_MODEL"):
        m.append(LlamaCppBackend("llama-local", Path(path)))
    m += [OllamaBackend("gemma", "gemma:latest")]
    # if k := os.getenv("OPENAI_API_KEY"):
    #     m.append(OpenAIBackend("gpt4o", "gpt-4o", "https://api.openai.com/v1", k))
    # if k := os.getenv("META_API_KEY"):
    #     m.append(OpenAIBackend("llama3", "llama3-70b-chat", "https://api.meta.ai/v1", k))
    if k := os.getenv("ANTHROPIC_API_KEY"):
        m.append(AnthropicBackend("claude3", "claude-3-opus-20240229", k))
    # Add Llama API backend if LLAMA_API_KEY is set
    if k := os.getenv("LLAMA_API_KEY"):
        m.append(
            LlamaAPIBackend(
                "llama-api",
                "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "https://api.llama.com/v1/chat/completions",
                k,
            )
        )
    return ConsensusEngine(m)

###############################################################################
# FastAPI app (unchanged from v5, trimmed)                                    #
###############################################################################
# … [make_app(engine) definition unchanged] …
###############################################################################
# CLI                                                                         #
###############################################################################

def run_cli(engine: ConsensusEngine, query: str | None):
    console.print(f"[cyan]{BANNER}[/]")
    if query:
        asyncio.run(_ask(engine, query))
        return
    while True:
        try:
            q = Prompt.ask("[bold green]>>[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break
        asyncio.run(_ask(engine, q))

async def _ask(engine: ConsensusEngine, q: str):
    num_rounds = 3  # Total back-and-forth rounds (including first)
    prompts = [q, q]  # One per model
    if len(engine.models) != 2:
        console.print("[red]Debate mode is supported for exactly two models.[/]")
        # fallback to regular "all answer once"
        with console.status("[bold yellow]Thinking…[/]"):
            try:
                results, ans = await engine.query(q)
            except Exception as e:
                console.print(f"[red]Error:[/] {e}")
                return
        for model, resp in results:
            console.print(f"[magenta][{model}][/]: {resp}")
        console.print(f"\n[bold white]Consensus:[/] {ans}\n")
        return

    # Two models: explicit debate mode
    model_names = [m.name for m in engine.models]
    last_responses = [q, q]
    import time
    overall_start = time.monotonic()
    per_model_times = [[], []]  # times[i][round]
    for round_i in range(num_rounds):
        with console.status(f"[bold yellow]Round {round_i+1}…[/]"):
            outs = []
            times = []
            # Always [0] then [1]
            for idx in (0, 1):
                prompt = prompts[idx]
                model = model_names[idx]
                console.print(f"[yellow][{model} | round {round_i+1} | thinking...][/]")
                t0 = time.monotonic()
                try:
                    resp = await engine.models[idx].complete(prompt)
                except Exception as e:
                    resp = f"[Error: {e}]"
                delta = time.monotonic() - t0
                times.append(delta)
                outs.append(resp)
                per_model_times[idx].append(delta)
        # Print both responses with timing
        for idx, (model, resp) in enumerate(zip(model_names, outs)):
            t = times[idx]
            console.print(f"[bold cyan][{model} | round {round_i+1} | {t:.2f}s][/]: {resp}")
        # For next round, alternating: give each model the other's answer as prompt
        prompts = [outs[1], outs[0]]
    total_time = time.monotonic() - overall_start

    # Optionally, majority vote at the end using all last responses
    try:
        ans = engine.vote_fn(outs)
        if not engine.safety_fn(ans):
            raise RuntimeError("Consensus answer failed safety gate")
        console.print(f"\n[bold white]Consensus:[/] {ans}\n")
        console.print(
            f"[green]Total debate time:[/] {total_time:.2f}s "
            f"({model_names[0]} avg/round: {sum(per_model_times[0])/num_rounds:.2f}s, "
            f"{model_names[1]} avg/round: {sum(per_model_times[1])/num_rounds:.2f}s)\n"
        )
    except Exception as e:
        console.print(f"[yellow]No consensus or failed safety: {e}[/]")
        console.print(
            f"[green]Total debate time:[/] {total_time:.2f}s\n"
        )

###############################################################################
# Entrypoint                                                                  #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yukon Systems TOROS Consensus LLM")
    parser.add_argument("-q", "--query", help="Prompt to ask via CLI")
    parser.add_argument("--web", action="store_true", help="Start web server instead of CLI")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    args = parser.parse_args()

    eng = build_engine()

    if args.web:
        import uvicorn
        from consensus_web import make_app  # pseudo import placeholder
        uvicorn.run(make_app(eng), host="0.0.0.0", port=args.port)
    else:
        run_cli(eng, args.query)

