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

# -------------------------------------------------------------------------------
"""
CONSENSUS ENGINE & AGGREGATION LOGIC — COMPARISON DOCUMENTATION
-------------------------------------------------------------------------------

1. consensus_agent.py / consensus_backends.py (“YUKON/Toros” agent)

Consensus architecture:
- Provider-agnostic consensus engine.
- Modular LLM backend interface (`LLMBackend` abstract class in consensus_backends.py).
- Multiple LLM providers easily plugged into unified workflow via backend subclasses (OpenAI, Anthropic, Llama.cpp, etc.).

Consensus/voting logic:
- Single-round answer aggregation.
- Each model runs on the input question independently.
- Their outputs are collected into a list.
- Majority vote implemented (`majority_vote(out: List[str]) -> str`), which picks the most common answer among the models.
- No explicit confidence tracking, stepwise debate, or answer justification.

Aggregation and final answer:
- Only majority-based consensus used.
- No explanations, confidence, or iterative debate among agents.
- Focus is on routing/model invocation and simple result aggregation.

Orchestration:
- Command-line tool, web UI, and API interface; pipeline driven by user input and routed to backends.

# -------------------------------------------------------------------------------
2. ReConcile/generation.py and ReConcile/utils.py

Consensus architecture:
- Multi-model orchestration with explicit *multi-round group debate* logic.
- Integrates OpenAI, Claude, PaLM, etc., with procedural orchestration (may require some refactor for async).

Consensus/voting logic:
- Multi-stage debate & confidence tracking:
  - First, each model answers independently (“round 0”).
  - Results are aggregated and structured (answers, explanations, and model confidence).
  - “Debate prompts” are constructed highlighting disagreement and providing group context.
  - Additional rounds: models review others’ answers/explanations + debate prompts, then re-answer with more information each time.
- Voting:
  - Both simple majority vote and *confidence-weighted voting* available.
  - Votes tracked: answers (`majority_ans`), explanations, and confidence (`weighted_vote`).
  - Explicit logic for constructing the debate prompt (Counter tallies votes, explanations grouped by answer).

Aggregation and final answer:
- For each round:
  - `vote_N`: all model predictions in that round.
  - `majority_ans_N`: most common answer.
  - `weighted_max_N`: answer with the highest combined (transformed) confidence scores across models.
  - Full explanations and confidence levels included for each model and round.
  - Prompts for further debate if not all agents agree.
- Result is not just the answer:
  - Dictionary per sample containing all agent outputs, explanations, confidence, meta-evaluation fields for further analysis/experiments.
- Meta-evaluation utilities: Accuracy calculated for each voting mode, for detailed benchmarking.

Orchestration:
- Dataset-driven experimentation (batch-running over many QA tasks).
- Models called with explicit context prep including other agent answers in later rounds.

# -------------------------------------------------------------------------------

Summary Table

| Feature                   | consensus_agent.py          | ReConcile                         |
|---------------------------|----------------------------|-----------------------------------|
| Models                    | Arbitrary (pluggable sync/async backends) | OpenAI, Claude, PaLM, pluggable  |
| Aggregation Logic         | Single-round, majority vote| Multi-round, majority & confidence|
| Explanation Tracking      | No                         | Yes                               |
| Confidence Tracking       | No                         | Yes (agent self-scored + weighted)|
| Debate Mechanism          | No                         | Yes (prompt, response, re-debate) |
| Dataset/Bulk Support      | Limited                    | Integrated (SQA, Aqua, etc.)      |
| Orchestration             | CLI/API/WebUI, per-query   | Batch/task-focused                |
| Meta-Evaluation           | No                         | Yes (accuracy by method, etc.)    |

# -------------------------------------------------------------------------------

Implications for Integration

- Your agent’s backend system aligns well with ReConcile’s model calling interface.
- Current consensus logic is simple (majority vote); ReConcile provides all utilities for debate, confidence voting, and structured aggregation.
- Integrating debate and confidence scoring requires porting certain context prep, voting, and result aggregation functions, plus possible refactoring for async interface harmony.

# -------------------------------------------------------------------------------

INTERFACE INTEGRATION OPPORTUNITIES: YUKON <-> ReConcile
# -------------------------------------------------------------------------------

-- Key control points in your consensus_agent.py workflow:
   1. Model invocation/aggregation (ConsensusEngine.query): where all model answers are gathered and voted.
   2. CLI/Web debate mode: supports multiple "rounds", extensible for ReConcile-style debate, aggregation, and result structuring.

-- ReConcile debate/aggregation functions:
   - N rounds of multi-model, prompt-aware answering across multiple agents.
   - Aggregation builds explanations/confidence, selects majority/weighted, forms "debate prompt."

-- Direct integration points (with API notes):
   - After each round, aggregate using new voting/aggregation helpers.
   - Iterate, passing new debate prompt/context if agents disagree.
   - Refactor legacy paths to use dict/structured output from all backends.

-- API/async adaptation required:
   - Port aggregation helpers to run over async batch results.
   - Make backends return dict {"answer", "reasoning", "confidence"}.

-- Opportunities for code re-use:
   - Port ReConcile aggregation/debate functions as helpers.
   - Use modular backends to compose new consensus workflows.

# -------------------------------------------------------------------------------

INTEGRATION ROADMAP: ReConcile Features -> Yukon Consensus System
# -------------------------------------------------------------------------------

1. Feature Prioritization
  [P1] Structured Explanation Returns + Confidence
  [P2] Confidence-Weighted and Majority Voting
  [P3] Debate Prompt Construction & Multi-Round Debate
  [P4] Meta-Evaluation Helpers (Accuracy, Analysis)

2. Code Refactor/Adaptation Checklist
  [A] Backend Results API Harmonization
  [B] Async Wrapping/Bridging
  [C] Debate Manager / Aggregation Utility Module Boundaries
  [D] Batch vs. Single Query Normalization

3. Example Proposed Interface Stubs
  class LLMBackend(ABC):   ...   async def complete(self, prompt: str) -> dict: ...
  class DebateManager:     ...   async def debate(self, prompt: str) -> dict: ...

This roadmap enables modular, selective porting of ReConcile's advanced consensus features,
while ensuring compatibility and maintainability with your async, backend-agnostic system.
# -------------------------------------------------------------------------------
"""
import asyncio
import os
import argparse
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

    async def query(self, prompt: str, use_debate: bool = True, rounds: int = 3):
        """
        If use_debate, run multi-round consensus using DebateManager.
        Otherwise, fall back to single-round legacy mode for testing/fallback.
        """
        if use_debate:
            from consensus_debate import DebateManager
            debate_mgr = DebateManager(self.models, rounds=rounds)
            result = await debate_mgr.debate(prompt)
            # For legacy interface: expose agent_outputs list and majority/weighted answer
            outs = [(self.models[i].name, agent) for i, agent in enumerate(result["agent_outputs"])]
            ans_majority = result.get("winner_majority")
            ans_weighted = result.get("winner_weighted")
            explanations = result.get("explanations", [])
            return {
                "results": outs,
                "majority": ans_majority,
                "confidence_weighted": ans_weighted,
                "explanations": explanations,
                "rounds": result["rounds"]
            }
        # --- Legacy single-round fallback mode
        outs = await asyncio.gather(*(m.complete(prompt) for m in self.models), return_exceptions=True)
        results = []
        for i, o in enumerate(outs):
            # o is guaranteed dict as of new backend API
            if isinstance(o, dict) and o.get("answer"):
                results.append((self.models[i].name, o))
            elif isinstance(o, str) and o:
                results.append((self.models[i].name, {"answer": o, "reasoning": "", "confidence": 0.0}))
        if not results:
            raise RuntimeError("All model calls failed")
        answers = [res[1]["answer"] for res in results]
        ans = self.vote_fn(answers)
        if not self.safety_fn(ans):
            raise RuntimeError("Consensus answer failed safety gate")
        return {
            "results": results,
            "majority": ans,
            "confidence_weighted": None,
            "explanations": [r[1].get("reasoning", "") for r in results],
            "rounds": [ [r[1] for r in results] ]
        }

###############################################################################
# Build default engine                                                        #
###############################################################################

def build_model_registry() -> List['LLMBackend']:
    """Build and return all *available* LLMBackend instances, unscreened for CLI/API selection."""
    from consensus_backends import (
        LlamaCppBackend,
        OllamaBackend,
        OpenAIBackend,
        AnthropicBackend,
        LlamaAPIBackend,
        ReasoningBackend,
    )
    m: List['LLMBackend'] = []
    if path := os.getenv("LLAMA_MODEL"):
        m.append(LlamaCppBackend("llama-local", Path(path)))
    # Define reasoning-optimized models - these models are particularly good at step-by-step reasoning
    reasoning_models = [
        OllamaBackend("llama3-70b", "llama3:70b"),
        OllamaBackend("gemma2", "gemma2:latest"),
        OllamaBackend("mistral", "mistral:latest"),
        OllamaBackend("deepseek-coder", "deepseek-coder:latest"),  # Updated to use available model
    ]
    
    # Standard models
    standard_models = [
        OllamaBackend("ARG_A", "ARG_A:latest"),
        OllamaBackend("ARG_B", "ARG_B:latest"),
        OllamaBackend("gemma", "gemma:latest"),
        OllamaBackend("llama3-abliterated", "superdrew100/llama3-abliterated:latest"),
    ]
    
    # Add all models to registry
    m += standard_models
    
    # Add reasoning-focused models
    m += reasoning_models
    # if k := os.getenv("OPENAI_API_KEY"):
    #     m.append(OpenAIBackend("gpt4o", "gpt-4o", "https://api.openai.com/v1", k))
    # if k := os.getenv("META_API_KEY"):
    #     m.append(OpenAIBackend("llama3", "llama3-70b-chat", "https://api.meta.ai/v1", k))
    # Add Claude 3 Opus (strong reasoning capabilities)
    if k := os.getenv("ANTHROPIC_API_KEY"):
        claude_backend = AnthropicBackend("claude3", "claude-3-opus-20240229", k)
        m.append(claude_backend)
        # Also add a reasoning-enhanced version of Claude
        m.append(ReasoningBackend(claude_backend))
        
    # Add Llama API backend if LLAMA_API_KEY is set
    if k := os.getenv("LLAMA_API_KEY"):
        llama_backend = LlamaAPIBackend(
            "llama-api",
            "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "https://api.llama.com/v1/chat/completions",
            k,
        )
        m.append(llama_backend)
        # Also add a reasoning-enhanced version of Llama API
        m.append(ReasoningBackend(llama_backend))
    return m

def build_engine(selected_names: List[str]|None = None) -> ConsensusEngine:
    """
    Build a ConsensusEngine with only the selected model names.
    If selected_names is None, use all available models.
    """
    registry = build_model_registry()
    if selected_names is None:
        filtered = registry
    else:
        names_lower = [n.lower() for n in selected_names]
        filtered = [b for b in registry if b.name.lower() in names_lower]
        # fallback to all if filter gives nothing
        if not filtered:
            filtered = registry[:3]
    model_names = [backend.name for backend in filtered]
    print(f"[build_engine] Active backend models: {model_names}")
    return ConsensusEngine(filtered)

###############################################################################
# FastAPI app (unchanged from v5, trimmed)                                    #
###############################################################################
# … [make_app(engine) definition unchanged] …
###############################################################################
# CLI                                                                         #
###############################################################################

def print_help_and_settings(engine, all_available_models=None, default_rounds=3):
    from rich.panel import Panel
    import os
    # Prepare the flags section as a consistently aligned table
    flag_lines = [
        ("-q",     "PROMPT", "Run a single CLI question and exit"),
        ("--web",  "",       "Start browser/server UI (http://localhost:8000)"),
        ("--port", "N",      "Set web server port (default 8000)"),
        ("--mode", "MODE",   'Select output mode: "transparent" (default, shows full debate) or "quiet" (progress bar + verdict only)'),
        ("--help", "",       "Show this usage info"),
    ]
    flag_fmt = "  [cyan]{:<7}[/cyan] [magenta]{:<8}[/magenta]  [white]{}[/white]"
    flag_table = "[bold yellow]Flags:[/bold yellow]\n" + "\n".join(
        flag_fmt.format(flag, param, desc) for flag, param, desc in flag_lines
    )

    help_text = f"""[bold white]Usage:[/bold white]  [green]python consensus_agent.py [--web] [-q 'Your question'][/green]

{flag_table}

[bold yellow]CLI Example:[/bold yellow]
  [green]python consensus_agent.py -q "What is the capital of France?"[/green]

[bold white]Interactive Use:[/bold white]
  [green]python consensus_agent.py[/green] and enter your query at the [bold green]>>[/] prompt.

[bold yellow]Shortcuts:[/bold yellow]  [cyan]Ctrl+C[/cyan] or EOF (Ctrl+D) to quit.
"""

    # --- Settings summary
    actives = [m.name for m in engine.models]
    rounds = default_rounds
    available = [m for m in all_available_models] if all_available_models else actives
    env_keys = []
    for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLAMA_API_KEY", "LLAMA_MODEL"]:
        present = os.getenv(k, False)
        env_keys.append(f"{k}=[{'set' if present else 'unset'}]")
    settings = (
        f"[bold white]Available LLMs:[/bold white] [magenta]{', '.join(available)}[/magenta]\n"
        f"[bold white]Active LLMs (selected):[/bold white] [cyan]{', '.join(actives)}[/cyan]\n"
        f"[bold white]Debate Rounds:[/bold white] [green]{rounds}[/green]\n"
        f"[bold white]Environment/Keys:[/bold white] {'  '.join(env_keys)}\n"
    )

    help_panel = Panel.fit(help_text, style="bold magenta", border_style="magenta", title="[bold yellow]HELP[/bold yellow]", title_align="left")
    settings_panel = Panel.fit(settings, style="bold green", border_style="green", title="[bold white]SETTINGS[/bold white]", title_align="left")
    console.print(help_panel)
    console.print(settings_panel)

def run_cli(engine: ConsensusEngine, query: str | None, mode: str = "transparent", all_available_models=None):
    console.print(f"[cyan]{BANNER}[/]")
    print_help_and_settings(engine, all_available_models=all_available_models)
    if query:
        asyncio.run(_ask(engine, query, mode=mode))
        return
    while True:
        try:
            q = Prompt.ask("[bold green]>>[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break
        asyncio.run(_ask(engine, q, mode=mode))

async def _ask(engine: ConsensusEngine, q: str, mode: str = "transparent"):
    num_rounds = 3  # debate rounds if using new consensus
    use_debate = True  # set to False for legacy fallback mode (single round)
    import time
    start_time = time.monotonic()

    # -------------------------
    # Progress bar for "quiet" mode:
    show_full = (mode == "transparent")
    result = None
    if not show_full:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        bar = Progress(
            SpinnerColumn(),
            TextColumn("[yellow]Debating…[/yellow]"),
            BarColumn(),
            TimeElapsedColumn(),
            expand=True
        )
        with bar:
            # Assume debate across N rounds: estimate progress as rounds complete
            # If backend implementation gets progress callback in future, hook it here.
            task = bar.add_task("[bold green]Consensus running...", total=num_rounds)
            # We'll just tick the progress bar once per expected round for now.
            from asyncio import sleep
            for n in range(num_rounds):
                # Simulate per-round progress—remove if engine exposes callback in the future
                bar.update(task, advance=1)
                await sleep(0.1)  # Just animation pacing until result returns
            try:
                result = await engine.query(q, use_debate=True, rounds=num_rounds)
            except Exception as e:
                bar.stop()
                console.print(f"[red]Error:[/] {e}")
                return
            bar.update(task, completed=num_rounds)
    else:
        try:
            with console.status("[bold yellow]Aggregating multi-model consensus…[/]"):
                result = await engine.query(q, use_debate=use_debate, rounds=num_rounds)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            return

    # ---------- Output/results ----------
    rounds = result.get("rounds", [])
    model_names = [m.name for m in engine.models]
    from rich.panel import Panel
    from rich.text import Text
    import time
    total_time = time.monotonic() - start_time

    if show_full:
        console.rule("[bold white]🗣️ Debate History (Round-by-Round)[/]")
        for ridx, round_results in enumerate(rounds, 1):
            console.rule(f"[bold yellow]Round {ridx}[/]", style="yellow", characters="─")
            from rich.table import Table
            round_table = Table(show_header=True, header_style="bold magenta", box=None, expand=False, padding=(0,1))
            round_table.add_column("Model", style="bold cyan")
            round_table.add_column("Answer", style="white")
            round_table.add_column("Conf.", justify="center", style="green")
            round_table.add_column("Conf. Reasoning Time", style="yellow", justify="center")
            round_table.add_column("Reasoning", style="dim", overflow="fold")
            for midx, resp in enumerate(round_results):
                model_name = model_names[midx] if midx < len(model_names) else f"Model{midx+1}"
                ans = str(resp.get("answer", "")) if isinstance(resp, dict) else str(resp)
                reasoning = str(resp.get("reasoning", "")) if isinstance(resp, dict) else ""
                confidence = f"{float(resp.get('confidence', 0.0)):.2f}" if isinstance(resp, dict) else "0.00"
                tval = resp.get("time", None)
                timing_s = f"{tval:.2f}s" if isinstance(tval, (float, int)) else "—"
                round_table.add_row(
                    f"[cyan]{model_name}[/]",
                    f"[white]{ans}[/]",
                    f"[green]{confidence}[/]",
                    f"[yellow]{timing_s}[/]",
                    f"[dim]{reasoning}[/]"
                )
            console.print(round_table)
            console.print()
        # Display per-model responses for the final round
        final_panel = Panel.fit(
            "[b white]Model Responses (Final Round)[/]",
            border_style="blue",
            title="[bold blue] Final Output [/]",
            title_align="center",
        )
        console.print(final_panel)
        fin_table = Table(show_header=True, header_style="bold magenta", box=None, expand=False, padding=(0,1))
        fin_table.add_column("Model", style="bold cyan")
        fin_table.add_column("Answer", style="white")
        fin_table.add_column("Conf.", style="green")
        fin_table.add_column("Conf. Reasoning Time", style="yellow", justify="center")
        fin_table.add_column("Reasoning", style="dim", overflow="fold")
        for model, resp in result["results"]:
            ans = resp.get("answer", "") if isinstance(resp, dict) else str(resp)
            reasoning = resp.get("reasoning", "") if isinstance(resp, dict) else ""
            confidence = f"{float(resp.get('confidence', 0.0)):.2f}" if isinstance(resp, dict) else "0.00"
            tval = resp.get("time", None)
            timing_s = f"{tval:.2f}s" if isinstance(tval, (float, int)) else "—"
            fin_table.add_row(
                f"[cyan]{model}[/]",
                f"[white]{ans}[/]",
                f"[green]{confidence}[/]",
                f"[yellow]{timing_s}[/]",
                f"[dim]{reasoning}[/]"
            )
        console.print(fin_table)

    # Display consensus verdicts as boxed/highlighted text — always show these
    consensus_txt = Text.assemble(
        ("Consensus by Majority: ", "bold white"), (str(result["majority"]), "green bold")
    )
    console.rule("[bold]Verdicts[/]", style="bold green", characters="─")
    console.print(Panel.fit(consensus_txt, border_style="green", subtitle="Majority", subtitle_align="right"))
    if result.get("confidence_weighted") is not None:
        wtxt = Text.assemble(
            ("Consensus by Confidence-Weighted: ", "bold white"), (str(result["confidence_weighted"]), "yellow bold")
        )
        console.print(Panel.fit(wtxt, border_style="yellow", subtitle="Weighted", subtitle_align="right"))

    # Rationales (skip in quiet if none)
    if show_full or result.get("explanations"):
        if result.get("explanations"):
            console.print("\n[yellow bold]Rationales:[/]")
            for i, expl in enumerate(result.get("explanations", [])):
                timing = None
                # Try to find corresponding timing from agent_outputs/final round
                try:
                    timing = result.get("agent_outputs", [])[i].get("time")
                except Exception:
                    timing = None
                timing_s = f"{timing:.2f}s" if isinstance(timing, (float, int)) else "—"
                if expl:
                    console.print(f"  [bold cyan]{model_names[i]}[/]: [yellow]{timing_s}[/] [dim]{expl}[/]")
    
    # Display error statistics if available
    if "error_count" in result and "total_count" in result:
        error_count = result["error_count"]
        total_count = result["total_count"]
        error_pct = (error_count / total_count) * 100 if total_count > 0 else 0
        
        # Choose color based on error percentage
        error_color = "green" if error_pct < 25 else "yellow" if error_pct < 50 else "red"
        
        # Display error statistics
        error_text = f"[{error_color}]Error responses: {error_count}/{total_count} ({error_pct:.1f}%)[/]"
        console.print(error_text)
        
        # Display warning if high error rate
        if error_pct >= 50:
            console.print("[bold red]Warning:[/bold red] High rate of error responses detected. Results may be unreliable.")
        elif error_pct >= 25:
            console.print("[bold yellow]Note:[/bold yellow] Some error responses detected. Consider checking model availability.")
            
    # Summary / aggregation time
###############################################################################
# Entrypoint                                                                  #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yukon Systems TOROS Consensus LLM")
    parser.add_argument("-q", "--query", help="Prompt to ask via CLI")
    parser.add_argument("--web", action="store_true", help="Start web server instead of CLI")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    parser.add_argument("--mode", choices=["transparent", "quiet"], default="transparent", help="Output mode: full debate or only verdict & timing (with progress bar)")
    parser.add_argument("--models", help="Comma-separated list of model names (e.g., gemma,arb_a,llama3-abliterated) to enable")
    args = parser.parse_args()

    # Build the registry just once early for prompt and help display
    all_registry = build_model_registry()
    all_available_names = [m.name for m in all_registry]

    selected_models = None

    if args.models:
        # Use the --models flag to select which backends to use
        wanted = [s.strip() for s in args.models.split(",") if s.strip()]
        selected_models = wanted
    else:
        # Interactive selection if not provided (just in CLI mode)
        if not args.web:
            from rich.prompt import Prompt
            from rich.table import Table

            console.print("[magenta bold]Choose which models to ENABLE for this session:[/]")
            table = Table(show_header=True, header_style="bold magenta", box=None, expand=False, padding=(0,1))
            table.add_column("#", style="bold cyan", justify="center")
            table.add_column("Model Name", style="bold", justify="left")
            table.add_column("Backend/Tag", style="dim", justify="left")
            for idx, backend in enumerate(all_registry):
                # For readability, display backend tag if available
                tag_hint = getattr(backend, "tag", getattr(backend, "model_path", ""))
                table.add_row(f"{idx+1}", backend.name, str(tag_hint))
            console.print(table)
            console.print("[yellow]Enter [cyan]numbers[/cyan] (comma-separated) for models to enable (e.g., [green]1,3,4[/green]):")
            raw = Prompt.ask("Models", default="1,2,3")
            try:
                nums = [int(s)-1 for s in raw.split(",") if s.strip().isdigit()]
                selected_names = [all_registry[i].name for i in nums if 0 <= i < len(all_registry)]
                if not selected_names:
                    # fallback to first three
                    selected_names = all_available_names[:3]
                selected_models = selected_names
            except Exception:
                selected_models = all_available_names[:3]

    # If nothing selected by flag or prompt, default to first 3
    if not selected_models or not any(n in all_available_names for n in selected_models):
        selected_models = all_available_names[:3]

    eng = build_engine(selected_models)

    if args.web:
        import uvicorn
        from consensus_web import make_app  # pseudo import placeholder
        uvicorn.run(make_app(eng), host="0.0.0.0", port=args.port)
    else:
        run_cli(eng, args.query, mode=args.mode, all_available_models=all_available_names)
