"""
consensus_debate.py — Utilities for multi-round debate, voting, and aggregation
Adapted from ReConcile (github.com/dinobby/ReConcile) for async, provider-agnostic consensus engine.

- Confidence-weighted and majority voting
- Multi-round debate/aggregation manager
- Structured explanation/result logic
- Debate prompt synthesis
"""

import asyncio
from typing import List, Dict, Any, Optional
from collections import Counter

################################################################################
# Types & Normalization Utilities
################################################################################

def normalize_agent_result(res: Any) -> Dict[str, Any]:
    """Normalize backend outputs (legacy str or dict) to expected dict format."""
    if isinstance(res, dict):
        return {
            "answer": str(res.get("answer", "")),
            "reasoning": str(res.get("reasoning", "")),
            "confidence": float(res.get("confidence", 0.0)),
        }
    # fallback for old str-style response
    return {"answer": str(res), "reasoning": "", "confidence": 0.0}

def trans_confidence(x: float) -> float:
    x = float(x)
    if x <= 0.6: return 0.1
    if 0.8 > x > 0.6: return 0.3
    if 0.9 > x >= 0.8: return 0.5
    if 1 > x >= 0.9: return 0.8
    if x == 1: return 1

################################################################################
# Confidence-Weighted & Majority Voting
################################################################################

def majority_vote(agent_results: List[Dict[str, Any]]) -> Optional[str]:
    """Return the most common answer (majority vote), or None if none."""
    answers = [r.get("answer", "") for r in agent_results]
    if not answers:
        return None
    return max(set(answers), key=answers.count)

def confidence_weighted_vote(agent_results: List[Dict[str, Any]]) -> Optional[str]:
    """Return the answer with max summed confidence weight."""
    weights = {}
    for r in agent_results:
        a = r.get("answer", "")
        c = trans_confidence(float(r.get("confidence", 0.0)))
        weights[a] = weights.get(a, 0.0) + c
    if not weights:
        return None
    return max(weights, key=weights.get)

def aggregate_explanations(agent_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group explanations by answer label, for debate prompt construction."""
    exps = {}
    for r in agent_results:
        ans = r.get("answer", "")
        exp = r.get("reasoning", "")
        exps.setdefault(ans, []).append(exp)
    return exps

################################################################################
# Debate Prompt Construction
################################################################################

def build_debate_prompt(agent_results: List[Dict[str, Any]]) -> str:
    """
    Synthesize a prompt aggregating model disagreements and rationales.
    Used as context for additional debate rounds.
    """
    answers = [r["answer"] for r in agent_results]
    expl_by_ans = aggregate_explanations(agent_results)
    ctr = Counter(answers).most_common(2)
    s = ""
    for ans, count in ctr:
        s += f"There are {count} agents who think the answer is '{ans}'.\n"
        reasonings = expl_by_ans.get(ans, [])
        if reasonings:
            s += "\n".join([f"One agent says: {e}" for e in reasonings])
        s += "\n\n"
    return s.strip()

################################################################################
# Debate Manager (Async, Multi-round)
################################################################################

class DebateManager:
    """
    Orchestrates async, multi-round group debate among LLM backends.
    """

    def __init__(self, backends, rounds: int = 2):
        self.backends = backends
        self.rounds = rounds

    async def debate(self, prompt: str) -> Dict[str, Any]:
        """Run N debate rounds; produce structured output comparing voting strategies."""
        context_prompts = [prompt for _ in self.backends]
        history: List[List[Dict[str, Any]]] = []
        # Per-round outputs: [[dict, dict, ...], ...]
        for r in range(self.rounds):
            # Await all models' completions for current prompts
            outs = await asyncio.gather(
                *[b.complete(context_prompts[i]) for i, b in enumerate(self.backends)]
            )
            results = [normalize_agent_result(o) for o in outs]
            history.append(results)
            # Only build new debate prompt for additional rounds
            if r < self.rounds - 1:
                debate_text = build_debate_prompt(results)
                # Next round: each model gets this joint prompt + original input
                context_prompts = [
                    f"{prompt}\n\n{debate_text}" for _ in self.backends
                ]
        # Aggregate final round
        final = history[-1]
        maj_ans = majority_vote(final)
        weighted_ans = confidence_weighted_vote(final)
        explanations = [r.get("reasoning", "") for r in final]
        return {
            "rounds": history,
            "winner_majority": maj_ans,
            "winner_weighted": weighted_ans,
            "explanations": explanations,
            "agent_outputs": final,
        }

################################################################################
# Example CLI/Engine Integration Comments (pseudocode):
################################################################################
#
# In consensus_agent.py:
#
# - Replace/augment ConsensusEngine.query and _ask to call:
#       debate_mgr = DebateManager(engine.models, rounds=3)
#       result = await debate_mgr.debate(prompt)
# - Display result["winner_majority"], result["winner_weighted"], and rationales in CLI/web UI.
#

