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
    """Normalize backend outputs (legacy str or dict) to expected dict format, including 'time'."""
    if isinstance(res, dict):
        out = {
            "answer": str(res.get("answer", "")),
            "reasoning": str(res.get("reasoning", "")),
            "confidence": float(res.get("confidence", 0.0)),
        }
        # Hard requirement: propagate timing if available
        if "time" in res:
            out["time"] = float(res["time"])
        return out
    # fallback for old str-style response
    return {"answer": str(res), "reasoning": "", "confidence": 0.0, "time": 0.0}

def is_error_response(response: Dict[str, Any]) -> bool:
    """
    Detect if a response appears to be an error message rather than a valid answer.
    
    Args:
        response: The agent response dictionary
        
    Returns:
        bool: True if the response appears to be an error, False otherwise
    """
    answer = str(response.get("answer", "")).lower()
    reasoning = str(response.get("reasoning", "")).lower()
    
    # Check for common error patterns
    error_patterns = [
        "api error", 
        "error", 
        "not found", 
        "failed", 
        "cannot connect",
        "timeout",
        "unavailable",
        "404"
    ]
    
    # First check if the answer starts with an error bracket or includes error text
    if (answer.startswith("[") and "error" in answer.lower()) or \
       any(pattern in answer for pattern in error_patterns):
        return True
    
    # Then check if the reasoning contains error messages
    if any(pattern in reasoning for pattern in error_patterns) and len(answer) < 100:
        # Only count it as an error if the answer is short - longer answers with
        # minor error mentions in reasoning are probably still valid responses
        return True
        
    # Check for JSON error responses
    if answer.startswith("{") and "error" in answer:
        return True
        
    return False

def trans_confidence(x: float) -> float:
    x = float(x)
    if x <= 0.6: return 0.1
    if 0.8 > x > 0.6: return 0.3
    if 0.9 > x >= 0.8: return 0.5
    if 1 > x >= 0.9: return 0.8
    if x == 1: return 1
    
def assess_reasoning_quality(reasoning: str) -> float:
    """
    Evaluate the quality of reasoning based on structure, logical markers, and completeness.
    Enhanced to better identify and reward structured reasoning patterns.
    
    Args:
        reasoning: The reasoning text to evaluate
        
    Returns:
        float: A quality score between 0.0 and 1.0
    """
    # If the reasoning appears to be an error message, return 0.0
    error_patterns = ["error", "not found", "failed", "cannot connect", "timeout", "unavailable"]
    if any(pattern in reasoning.lower() for pattern in error_patterns) and len(reasoning) < 200:
        return 0.0
    
    quality = 0.0
    reasoning_lower = reasoning.lower()
    
    # PREMISES SECTION: Check for explicit premises/assumptions section (0.25 max)
    premises_markers = ["premise", "assumption", "given", "initial fact", "precondition", "starting point"]
    has_explicit_premises = any(marker in reasoning_lower for marker in premises_markers)
    if has_explicit_premises:
        quality += 0.25
    
    # STRUCTURE: Check for structured format (numbered points, steps, etc.) (0.25 max)
    structure_markers = ["step 1", "first", "1.", "point 1", "initially", "step-by-step"]
    
    # Check for numbered list structure
    numbered_structure = False
    for i in range(1, 5):  # Check for patterns like "1.", "2.", etc.
        if f"{i}." in reasoning or f"{i})" in reasoning or f"step {i}" in reasoning_lower:
            numbered_structure = True
            break
            
    if numbered_structure:
        quality += 0.25
    elif any(marker in reasoning_lower for marker in structure_markers):
        quality += 0.15  # Less points for just mentioning structure without clear numbering
    
    # LOGICAL FLOW: Check for logical connectors and reasoning markers (0.25 max)
    logical_markers = ["therefore", "because", "since", "given that", "it follows", 
                      "consequently", "thus", "hence", "as a result"]
    marker_count = sum(1 for marker in logical_markers if marker in reasoning_lower)
    quality += min(0.25, marker_count * 0.05)  # 0.05 points per marker, up to 0.25
    
    # CONCLUSION: Check for explicit conclusion (0.15 max)
    conclusion_markers = ["in conclusion", "to conclude", "the answer is", 
                         "finally", "in summary", "to summarize", "therefore"]
    if any(marker in reasoning_lower for marker in conclusion_markers):
        quality += 0.15
    
    # LENGTH & COMPLETENESS: Reward substantive responses (0.1 max)
    words = reasoning.split()
    if len(words) > 200:  # Reasonably detailed response
        quality += 0.1
    elif len(words) > 100:  # Moderate response
        quality += 0.07
    elif len(words) > 50:  # Brief response
        quality += 0.03
    
    # BONUS: Special detection for "PREMISES/REASONING STEPS/CONCLUSION" format (extra 0.1)
    # This is the ideal format we're trying to encourage
    if "premise" in reasoning_lower and "step" in reasoning_lower and "conclusion" in reasoning_lower:
        quality += 0.1
    
    # Avoid exceeding 1.0
    return min(1.0, quality)

################################################################################
# Confidence-Weighted & Majority Voting
################################################################################

def majority_vote(agent_results: List[Dict[str, Any]]) -> Optional[str]:
    """
    Return the most common answer (majority vote), excluding error responses.
    
    This function filters out responses that appear to be error messages before
    determining the majority answer.
    
    Args:
        agent_results: List of agent response dictionaries
        
    Returns:
        The most common valid answer, or None if none
    """
    # Filter out error responses
    valid_results = [r for r in agent_results if not is_error_response(r)]
    
    # If no valid results, try with all results as fallback
    if not valid_results and agent_results:
        log_message = "Warning: All responses appear to be errors, using all responses for majority vote"
        try:
            import logging
            logging.warning(log_message)
        except ImportError:
            print(log_message)
        valid_results = agent_results
    
    answers = [r.get("answer", "") for r in valid_results]
    if not answers:
        return None
    return max(set(answers), key=answers.count)

def confidence_weighted_vote(agent_results: List[Dict[str, Any]], 
                              reasoning_weight: float = 0.4) -> Optional[str]:
    """
    Return the answer with max summed confidence weight, incorporating reasoning quality.
    Prioritizes valid responses over error messages.
    
    Args:
        agent_results: List of agent response dictionaries
        reasoning_weight: Weight to give reasoning quality (0.0-1.0) vs. model confidence
                          Default increased to 0.4 to give more weight to reasoning quality
        
    Returns:
        The answer with the highest weighted score, or None if no answers
    """
    # Filter out error responses
    valid_results = [r for r in agent_results if not is_error_response(r)]
    
    # If no valid results, try with all results as fallback
    if not valid_results and agent_results:
        log_message = "Warning: All responses appear to be errors, using all responses for confidence weighted vote"
        try:
            import logging
            logging.warning(log_message)
        except ImportError:
            print(log_message)
        valid_results = agent_results
    
    weights = {}
    for r in valid_results:
        a = r.get("answer", "")
        confidence = float(r.get("confidence", 0.0))
        reasoning = r.get("reasoning", "")
        
        # If this is an error response (in fallback case), drastically reduce its weight
        if is_error_response(r):
            confidence *= 0.1  # Reduce confidence of error responses
        
        # Assess reasoning quality and blend with confidence
        reasoning_quality = assess_reasoning_quality(reasoning)
        adjusted_confidence = confidence * (1 - reasoning_weight) + reasoning_quality * reasoning_weight
        
        # Apply transformation to the adjusted confidence
        c = trans_confidence(adjusted_confidence)
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
        # Optionally could aggregate time info here by answer, but not needed for UI, so skip.
        exps.setdefault(ans, []).append(exp)
    return exps

################################################################################
# Debate Prompt Construction
################################################################################

def build_debate_prompt(agent_results: List[Dict[str, Any]]) -> str:
    """
    Synthesize a prompt aggregating model disagreements and rationales.
    Enhanced to request structured, step-by-step reasoning in responses.
    """
    answers = [r["answer"] for r in agent_results]
    expl_by_ans = aggregate_explanations(agent_results)
    ctr = Counter(answers).most_common(2)
    
    s = "Please think step-by-step and provide structured reasoning with:\n"
    s += "1. Key premises and assumptions\n"
    s += "2. Logical reasoning steps\n"
    s += "3. Your final conclusion\n\n"
    s += "Consider these different viewpoints from other agents:\n\n"
    
    for ans, count in ctr:
        s += f"There are {count} agents who think the answer is '{ans}'.\n"
        reasonings = expl_by_ans.get(ans, [])
        if reasonings:
            s += "\n".join([f"One agent says: {e}" for e in reasonings])
        s += "\n\n"
    
    s += "Based on the information above, reconsider the question using structured reasoning."
    s += "\nMake sure to explicitly explain your reasoning process and how you arrived at your conclusion."
    # Optional: Could include timing summary in debate prompt if desired
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
                *[b.complete(context_prompts[i]) for i, b in enumerate(self.backends)],
                return_exceptions=True
            )
            
            # Normalize responses, converting exceptions to error responses
            results = []
            for i, out in enumerate(outs):
                if isinstance(out, Exception):
                    # Convert exception to error response
                    error_msg = f"[Error from {self.backends[i].name}]: {str(out)}"
                    results.append({
                        "answer": error_msg,
                        "reasoning": error_msg,
                        "confidence": 0.0,
                        "time": 0.0,
                        "is_error": True
                    })
                else:
                    # Normalize regular response
                    result = normalize_agent_result(out)
                    # Mark as error if it matches error patterns
                    if is_error_response(result):
                        result["is_error"] = True
                    results.append(result)
            
            history.append(results)
            
            # Only build new debate prompt for additional rounds
            if r < self.rounds - 1:
                # Filter out error responses for debate prompts
                valid_results = [r for r in results if not is_error_response(r)]
                
                # If we have valid results, use only those for the debate prompt
                # Otherwise, use all results (including errors)
                debate_inputs = valid_results if valid_results else results
                debate_text = build_debate_prompt(debate_inputs)
                
                # Next round: each model gets this joint prompt + original input
                context_prompts = [
                    f"{prompt}\n\n{debate_text}" for _ in self.backends
                ]
        
        # Aggregate final round
        final = history[-1]
        
        # Use improved voting functions
        maj_ans = majority_vote(final)
        weighted_ans = confidence_weighted_vote(final)
        
        # Get explanations preferring non-error responses
        valid_final = [r for r in final if not is_error_response(r)]
        explanations = [r.get("reasoning", "") for r in (valid_final if valid_final else final)]
        
        # Count error vs non-error responses for diagnostics
        error_count = sum(1 for r in final if is_error_response(r))
        total_count = len(final)
        
        return {
            "rounds": history,
            "winner_majority": maj_ans,
            "winner_weighted": weighted_ans,
            "explanations": explanations,
            "agent_outputs": final,
            "error_count": error_count,
            "total_count": total_count
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

