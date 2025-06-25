import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from consensus_debate import majority_vote, confidence_weighted_vote, DebateManager
from consensus_backends import LLMBackend

class FixedBackend(LLMBackend):
    def __init__(self, name, answer, confidence=0.0):
        super().__init__(name)
        self._answer = answer
        self._conf = confidence

    async def complete(self, prompt: str):
        return {"answer": self._answer, "reasoning": "", "confidence": self._conf}

def test_majority_vote():
    res = [{"answer": "A"}, {"answer": "B"}, {"answer": "B"}]
    assert majority_vote(res) == "B"

def test_confidence_weighted_vote():
    res = [
        {"answer": "yes", "confidence": 0.6},
        {"answer": "no", "confidence": 0.9},
        {"answer": "yes", "confidence": 0.6},
    ]
    assert confidence_weighted_vote(res) == "no"

def test_debate_manager():
    backends = [
        FixedBackend("a", "yes", 0.6),
        FixedBackend("b", "no", 0.9),
        FixedBackend("c", "yes", 0.6),
    ]
    mgr = DebateManager(backends, rounds=1)
    out = asyncio.run(mgr.debate("Q"))
    assert out["winner_majority"] == "yes"
    assert out["winner_weighted"] == "no"
