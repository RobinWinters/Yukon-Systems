from fastapi import FastAPI
from pydantic import BaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from consensus_agent import ConsensusEngine


def make_app(engine: 'ConsensusEngine') -> FastAPI:
    app = FastAPI(title="Toros Consensus API")

    class Question(BaseModel):
        query: str
        rounds: int = 3

    @app.post("/api/ask")
    async def ask(question: Question):
        return await engine.query(question.query, use_debate=True, rounds=question.rounds)

    @app.get("/")
    async def root():
        return {"message": "Toros consensus API"}

    return app
