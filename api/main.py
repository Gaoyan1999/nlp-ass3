"""FastAPI entry point for the Resume -> Job RAG Matcher."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.category import VALID_CATEGORIES, resolve_resume_category
from rag.generator import CareerCoachGenerator, build_fallback_generation
from rag.models import RagGeneration, ScoredJob
from rag.retriever import PineconeJobRetriever
from rag.scorer import score_jobs


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = ROOT_DIR / "frontend" / "index.html"

load_dotenv()
app = FastAPI(title="Resume Job RAG Matcher", version="0.1.0")


Category = Literal["HR", "IT", "Business-Dev", "Finance", "Sales"]


class MatchRequest(BaseModel):
    resume_text: str = Field(min_length=30)
    top_k: int = Field(default=10, ge=1, le=20)
    category_filter: Optional[Category] = None
    predicted_category: Optional[Category] = None
    use_llm: bool = True


class MatchResponse(BaseModel):
    predicted_category: Optional[str]
    category_source: str
    retrieval_category_filter: Optional[str]
    matches: list[ScoredJob]
    generation: RagGeneration


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/categories")
def categories() -> dict[str, list[str]]:
    return {"categories": sorted(VALID_CATEGORIES)}


@app.post("/match", response_model=MatchResponse)
def match(request: MatchRequest) -> MatchResponse:
    try:
        predicted_category, category_source = resolve_resume_category(
            resume_text=request.resume_text,
            supplied_category=request.predicted_category,
        )

        retriever = PineconeJobRetriever()
        retrieved = retriever.retrieve(
            request.resume_text,
            top_k=request.top_k,
            category_filter=request.category_filter,
        )
        scored = score_jobs(
            resume_text=request.resume_text,
            jobs=retrieved,
            predicted_resume_category=predicted_category,
        )

        if request.use_llm:
            try:
                generation = CareerCoachGenerator().generate(
                    resume_text=request.resume_text,
                    scored_jobs=scored,
                )
            except Exception as exc:
                generation = build_fallback_generation(scored)
                generation.notes = (
                    "LLM generation failed, so deterministic skill gaps are shown. "
                    f"Error type: {exc.__class__.__name__}."
                )
        else:
            generation = build_fallback_generation(scored)
            generation.notes = "LLM generation disabled; deterministic skill gaps are shown."

        return MatchResponse(
            predicted_category=predicted_category,
            category_source=category_source,
            retrieval_category_filter=request.category_filter,
            matches=scored,
            generation=generation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        if "PINECONE_API_KEY is required" in str(exc) or "OPENAI_API_KEY" in str(exc):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
