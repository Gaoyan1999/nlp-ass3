"""FastAPI entry point for the Resume -> Job RAG Matcher."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.category import VALID_CATEGORIES, resolve_resume_category
from rag.eval_store import append_eval_record, get_eval_record, list_eval_records, save_evaluation
from rag.evaluator import GenerationEvaluator
from rag.generator import CareerCoachGenerator, build_fallback_generation
from rag.models import RagGeneration, ScoredJob
from rag.pdf_resume import pdf_bytes_to_markdown
from rag.retriever import PineconeJobRetriever
from rag.scorer import score_jobs


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = ROOT_DIR / "frontend" / "index.html"
FRONTEND_EVAL = ROOT_DIR / "frontend" / "eval.html"

load_dotenv()
app = FastAPI(title="Resume Job RAG Matcher", version="0.1.0")


Category = Literal["HR", "INFORMATION-TECHNOLOGY", "BUSINESS-DEVELOPMENT", "FINANCE", "SALES"]


class MatchRequest(BaseModel):
    resume_text: str = Field(min_length=30)
    top_k: int = Field(default=10, ge=1, le=20)
    category_filter: Optional[Category] = None
    predicted_category: Optional[Category] = None
    use_llm: bool = True
    llm_model: Optional[str] = Field(default=None, max_length=80)


class MatchResponse(BaseModel):
    predicted_category: Optional[str]
    category_source: str
    retrieval_category_filter: Optional[str]
    generation_source: str
    generation_model: Optional[str] = None
    resume_markdown: Optional[str] = None
    matches: list[ScoredJob]
    generation: RagGeneration


class EvalRequest(BaseModel):
    evaluator_model: Optional[str] = Field(default=None, max_length=80)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)


@app.get("/eval")
def eval_page() -> FileResponse:
    return FileResponse(FRONTEND_EVAL)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/categories")
def categories() -> dict[str, list[str]]:
    return {"categories": sorted(VALID_CATEGORIES)}


@app.get("/eval-records")
def eval_records() -> dict[str, object]:
    return {"records": [record.model_dump() for record in list_eval_records()]}


@app.get("/eval-records/{record_id}")
def eval_record(record_id: str) -> dict[str, object]:
    try:
        return get_eval_record(record_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Eval record not found.") from exc


@app.post("/eval-records/{record_id}/evaluate")
def evaluate_record(record_id: str, request: Optional[EvalRequest] = None) -> dict[str, object]:
    try:
        record = get_eval_record(record_id)
        evaluation = GenerationEvaluator(model=request.evaluator_model if request else None).evaluate(record)
        return save_evaluation(record_id=record_id, evaluation=evaluation).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Eval record not found.") from exc
    except Exception as exc:
        raise_api_error(exc)


@app.post("/match", response_model=MatchResponse)
def match(request: MatchRequest) -> MatchResponse:
    try:
        return run_match_pipeline(
            resume_text=request.resume_text,
            top_k=request.top_k,
            category_filter=request.category_filter,
            predicted_category=request.predicted_category,
            use_llm=request.use_llm,
            llm_model=request.llm_model,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise_api_error(exc)


@app.post("/match-pdf", response_model=MatchResponse)
async def match_pdf(
    resume_pdf: UploadFile = File(...),
    top_k: int = Form(default=10),
    category_filter: Optional[str] = Form(default=None),
    predicted_category: Optional[str] = Form(default=None),
    use_llm: bool = Form(default=True),
    llm_model: Optional[str] = Form(default=None),
) -> MatchResponse:
    try:
        resume_markdown = pdf_bytes_to_markdown(
            await resume_pdf.read(),
            filename=resume_pdf.filename or "resume.pdf",
        )
        return run_match_pipeline(
            resume_text=resume_markdown,
            top_k=max(1, min(20, top_k)),
            category_filter=category_filter or None,
            predicted_category=predicted_category or None,
            use_llm=use_llm,
            llm_model=llm_model or None,
            resume_markdown=resume_markdown,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise_api_error(exc)


def run_match_pipeline(
    *,
    resume_text: str,
    top_k: int,
    category_filter: Optional[str],
    predicted_category: Optional[str],
    use_llm: bool,
    llm_model: Optional[str],
    resume_markdown: Optional[str] = None,
) -> MatchResponse:
    predicted_category_value, category_source = resolve_resume_category(
        resume_text=resume_text,
        supplied_category=predicted_category,
    )

    retriever = PineconeJobRetriever()
    retrieved = retriever.retrieve(
        resume_text,
        top_k=top_k,
        category_filter=category_filter,
    )
    scored = score_jobs(
        resume_text=resume_text,
        jobs=retrieved,
        predicted_resume_category=predicted_category_value,
    )

    if use_llm:
        try:
            generator = CareerCoachGenerator(model=llm_model)
            generation = generator.generate(
                resume_text=resume_text,
                scored_jobs=scored,
            )
            generation_source = "llm"
            generation_model = generator.model
        except Exception as exc:
            generation = build_fallback_generation(scored)
            generation.notes = (
                "LLM generation failed, so deterministic skill gaps are shown. "
                f"Error type: {exc.__class__.__name__}."
            )
            generation_source = "fallback"
            generation_model = llm_model
    else:
        generation = build_fallback_generation(scored)
        generation.notes = "LLM generation disabled; deterministic skill gaps are shown."
        generation_source = "disabled"
        generation_model = None

    if generation_source == "llm":
        try:
            append_eval_record(
                resume_text=resume_text,
                predicted_category=predicted_category_value,
                category_source=category_source,
                retrieval_category_filter=category_filter,
                generation_model=generation_model,
                matches=scored,
                generation=generation,
            )
        except Exception:
            pass

    return MatchResponse(
        predicted_category=predicted_category_value,
        category_source=category_source,
        retrieval_category_filter=category_filter,
        generation_source=generation_source,
        generation_model=generation_model,
        resume_markdown=resume_markdown,
        matches=scored,
        generation=generation,
    )


def raise_api_error(exc: Exception) -> None:
    if "PINECONE_API_KEY is required" in str(exc) or "OPENAI_API_KEY" in str(exc):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
