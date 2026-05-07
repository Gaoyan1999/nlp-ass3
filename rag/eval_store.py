"""Local JSONL storage for LLM generation eval records."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from rag.models import RagGeneration, ScoredJob


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS_PATH = ROOT_DIR / "data" / "llm_eval_records.jsonl"


class EvalRecordSummary(BaseModel):
    record_id: str
    created_at: str
    predicted_category: Optional[str] = None
    category_source: str
    generation_model: Optional[str] = None
    match_count: int
    evaluated: bool = False
    criterion_scores: dict[str, int] = Field(default_factory=dict)
    recommendation: Optional[str] = None
    evaluator_model: Optional[str] = None


class EvalRecord(BaseModel):
    record_id: str
    created_at: str
    resume_text: str
    predicted_category: Optional[str] = None
    category_source: str
    retrieval_category_filter: Optional[str] = None
    generation_model: Optional[str] = None
    matches: list[dict[str, Any]]
    generation: dict[str, Any]
    evaluation: Optional[dict[str, Any]] = None


def append_eval_record(
    *,
    resume_text: str,
    predicted_category: str | None,
    category_source: str,
    retrieval_category_filter: str | None,
    generation_model: str | None,
    matches: list[ScoredJob],
    generation: RagGeneration,
    path: Path = DEFAULT_RECORDS_PATH,
) -> EvalRecord:
    record = EvalRecord(
        record_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc).isoformat(),
        resume_text=resume_text,
        predicted_category=predicted_category,
        category_source=category_source,
        retrieval_category_filter=retrieval_category_filter,
        generation_model=generation_model,
        matches=[_serialise_scored_job(item) for item in matches],
        generation=generation.model_dump(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
    return record


def list_eval_records(path: Path = DEFAULT_RECORDS_PATH) -> list[EvalRecordSummary]:
    return [_summarise(record) for record in _read_records(path)]


def get_eval_record(record_id: str, path: Path = DEFAULT_RECORDS_PATH) -> EvalRecord:
    for record in _read_records(path):
        if record.record_id == record_id:
            return record
    raise KeyError(record_id)


def save_evaluation(
    *,
    record_id: str,
    evaluation: dict[str, Any],
    path: Path = DEFAULT_RECORDS_PATH,
) -> EvalRecord:
    records = _read_records(path)
    updated: EvalRecord | None = None
    for index, record in enumerate(records):
        if record.record_id == record_id:
            updated = record.model_copy(update={"evaluation": evaluation})
            records[index] = updated
            break
    if updated is None:
        raise KeyError(record_id)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
    return updated


def _read_records(path: Path) -> list[EvalRecord]:
    if not path.exists():
        return []
    records: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(EvalRecord.model_validate_json(line))
    return records


def _summarise(record: EvalRecord) -> EvalRecordSummary:
    evaluation = record.evaluation or {}
    return EvalRecordSummary(
        record_id=record.record_id,
        created_at=record.created_at,
        predicted_category=record.predicted_category,
        category_source=record.category_source,
        generation_model=record.generation_model,
        match_count=len(record.matches),
        evaluated=bool(record.evaluation),
        criterion_scores=_criterion_scores(evaluation),
        recommendation=str(evaluation.get("recommendation")) if evaluation.get("recommendation") else None,
        evaluator_model=str(evaluation.get("evaluator_model")) if evaluation.get("evaluator_model") else None,
    )


def _serialise_scored_job(item: ScoredJob) -> dict[str, Any]:
    return {
        "job": item.job.model_dump(),
        "score": item.score.model_dump(),
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _criterion_scores(evaluation: dict[str, Any]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for name in ["groundedness", "helpfulness", "hallucination_control", "actionability", "evidence_use"]:
        criterion = evaluation.get(name)
        if isinstance(criterion, dict):
            score = _as_float(criterion.get("score"))
            if score is not None:
                scores[name] = int(score)
    return scores

