"""LLM-as-judge evaluation for generated match explanations."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from rag.eval_store import EvalRecord


EVALUATOR_SYSTEM_PROMPT = """You are an independent evaluator for a resume-to-job RAG system.
Judge only the generation quality. Do not judge whether the category classifier is correct.

Use only the supplied resume text, retrieved job evidence, backend score breakdowns, and generated output.
Return strict JSON with:
- recommendation: "pass", "review", or "fail"
- groundedness: {score: 1-5, reason: string}
- helpfulness: {score: 1-5, reason: string}
- hallucination_control: {score: 1-5, reason: string, unsupported_claims: list[string]}
- actionability: {score: 1-5, reason: string}
- evidence_use: {score: 1-5, reason: string}
- summary: short string

Scoring rubric:
5 = excellent, 4 = good/minor issues, 3 = usable but notable issues, 2 = weak, 1 = poor.
Do not return an overall score. The application computes the total score by summing the five criterion scores, with a maximum of 25.
Groundedness means every explanation is supported by the resume, job skills/descriptions, or backend score data.
Hallucination control is high only when there are no unsupported skills, credentials, job facts, or numeric claims.
Actionability means the skill-gap suggestions are specific and tied to missing skills from retrieved jobs.
The LLM should not invent numeric scores or contradict the backend ranking."""


class GenerationEvaluator:
    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini")
        self.client = OpenAI()

    def evaluate(self, record: EvalRecord) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": build_eval_prompt(record)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        evaluation = coerce_evaluation_json(content)
        evaluation["evaluator_model"] = self.model
        return evaluation


def build_eval_prompt(record: EvalRecord) -> str:
    return json.dumps(
        {
            "record_id": record.record_id,
            "resume_text": record.resume_text[:5000],
            "predicted_category": record.predicted_category,
            "category_source": record.category_source,
            "retrieved_jobs": [_compact_match(item) for item in record.matches[:8]],
            "generated_output": record.generation,
            "task": "Evaluate whether the generated output is grounded, helpful, non-hallucinated, actionable, and faithful to backend evidence.",
        },
        ensure_ascii=False,
    )


def coerce_evaluation_json(content: str) -> dict[str, Any]:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    return {
        "groundedness": _criterion(raw.get("groundedness")),
        "helpfulness": _criterion(raw.get("helpfulness")),
        "hallucination_control": _criterion(raw.get("hallucination_control"), include_claims=True),
        "actionability": _criterion(raw.get("actionability")),
        "evidence_use": _criterion(raw.get("evidence_use")),
        "recommendation": _recommendation(raw.get("recommendation")),
        "summary": str(raw.get("summary") or "No evaluator summary returned.").strip(),
    }


def _compact_match(item: dict[str, Any]) -> dict[str, Any]:
    job = item.get("job", {}) if isinstance(item.get("job"), dict) else {}
    score = item.get("score", {}) if isinstance(item.get("score"), dict) else {}
    return {
        "job_id": job.get("job_id"),
        "job_title": job.get("job_title"),
        "category": job.get("category"),
        "job_description_excerpt": str(job.get("job_description", ""))[:1200],
        "job_skills": job.get("skills", [])[:30],
        "backend_score": {
            "match_score": score.get("match_score"),
            "semantic_pct": score.get("semantic_pct"),
            "skill_overlap_pct": score.get("skill_overlap_pct"),
            "category_bonus": score.get("category_bonus"),
            "matched_skills": score.get("matched_skills", [])[:20],
            "missing_skills": score.get("missing_skills", [])[:20],
        },
    }


def _criterion(value: Any, *, include_claims: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    result: dict[str, Any] = {
        "score": _score(value.get("score")),
        "reason": str(value.get("reason") or "No reason returned.").strip(),
    }
    if include_claims:
        claims = value.get("unsupported_claims", [])
        result["unsupported_claims"] = [str(item) for item in claims] if isinstance(claims, list) else []
    return result


def _score(value: Any) -> int:
    try:
        return max(1, min(5, round(float(value))))
    except (TypeError, ValueError):
        return 1


def _recommendation(value: Any) -> str:
    text = str(value or "review").strip().lower()
    return text if text in {"pass", "review", "fail"} else "review"
