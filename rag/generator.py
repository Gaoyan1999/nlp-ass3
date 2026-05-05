"""Grounded skill-gap generation with OpenAI."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from dotenv import load_dotenv

from rag.models import RagGeneration, RankedMatch, ScoredJob, SkillGap


SYSTEM_PROMPT = """You are a career coach. Use only the provided resume and retrieved job evidence.
Return JSON with:
- ranked_matches: list of objects with job_id, job_title, why_match, evidence, missing_skills
- skill_gaps: list of objects with skill, evidence_job_ids, suggestion
- notes: optional short caveat
Do not fabricate numeric scores. The backend has already computed the scores.
Keep evidence tied to the retrieved job descriptions or skill lists."""


class CareerCoachGenerator:
    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.client = OpenAI()

    def generate(self, *, resume_text: str, scored_jobs: list[ScoredJob]) -> RagGeneration:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(resume_text, scored_jobs)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return parse_generation_json(content, scored_jobs=scored_jobs)


def build_user_prompt(resume_text: str, scored_jobs: list[ScoredJob]) -> str:
    jobs_payload = []
    for item in scored_jobs:
        jobs_payload.append(
            {
                "job_id": item.job.job_id,
                "job_title": item.job.job_title,
                "category": item.job.category,
                "backend_match_score": item.score.match_score,
                "score_breakdown": {
                    "semantic_pct": item.score.semantic_pct,
                    "skill_overlap_pct": item.score.skill_overlap_pct,
                    "category_bonus": item.score.category_bonus,
                    "matched_skills": item.score.matched_skills,
                    "missing_skills": item.score.missing_skills[:15],
                },
                "skills": item.job.skills,
                "description_excerpt": item.job.job_description[:1600],
            }
        )
    return json.dumps(
        {
            "resume": resume_text[:6000],
            "retrieved_jobs": jobs_payload,
            "instruction": "Explain why the ranked jobs fit and identify repeated missing skills.",
        },
        ensure_ascii=False,
    )


def parse_generation_json(content: str, *, scored_jobs: list[ScoredJob]) -> RagGeneration:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return build_fallback_generation(scored_jobs)

    if not isinstance(raw, dict):
        return build_fallback_generation(scored_jobs)

    ranked_matches = [_coerce_ranked_match(item) for item in raw.get("ranked_matches", [])]
    ranked_matches = [item for item in ranked_matches if item is not None]
    skill_gaps = [_coerce_skill_gap(item) for item in raw.get("skill_gaps", [])]
    skill_gaps = [item for item in skill_gaps if item is not None]

    if not ranked_matches:
        fallback = build_fallback_generation(scored_jobs)
        ranked_matches = fallback.ranked_matches
    if not skill_gaps:
        fallback = build_fallback_generation(scored_jobs)
        skill_gaps = fallback.skill_gaps

    notes = raw.get("notes")
    return RagGeneration(
        ranked_matches=ranked_matches,
        skill_gaps=skill_gaps,
        notes=str(notes) if notes else None,
    )


def build_fallback_generation(scored_jobs: list[ScoredJob]) -> RagGeneration:
    """Deterministic backup used when model JSON is malformed."""
    ranked_matches = [
        RankedMatch(
            job_id=item.job.job_id,
            job_title=item.job.job_title,
            why_match=(
                f"Backend score {item.score.match_score}% with "
                f"{item.score.matched_skill_count}/{item.score.total_skill_count} listed skills found."
            ),
            evidence=item.score.matched_skills[:5],
            missing_skills=item.score.missing_skills[:8],
        )
        for item in scored_jobs
    ]

    missing_counter: Counter[str] = Counter()
    evidence: dict[str, list[str]] = {}
    for item in scored_jobs:
        for skill in item.score.missing_skills:
            missing_counter[skill] += 1
            evidence.setdefault(skill, []).append(item.job.job_id)

    skill_gaps = [
        SkillGap(
            skill=skill,
            evidence_job_ids=evidence[skill][:5],
            suggestion=f"Add concrete resume evidence for {skill} if you have this experience.",
        )
        for skill, _ in missing_counter.most_common(8)
    ]
    return RagGeneration(ranked_matches=ranked_matches, skill_gaps=skill_gaps)


def _coerce_ranked_match(item: Any) -> RankedMatch | None:
    if not isinstance(item, dict):
        return None
    job_id = str(item.get("job_id", "")).strip()
    job_title = str(item.get("job_title", "")).strip()
    why_match = str(item.get("why_match", "")).strip()
    if not job_id or not job_title or not why_match:
        return None
    return RankedMatch(
        job_id=job_id,
        job_title=job_title,
        why_match=why_match,
        evidence=_string_list(item.get("evidence", [])),
        missing_skills=_string_list(item.get("missing_skills", [])),
    )


def _coerce_skill_gap(item: Any) -> SkillGap | None:
    if isinstance(item, str):
        return SkillGap(skill=item, suggestion=f"Add concrete resume evidence for {item}.")
    if not isinstance(item, dict):
        return None
    skill = str(item.get("skill", "")).strip()
    if not skill:
        return None
    suggestion = str(item.get("suggestion", "")).strip() or f"Add concrete resume evidence for {skill}."
    return SkillGap(
        skill=skill,
        evidence_job_ids=_string_list(item.get("evidence_job_ids", [])),
        suggestion=suggestion,
    )


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
