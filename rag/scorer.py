"""Deterministic hybrid match scoring.

The LLM explains matches, but Python owns the numeric score:

    50% semantic similarity + 30% skill overlap + 20% category agreement
"""

from __future__ import annotations

import re

from rag.models import RetrievedJob, ScoreBreakdown, ScoredJob


SEMANTIC_FLOOR = 0.25
SEMANTIC_CEILING = 0.85


def normalise_skill(skill: str) -> str:
    """Lowercase and lightly canonicalise a skill phrase."""
    text = skill.casefold()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^\w+#. -]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,-")


def normalise_skill_set(skills: list[str]) -> set[str]:
    return {normalised for skill in skills if (normalised := normalise_skill(skill))}


def semantic_percentage(cosine_sim: float) -> float:
    scaled = (cosine_sim - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR)
    clamped = max(0.0, min(1.0, scaled))
    return round(clamped * 100, 1)


def extract_resume_skills(resume_text: str, candidate_skills: list[str]) -> set[str]:
    """Find which JD skills are explicitly mentioned in the resume text."""
    normalised_resume = normalise_skill(resume_text)
    found: set[str] = set()
    for skill in normalise_skill_set(candidate_skills):
        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
        if re.search(pattern, normalised_resume):
            found.add(skill)
    return found


def score_job(
    *,
    resume_text: str,
    job: RetrievedJob,
    predicted_resume_category: str | None = None,
) -> ScoreBreakdown:
    jd_skills = normalise_skill_set(job.skills)
    resume_skills = extract_resume_skills(resume_text, job.skills)
    matched_skills = sorted(jd_skills & resume_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    if jd_skills:
        skill_overlap_pct = round(len(matched_skills) / len(jd_skills) * 100, 1)
    else:
        skill_overlap_pct = 0.0

    category_bonus = 0.0
    if predicted_resume_category and predicted_resume_category.casefold() == job.category.casefold():
        category_bonus = 100.0

    semantic_pct = semantic_percentage(job.cosine_sim)
    match_score = round(
        0.5 * semantic_pct + 0.3 * skill_overlap_pct + 0.2 * category_bonus,
        1,
    )

    return ScoreBreakdown(
        match_score=match_score,
        semantic_pct=semantic_pct,
        skill_overlap_pct=skill_overlap_pct,
        category_bonus=category_bonus,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        matched_skill_count=len(matched_skills),
        total_skill_count=len(jd_skills),
    )


def score_jobs(
    *,
    resume_text: str,
    jobs: list[RetrievedJob],
    predicted_resume_category: str | None = None,
) -> list[ScoredJob]:
    scored = [
        ScoredJob(
            job=job,
            score=score_job(
                resume_text=resume_text,
                job=job,
                predicted_resume_category=predicted_resume_category,
            ),
        )
        for job in jobs
    ]
    return sorted(scored, key=lambda item: item.score.match_score, reverse=True)
