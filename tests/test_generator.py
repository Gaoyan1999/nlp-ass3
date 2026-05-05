from rag.generator import build_fallback_generation, parse_generation_json
from rag.models import RetrievedJob, ScoredJob
from rag.scorer import score_job


def _scored_job() -> ScoredJob:
    job = RetrievedJob(
        job_id="99",
        category="IT",
        job_title="ML Engineer",
        job_description="Build model pipelines.",
        skills=["Python", "MLOps", "Docker"],
        cosine_sim=0.67,
    )
    return ScoredJob(
        job=job,
        score=score_job(
            resume_text="Python model training",
            job=job,
            predicted_resume_category="IT",
        ),
    )


def test_fallback_generation_uses_backend_scores() -> None:
    generation = build_fallback_generation([_scored_job()])

    assert generation.ranked_matches[0].job_id == "99"
    assert "Backend score" in generation.ranked_matches[0].why_match
    assert {gap.skill for gap in generation.skill_gaps} == {"docker", "mlops"}


def test_parse_generation_json_falls_back_on_invalid_json() -> None:
    generation = parse_generation_json("not json", scored_jobs=[_scored_job()])

    assert generation.ranked_matches[0].job_title == "ML Engineer"
