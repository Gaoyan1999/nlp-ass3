from rag.models import RetrievedJob
from rag.scorer import score_job, semantic_percentage


def test_semantic_percentage_rescales_and_clamps() -> None:
    assert semantic_percentage(0.25) == 0.0
    assert semantic_percentage(0.85) == 100.0
    assert semantic_percentage(0.55) == 50.0
    assert semantic_percentage(0.95) == 100.0
    assert semantic_percentage(0.10) == 0.0


def test_score_job_combines_semantic_skills_and_category() -> None:
    job = RetrievedJob(
        job_id="1",
        category="IT",
        job_title="Data Engineer",
        job_description="Build pipelines and dashboards.",
        skills=["Python", "SQL", "Data pipelines", "AWS"],
        cosine_sim=0.73,
    )

    score = score_job(
        resume_text="I use Python and SQL to build data pipelines.",
        job=job,
        predicted_resume_category="IT",
    )

    assert score.semantic_pct == 80.0
    assert score.skill_overlap_pct == 75.0
    assert score.category_bonus == 100.0
    assert score.match_score == 82.5
    assert score.matched_skill_count == 3
    assert "aws" in score.missing_skills


def test_score_job_has_no_category_bonus_when_classifier_unavailable() -> None:
    job = RetrievedJob(
        job_id="2",
        category="Finance",
        job_title="Analyst",
        job_description="Analyse financial performance.",
        skills=["Excel"],
        cosine_sim=0.55,
    )

    score = score_job(resume_text="Excel reporting", job=job, predicted_resume_category=None)

    assert score.category_bonus == 0.0
    assert score.match_score == 55.0
