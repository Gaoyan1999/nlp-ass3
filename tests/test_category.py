from rag.category import normalise_category, resolve_resume_category
from rag.models import RetrievedJob
from rag.scorer import score_job


def test_normalise_saved_model_category_label() -> None:
    assert normalise_category("information-technology") == "INFORMATION-TECHNOLOGY"
    assert normalise_category("business development") == "BUSINESS-DEVELOPMENT"


def test_supplied_category_is_normalised_without_loading_classifier() -> None:
    category, source = resolve_resume_category(
        resume_text="Short resume text",
        supplied_category="it",
    )

    assert category == "INFORMATION-TECHNOLOGY"
    assert source == "request"


def test_category_bonus_accepts_dataset_aliases() -> None:
    job = RetrievedJob(
        job_id="1",
        category="IT",
        job_title="Developer",
        job_description="Build software.",
        skills=[],
        cosine_sim=0.55,
    )

    score = score_job(
        resume_text="Software developer",
        job=job,
        predicted_resume_category="INFORMATION-TECHNOLOGY",
    )

    assert score.category_bonus == 100.0
