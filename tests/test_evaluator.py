from pathlib import Path

from rag.eval_store import append_eval_record, get_eval_record, list_eval_records, save_evaluation
from rag.evaluator import coerce_evaluation_json
from rag.generator import build_fallback_generation
from rag.models import RetrievedJob, ScoredJob
from rag.scorer import score_job


def _scored_job() -> ScoredJob:
    job = RetrievedJob(
        job_id="7",
        category="INFORMATION-TECHNOLOGY",
        job_title="Backend Engineer",
        job_description="Build APIs and databases.",
        skills=["Python", "APIs", "SQL"],
        cosine_sim=0.61,
    )
    return ScoredJob(
        job=job,
        score=score_job(
            resume_text="Python API developer",
            job=job,
            predicted_resume_category="INFORMATION-TECHNOLOGY",
        ),
    )


def test_eval_store_round_trips_record_and_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    scored = [_scored_job()]
    generation = build_fallback_generation(scored)

    record = append_eval_record(
        resume_text="Python API developer",
        predicted_category="INFORMATION-TECHNOLOGY",
        category_source="saved_model",
        retrieval_category_filter=None,
        generation_model="gpt-4o-mini",
        matches=scored,
        generation=generation,
        path=path,
    )

    assert list_eval_records(path=path)[0].record_id == record.record_id
    assert get_eval_record(record.record_id, path=path).matches[0]["job"]["job_id"] == "7"
    assert get_eval_record(record.record_id, path=path).generation_model == "gpt-4o-mini"

    updated = save_evaluation(
        record_id=record.record_id,
        evaluation={
            "groundedness": {"score": 5},
            "helpfulness": {"score": 4},
            "hallucination_control": {"score": 5},
            "actionability": {"score": 4},
            "evidence_use": {"score": 4},
            "recommendation": "pass",
            "evaluator_model": "gpt-4o",
        },
        path=path,
    )

    summary = list_eval_records(path=path)[0]
    assert summary.evaluated is True
    assert "overall_score" not in updated.evaluation
    assert summary.criterion_scores == {
        "groundedness": 5,
        "helpfulness": 4,
        "hallucination_control": 5,
        "actionability": 4,
        "evidence_use": 4,
    }
    assert summary.generation_model == "gpt-4o-mini"
    assert summary.evaluator_model == "gpt-4o"


def test_coerce_evaluation_json_clamps_scores_and_defaults() -> None:
    evaluation = coerce_evaluation_json(
        """
        {
          "recommendation": "accept",
          "groundedness": {"score": 4.4, "reason": "Mostly grounded"},
          "helpfulness": {"score": 5, "reason": "Clear"},
          "hallucination_control": {
            "score": 0,
            "reason": "Unsupported claims exist",
            "unsupported_claims": ["Invented certification"]
          },
          "actionability": {"score": 3, "reason": "Some suggestions"},
          "evidence_use": {"score": 2, "reason": "Limited evidence"}
        }
        """
    )

    assert "overall_score" not in evaluation
    assert evaluation["recommendation"] == "review"
    assert evaluation["groundedness"]["score"] == 4
    assert evaluation["helpfulness"]["score"] == 5
    assert evaluation["hallucination_control"]["score"] == 1
    assert evaluation["hallucination_control"]["unsupported_claims"] == ["Invented certification"]
