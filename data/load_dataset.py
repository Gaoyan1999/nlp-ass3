"""Load the Hugging Face job-skill-set dataset."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "batuhanmtl/job-skill-set"
DEFAULT_OUTPUT_PATH = Path("data/job_postings.json")


@dataclass(frozen=True)
class JobPosting:
    job_id: str
    category: str
    job_title: str
    job_description: str
    job_skill_set: list[str]

    def embedding_text(self) -> str:
        """Text embedded for semantic retrieval."""
        skills = ", ".join(self.job_skill_set)
        return f"{self.job_title}\n{self.job_description}\nSkills: {skills}"

    def pinecone_metadata(self, description_chars: int = 6000) -> dict[str, object]:
        """Metadata stored beside the vector for filtering and RAG prompts."""
        return {
            "job_id": self.job_id,
            "category": self.category,
            "job_title": self.job_title,
            "job_description": self.job_description[:description_chars],
            "skills": self.job_skill_set,
        }


def parse_skill_set(value: object) -> list[str]:
    """Parse the dataset's stringified list of skills."""
    skills = ast.literal_eval(str(value))
    if not isinstance(skills, list):
        raise ValueError(f"Expected job_skill_set to be a list, got {type(skills).__name__}")
    return [str(skill).strip() for skill in skills if str(skill).strip()]


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def load_job_postings(limit: int | None = None) -> list[JobPosting]:
    """Load the train split from Hugging Face and return clean job postings."""
    dataset = load_dataset(DATASET_NAME, split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    postings: list[JobPosting] = []
    for row in dataset:
        postings.append(
            JobPosting(
                job_id=str(row["job_id"]),
                category=_clean_text(row["category"]),
                job_title=_clean_text(row["job_title"]),
                job_description=_clean_text(row["job_description"]),
                job_skill_set=parse_skill_set(row["job_skill_set"]),
            )
        )
    return postings


def write_json(postings: list[JobPosting], path: Path) -> None:
    """Write cleaned jobs to one JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(posting) for posting in postings], handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load all job postings and save them as JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    postings = load_job_postings()
    write_json(postings, args.output)
    print(f"Loaded {len(postings)} postings from {DATASET_NAME}")
    print(f"Wrote cleaned postings to {args.output}")


if __name__ == "__main__":
    main()
