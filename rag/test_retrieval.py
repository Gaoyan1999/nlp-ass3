"""Run a simple retrieval test against Pinecone.

This script is for learning Step 4 only:
resume text -> embedding -> Pinecone query -> retrieved jobs.
"""

from __future__ import annotations

import argparse

from rag.retriever import PineconeJobRetriever


DEFAULT_RESUME = """
Data analyst with experience in Python, SQL, pandas, dashboard reporting,
machine learning experiments, ETL scripts, cloud data workflows, and stakeholder communication.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Test resume-to-job retrieval.")
    parser.add_argument("--resume", default=DEFAULT_RESUME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    retriever = PineconeJobRetriever()
    jobs = retriever.retrieve(
        args.resume,
        top_k=args.top_k,
        category_filter=args.category,
    )

    print(f"Retrieved {len(jobs)} jobs")
    print()

    for index, job in enumerate(jobs, start=1):
        print(f"#{index} {job.job_title}")
        print(f"   job_id: {job.job_id}")
        print(f"   category: {job.category}")
        print(f"   cosine similarity: {job.cosine_sim:.4f}")
        print(f"   skills: {', '.join(job.skills[:8])}")
        print()


if __name__ == "__main__":
    main()
