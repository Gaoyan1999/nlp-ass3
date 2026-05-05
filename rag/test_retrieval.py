"""Run a simple retrieval test against Pinecone.

This script is for learning Step 4 only:
resume text -> embedding -> Pinecone query -> retrieved jobs.
"""

from __future__ import annotations

import argparse

from rag.retriever import PineconeJobRetriever


DEFAULT_RESUME = """
COMPANY WWW.3GHCRE.COM ROLE DESCRIPTION THIS IS A FULL-TIME SALES ASSOCIATE ROLE AT 3G HEALTHCARE REAL ESTATE. AS A SALES ASSOCIATE, YOU WILL BE RESPONSIBLE FOR PROSPECTING AND GENERATING NEW LEADS, CONDUCTING SALES PRESENTATIONS AND NEGOTIATIONS, AND BUILDING AND MAINTAINING RELATIONSHIPS WITH CLIENTS. THIS IS A HYBRID ROLE, LOCATED IN INDIANAPOLIS, IN, WITH THE FLEXIBILITY FOR SOME REMOTE WORK. QUALIFICATIONS STRONG INTERPERSONAL AND COMMUNICATION SKILLSEXCELLENT SALES AND NEGOTIATION SKILLSPROVEN TRACK RECORD IN MEETING AND EXCEEDING SALES TARGETSEXPERIENCE IN THE HEALTHCARE OR REAL ESTATE INDUSTRY IS A PLUSKNOWLEDGE OF SALES TECHNIQUES AND STRATEGIESABILITY TO WORK INDEPENDENTLY AND AS PART OF A TEAMPROFICIENCY IN MICROSOFT OFFICE AND CRM SOFTWAREBACHELOR'S DEGREE IN BUSINESS ADMINISTRATION, MARKETING, OR RELATED FIELD
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Test resume-to-job retrieval.")
    parser.add_argument("--resume", default=DEFAULT_RESUME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    retriever = PineconeJobRetriever()
    jobs = retriever.retrieve(args.resume, top_k=args.top_k, category_filter=args.category)

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
