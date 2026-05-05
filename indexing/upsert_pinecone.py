"""Build the Pinecone index for the job-skill-set dataset.

Usage:
    python -m indexing.upsert_pinecone

Required environment variables:
    OPENAI_API_KEY
    PINECONE_API_KEY
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

from dotenv import load_dotenv
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone

from data.load_dataset import JobPosting, load_job_postings
from indexing.embed import DEFAULT_EMBEDDING_DIMENSION, OpenAIEmbedder


def chunked(items: list[JobPosting], size: int) -> Iterable[list[JobPosting]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_pinecone_client() -> Pinecone:
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is required. Copy .env.example to .env first.")
    return Pinecone(api_key=api_key)


def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """Create the serverless Pinecone index if it does not already exist."""
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
            deletion_protection="disabled",
        )

    while True:
        description = pc.describe_index(index_name)
        status = getattr(description, "status", {}) or {}
        ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            return
        print("Waiting for Pinecone index to become ready...")
        time.sleep(5)


def upsert_jobs(
    postings: list[JobPosting],
    *,
    index_name: str,
    namespace: str,
    batch_size: int,
) -> None:
    pc = get_pinecone_client()
    ensure_index(
        pc,
        index_name=index_name,
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
    )
    index = pc.Index(index_name)
    embedder = OpenAIEmbedder()

    total = 0
    for batch in chunked(postings, batch_size):
        vectors = embedder.embed_texts([posting.embedding_text() for posting in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} embeddings, received {len(vectors)}.")
        records = []
        for posting, vector in zip(batch, vectors):
            records.append(
                {
                    "id": posting.job_id,
                    "values": vector,
                    "metadata": posting.pinecone_metadata(),
                }
            )
        index.upsert(vectors=records, namespace=namespace)
        total += len(records)
        print(f"Upserted {total}/{len(postings)} jobs")

    stats = index.describe_index_stats()
    print(stats)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Embed jobs and upsert them into Pinecone.")
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N rows for testing.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-name", default=os.getenv("PINECONE_INDEX_NAME", "job-skill-rag"))
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "job-skill-set"))
    args = parser.parse_args()

    postings = load_job_postings(limit=args.limit)
    upsert_jobs(
        postings,
        index_name=args.index_name,
        namespace=args.namespace,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
