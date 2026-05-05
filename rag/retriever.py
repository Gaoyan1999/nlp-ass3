"""Pinecone retrieval for resume-to-job matching."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

from indexing.embed import OpenAIEmbedder
from rag.models import RetrievedJob


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class PineconeJobRetriever:
    def __init__(
        self,
        *,
        index_name: str | None = None,
        namespace: str | None = None,
        embedder: OpenAIEmbedder | None = None,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is required for retrieval. Copy .env.example to .env, "
                "add your Pinecone key, then run `python3 -m indexing.upsert_pinecone` "
                "before calling /match."
            )
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "job-skill-rag")
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE", "job-skill-set")
        self.embedder = embedder or OpenAIEmbedder()
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(self.index_name)

    def retrieve(
        self,
        resume_text: str,
        *,
        top_k: int = 10,
        category_filter: str | None = None,
    ) -> list[RetrievedJob]:
        vector = self.embedder.embed_text(resume_text)
        metadata_filter = {"category": {"$eq": category_filter}} if category_filter else None
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=metadata_filter,
        )

        jobs: list[RetrievedJob] = []
        for match in _value(response, "matches", []):
            metadata = _value(match, "metadata", {}) or {}
            skills = metadata.get("skills", [])
            if isinstance(skills, str):
                skills = [skills]
            jobs.append(
                RetrievedJob(
                    job_id=str(metadata.get("job_id") or _value(match, "id", "")),
                    category=str(metadata.get("category", "")),
                    job_title=str(metadata.get("job_title", "")),
                    job_description=str(metadata.get("job_description", "")),
                    skills=[str(skill) for skill in skills],
                    cosine_sim=float(_value(match, "score", 0.0) or 0.0),
                )
            )
        return jobs
