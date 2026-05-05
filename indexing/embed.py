"""OpenAI embedding wrapper used by both indexing and retrieval."""

from __future__ import annotations

import os
from collections.abc import Iterable

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class OpenAIEmbedder:
    """Small wrapper around OpenAI embeddings.

    `text-embedding-3-small` returns 1536-dimensional vectors by default.
    """

    def __init__(self, model: str | None = None, batch_size: int = 64) -> None:
        load_dotenv()
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.batch_size = batch_size
        self.client = OpenAI()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in batched([self._truncate(text) for text in texts], self.batch_size):
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    @staticmethod
    def _truncate(text: str, max_chars: int = 24000) -> str:
        """Keep requests below the embedding model's token limit without tiktoken."""
        return text[:max_chars]
