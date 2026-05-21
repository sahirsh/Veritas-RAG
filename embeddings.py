from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
client: genai.Client | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    """Aligned `texts` and `vectors` (only non-empty stripped strings are embedded)."""

    model: str
    texts: list[str]
    vectors: list[list[float]]


class EmbeddingError(RuntimeError):
    pass


def default_embedding_model() -> str:
    return os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def _gemini_error_message(exc: BaseException | None) -> str:
    if exc is None:
        return "The embedding service is temporarily unavailable. Please try again later."

    text = str(exc).lower()
    if "api_key" in text or "api key" in text or "permission" in text:
        return (
            "Gemini embedding service authentication failed. "
            "Verify GEMINI_API_KEY if you operate this API."
        )
    if "quota" in text or "rate limit" in text or "resource_exhausted" in text:
        return "The Gemini embedding service rate limit was exceeded. Please try again shortly."

    return "The embedding service is temporarily unavailable. Please try again later."


def _client() -> genai.Client:
    global client
    if client is None:
        client = genai.Client()
    return client


def _embedding_values(item: Any) -> list[float]:
    values = getattr(item, "values", None)
    if values is None and isinstance(item, dict):
        values = item.get("values")
    if isinstance(values, list):
        return [float(v) for v in values]

    embedding = getattr(item, "embedding", None)
    if embedding is None and isinstance(item, dict):
        embedding = item.get("embedding")
    if embedding is not None:
        return _embedding_values(embedding)

    raise EmbeddingError("The embedding service returned an unexpected response.")


def _embedding_batch_values(response: Any) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if embeddings is not None:
        return [_embedding_values(item) for item in embeddings]

    return [_embedding_values(response)]


async def embed_texts_gemini(
    texts: list[str],
    *,
    model: str | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 128,
    max_retries: int = 5,
) -> EmbeddingResult:
    """
    Create embeddings for a list of texts using Google Gemini's Embeddings API.

    Requires env var GEMINI_API_KEY.
    Model defaults to GEMINI_EMBEDDING_MODEL or gemini-embedding-001.
    """
    resolved_model = model or default_embedding_model()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EmbeddingError("Gemini embedding service is not configured on this server.")

    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return EmbeddingResult(model=resolved_model, texts=[], vectors=[])

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    vectors: list[list[float]] = []

    async def _call_with_retries(batch: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                batch_vectors = await asyncio.to_thread(
                    _embed_batch_sync,
                    batch,
                    resolved_model,
                    task_type,
                )
                if not batch_vectors:
                    raise EmbeddingError("The embedding service returned an empty response.")
                return batch_vectors
            except EmbeddingError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                await asyncio.sleep(min(2**attempt, 20))
        if last_exc is not None:
            logger.error(
                "Gemini embedding request failed after %s attempts: %s",
                max_retries + 1,
                last_exc,
                exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
            )
        msg = _gemini_error_message(last_exc)
        raise EmbeddingError(msg) from last_exc

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        vectors.extend(await _call_with_retries(batch))

    return EmbeddingResult(model=resolved_model, texts=cleaned, vectors=vectors)


def _embed_batch_sync(batch: list[str], model: str, task_type: str) -> list[list[float]]:
    response = _client().models.embed_content(
        model=model,
        contents=batch,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )
    vectors = _embedding_batch_values(response)
    if len(vectors) != len(batch):
        raise EmbeddingError("The embedding service returned a mismatched batch size.")
    return vectors



