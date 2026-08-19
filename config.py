"""Runtime configuration read from environment variables."""

from __future__ import annotations

import os

APP_VERSION = "1.0.0"

# How long uploaded documents live before being eligible for purge.
DOCUMENT_TTL_HOURS: int = int(os.environ.get("DOCUMENT_TTL_HOURS", "48"))


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean from an environment variable.

    Accepts common truthy spellings ("1", "true", "yes", "on"), case-insensitive.
    Anything else (or an unset variable) falls back to `default`.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# When True, route chat generation to a local LM Studio server instead of Groq.
USE_LOCAL_MODELS: bool = _env_bool("USE_LOCAL_MODELS", False)
# OpenAI-compatible endpoint exposed by LM Studio (only used when USE_LOCAL_MODELS).
LOCAL_LLM_BASE_URL: str = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
# Model identifier to request from the local server.
LOCAL_LLM_MODEL: str = os.environ.get("LOCAL_LLM_MODEL", "local-model")

# The same toggle also switches embeddings from Gemini to a local
# sentence-transformers model. bge-small-en-v1.5 outputs 384 dims vs Gemini's 768.
LOCAL_EMBEDDING_MODEL: str = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LOCAL_EMBEDDING_DIM: int = int(os.environ.get("LOCAL_EMBEDDING_DIM", "384"))

# Vector dimension of the pgvector column. It MUST match the active embedding
# model, and query/document vectors must come from the same model, so a database
# is single-model: switching this requires wiping and re-embedding that database.
EMBEDDING_DIM: int = LOCAL_EMBEDDING_DIM if USE_LOCAL_MODELS else 768


def database_configured() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def gemini_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def groq_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [part.strip() for part in raw.split(",") if part.strip()]
