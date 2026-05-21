"""Runtime configuration read from environment variables."""

from __future__ import annotations

import os

APP_VERSION = "1.0.0"


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
