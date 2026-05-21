from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure tests do not require a live database unless explicitly enabled.
os.environ.setdefault("DATABASE_URL", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("GROQ_API_KEY", "")

from main import app  # noqa: E402


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
