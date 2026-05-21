from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Veritas-RAG"
    assert "version" in body


@pytest.mark.asyncio
async def test_health_without_database(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("healthy", "degraded")
    assert body["database"] in ("unconfigured", "unreachable", "ready")
    assert body["gemini"] in ("configured", "unconfigured")
    assert body["groq"] in ("configured", "unconfigured")


@pytest.mark.asyncio
async def test_query_requires_database(client):
    resp = await client.post("/query", json={"question": "What is this?"})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_list_documents_requires_database(client):
    resp = await client.get("/documents")
    assert resp.status_code == 503
