"""Async SQLAlchemy engine and session factory for Postgres (psycopg 3 driver)."""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base for future ORM models (chunks, documents, etc.)."""


def get_async_database_url() -> str | None:
    """
    SQLAlchemy async URLs for psycopg 3 use postgresql+psycopg://
    while Compose typically sets DATABASE_URL=postgresql://...
    """
    raw = os.environ.get("DATABASE_URL")
    if not raw:
        return None
    if raw.startswith("postgresql+psycopg://"):
        return raw
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


def create_engine_and_sessionmaker():
    url = get_async_database_url()
    if not url:
        return None, None
    engine = create_async_engine(url, pool_pre_ping=True)
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return engine, session_factory


async def dispose_engine(engine) -> None:
    if engine is not None:
        await engine.dispose()
