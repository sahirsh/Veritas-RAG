"""add document_chunks table

Revision ID: 2a9f3c1e8b47
Revises: 87b583e2328a
Create Date: 2026-04-19 12:00:00.000000

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from models import EMBEDDING_DIM


revision = "2a9f3c1e8b47"
down_revision = "87b583e2328a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_chunk_index",
        ),
    )
    op.create_index(
        op.f("ix_document_chunks_document_id"),
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.execute(
        sa.text(
            "CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw"))
    op.drop_index(op.f("ix_document_chunks_document_id"), table_name="document_chunks")
    op.drop_table("document_chunks")
