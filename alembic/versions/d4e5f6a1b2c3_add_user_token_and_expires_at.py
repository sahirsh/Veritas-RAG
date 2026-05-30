"""add user_token and expires_at to documents

Revision ID: d4e5f6a1b2c3
Revises: a1b2c3d4e5f6
Create Date: 2026-05-23 12:00:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "d4e5f6a1b2c3"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None

# Sentinel token assigned to all pre-isolation rows so they remain queryable.
LEGACY_TOKEN = "00000000-0000-0000-0000-000000000000"


def upgrade() -> None:
    # Add columns as nullable first so we can backfill before enforcing NOT NULL.
    op.add_column(
        "documents",
        sa.Column("user_token", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Backfill existing rows.
    op.execute(
        sa.text(
            "UPDATE documents "
            f"SET user_token = '{LEGACY_TOKEN}', "
            "    expires_at = NOW() AT TIME ZONE 'UTC' + INTERVAL '48 hours' "
            "WHERE user_token IS NULL"
        )
    )

    # Tighten to NOT NULL now that every row has a value.
    op.alter_column("documents", "user_token", nullable=False)
    op.alter_column("documents", "expires_at", nullable=False)

    # Index speeds up the per-token WHERE clause on every document endpoint.
    op.create_index(
        "ix_documents_user_token",
        "documents",
        ["user_token"],
        unique=False,
    )
    # Composite index for the common pattern: token + expiry filter together.
    op.create_index(
        "ix_documents_user_token_expires_at",
        "documents",
        ["user_token", "expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_documents_user_token_expires_at", table_name="documents")
    op.drop_index("ix_documents_user_token", table_name="documents")
    op.drop_column("documents", "expires_at")
    op.drop_column("documents", "user_token")
