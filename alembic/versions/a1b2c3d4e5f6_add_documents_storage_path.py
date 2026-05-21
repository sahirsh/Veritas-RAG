"""add documents storage_path

Revision ID: a1b2c3d4e5f6
Revises: 2a9f3c1e8b47
Create Date: 2026-05-16 12:00:00.000000

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "a1b2c3d4e5f6"
down_revision = "2a9f3c1e8b47"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("storage_path", sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "storage_path")
