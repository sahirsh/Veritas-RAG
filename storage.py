"""Helpers for PDF files stored on disk."""

from __future__ import annotations

from pathlib import Path


def resolve_uploaded_file(upload_dir: Path, storage_path: str) -> Path:
    """
    Resolve a stored filename under upload_dir and reject path traversal.
    """
    base = upload_dir.resolve()
    candidate = (base / storage_path).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError("Invalid storage path")
    return candidate


def delete_uploaded_file(upload_dir: Path, storage_path: str | None) -> None:
    if not storage_path:
        return
    try:
        path = resolve_uploaded_file(upload_dir, storage_path)
    except ValueError:
        return
    path.unlink(missing_ok=True)
