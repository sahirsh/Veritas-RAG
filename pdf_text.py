"""Extract plain text from PDF files using PyMuPDF (fitz)."""

from __future__ import annotations

from pathlib import Path


class PdfExtractError(Exception):
    """Raised when a PDF cannot be opened or text extraction fails."""


def extract_text_from_pdf(path: Path) -> str:
    """
    Read a PDF from disk and return concatenated text for all pages.

    Pages are separated by two newlines. Returns an empty string if there is
    no extractable text (e.g. scanned pages without OCR).
    """
    import fitz

    path = Path(path)
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise PdfExtractError(
            "This file could not be opened as a PDF. It may be corrupted or incomplete."
        ) from exc
    try:
        parts: list[str] = []
        for page in doc:
            try:
                parts.append(page.get_text())
            except Exception as exc:
                raise PdfExtractError(
                    "This PDF could not be read fully. It may be corrupted."
                ) from exc
        return "\n\n".join(parts)
    finally:
        doc.close()


def chunk_text(text: str, *, words_per_chunk: int = 500, overlap_words: int = 0) -> list[str]:
    """
    Split text into word-based chunks.

    - Words are defined by whitespace splitting.
    - Output chunks are joined with single spaces (whitespace normalized).
    - If `overlap_words` > 0, each chunk repeats the last N words of the
      previous chunk to preserve context across boundaries.
    """
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be > 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if overlap_words >= words_per_chunk:
        raise ValueError("overlap_words must be < words_per_chunk")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = words_per_chunk - overlap_words
    for start in range(0, len(words), step):
        end = min(start + words_per_chunk, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break

    return chunks
