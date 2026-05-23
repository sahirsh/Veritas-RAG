from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    pass


SYSTEM_PROMPT = (
    "You are a knowledgeable research assistant. Your job is to answer the user's "
    "question using the numbered passages provided below as your source of truth.\n"
    "\n"
    "How to answer:\n"
    "- Ground every factual claim in the provided passages. Do not invent facts that "
    "are not supported by the text.\n"
    "- Write a complete, helpful response. A one-word answer is almost never enough. "
    "Aim for 2-5 sentences for simple questions, and longer when the question deserves "
    "more explanation. Provide context and connect related details across passages.\n"
    "- When passages contain partial or conflicting information, acknowledge it. If the "
    "passages truly don't address the question, say so honestly rather than guessing.\n"
    "- You may rephrase, summarize, and synthesize freely. Quoting short phrases verbatim "
    "is fine when precision matters. Light common-sense reasoning over the passages is "
    "allowed (definitions, plain implications, connecting facts), but do not introduce "
    "outside knowledge that isn't grounded in the text.\n"
    "- Write naturally for a curious reader. Avoid robotic, telegraphic answers.\n"
    "\n"
    "Output format — return ONLY a JSON object with these keys and nothing else:\n"
    '  - "answer": string — your full response to the user\n'
    '  - "citations": array of integers — passage numbers (1-indexed) you actually used\n'
    "\n"
    "Output rules:\n"
    "- If you used any passage, citations must include at least one valid passage number.\n"
    "- Only cite passage numbers that appear in the provided text.\n"
    "- Do not include any other keys. Do not wrap the JSON in markdown code fences."
)


@dataclass(frozen=True)
class GenerationResult:
    model: str
    answer: str
    citation_passage_numbers: list[int]


def _groq_error_message(exc: BaseException | None) -> str:
    if exc is None:
        return "The AI service is temporarily unavailable. Please try again later."

    text = str(exc).lower()
    if "api key" in text or "authentication" in text or "unauthorized" in text:
        return (
            "Groq chat service authentication failed. "
            "Verify GROQ_API_KEY if you operate this API."
        )
    if "rate limit" in text or "quota" in text or "too many requests" in text:
        return "The Groq chat service rate limit was exceeded. Please try again shortly."

    return "The AI service is temporarily unavailable. Please try again later."


async def generate_answer_groq(
    question: str,
    context_chunks: list[str],
    *,
    model: str | None = None,
    max_retries: int = 5,
) -> GenerationResult:
    """
    Answer `question` using Groq Chat Completions, constrained to `context_chunks`.

    Requires env var GROQ_API_KEY. Model defaults to GROQ_CHAT_MODEL or llama-3.1-8b-instant.

    Returns the resolved model name and answer text.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GenerationError("Groq chat service is not configured on this server.")

    resolved_model = model or os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")

    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    from groq import AsyncGroq

    client = AsyncGroq(api_key=api_key)

    parts = ["### Provided text\n"]
    for i, chunk in enumerate(context_chunks, start=1):
        parts.append(f"[Passage {i}]\n{chunk.strip()}\n")
    parts.append("\n### Question\n")
    parts.append(question.strip())
    user_message = "".join(parts)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0] if resp.choices else None
            content = getattr(choice.message, "content", None) if choice else None
            if content is None:
                raise GenerationError("Chat completion returned empty content")
            parsed = _parse_generation_json(content)
            answer = str(parsed.get("answer", "")).strip()
            if not answer:
                raise GenerationError("Chat completion returned empty answer")
            citations = _parse_citations(parsed.get("citations"))
            return GenerationResult(
                model=resolved_model,
                answer=answer,
                citation_passage_numbers=citations,
            )
        except GenerationError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(2**attempt, 20))

    if last_exc is not None:
        logger.error(
            "Groq generation request failed after %s attempts: %s",
            max_retries + 1,
            last_exc,
            exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
        )
    msg = _groq_error_message(last_exc)
    raise GenerationError(msg) from last_exc


def _parse_generation_json(content: str) -> dict[str, Any]:
    """
    The model is instructed to return raw JSON. In practice, it may occasionally
    wrap it in text; we try a strict parse first, then a best-effort extraction.
    """
    s = (content or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Best-effort: extract the first top-level JSON object.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    raise GenerationError("Could not parse model JSON response")


def _parse_citations(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            n = int(item)
        except Exception:
            continue
        if n > 0:
            out.append(n)
    # de-dupe preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped
