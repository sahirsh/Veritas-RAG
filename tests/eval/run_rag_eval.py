"""
Run Veritas-RAG golden eval with DeepEval generation metrics.

Prerequisites:
  - API running (default http://localhost:8000), e.g. docker compose up
  - GEMINI_API_KEY and GROQ_API_KEY in .env (RAG + DeepEval judge)
  - PDFs: place under tests/eval/fixtures/ OR map paths in tests/eval/pdf_paths.json

Usage:
  python tests/eval/run_rag_eval.py
  python tests/eval/run_rag_eval.py --limit 3
  python tests/eval/run_rag_eval.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from groq_judge import GroqJudge
_REPO_ROOT = _EVAL_DIR.parent.parent
_GOLDENS_PATH = _EVAL_DIR / "goldens.json"
_FIXTURES_DIR = _EVAL_DIR / "fixtures"
_PDF_PATHS_FILE = _EVAL_DIR / "pdf_paths.json"
_RESULTS_PATH = _EVAL_DIR / "results.json"

# Stable eval identity — separate from browser cookies.
EVAL_USER_TOKEN = "a1b2c3d4-e5f6-4789-a012-3456789abcde"
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"
METRIC_THRESHOLD = 0.7


def _load_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _load_goldens() -> dict[str, Any]:
    with _GOLDENS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _load_pdf_paths() -> dict[str, str]:
    paths: dict[str, str] = {}
    if _PDF_PATHS_FILE.is_file():
        with _PDF_PATHS_FILE.open(encoding="utf-8") as f:
            paths.update(json.load(f))
    return paths


def _resolve_pdf(pdf_name: str, path_overrides: dict[str, str]) -> Path | None:
    if pdf_name in path_overrides:
        candidate = Path(path_overrides[pdf_name]).expanduser()
        if candidate.is_file():
            return candidate
        print(f"  [warn] Mapped PDF missing: {candidate}", file=sys.stderr)
    fixture = _FIXTURES_DIR / pdf_name
    if fixture.is_file():
        return fixture
    return None


def _headers() -> dict[str, str]:
    return {"X-User-Token": EVAL_USER_TOKEN}


def _api_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def _check_health(api_url: str) -> dict[str, Any]:
    resp = requests.get(_api_url(api_url, "/health"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _list_documents(api_url: str) -> list[dict[str, Any]]:
    resp = requests.get(_api_url(api_url, "/documents"), headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json().get("items", [])


def _upload_pdf(api_url: str, pdf_path: Path, upload_as: str | None = None) -> dict[str, Any]:
    stored_name = upload_as or pdf_path.name
    with pdf_path.open("rb") as fh:
        resp = requests.post(
            _api_url(api_url, "/documents/upload"),
            headers=_headers(),
            files={"file": (stored_name, fh, "application/pdf")},
            timeout=300,
        )
    if not resp.ok:
        raise RuntimeError(f"Upload failed for {pdf_path.name}: {resp.status_code} {resp.text}")
    return resp.json()


def _ensure_documents(
    api_url: str, pdf_names: set[str], path_overrides: dict[str, str]
) -> dict[str, int]:
    """Return mapping golden pdf filename -> document id."""
    existing = {doc["filename"]: int(doc["id"]) for doc in _list_documents(api_url)}
    doc_ids: dict[str, int] = {}

    for pdf_name in sorted(pdf_names):
        if pdf_name in existing:
            doc_ids[pdf_name] = existing[pdf_name]
            print(f"  reusing document id={existing[pdf_name]} for {pdf_name}")
            continue

        pdf_path = _resolve_pdf(pdf_name, path_overrides)
        if pdf_path is None:
            print(f"  [skip] PDF not found for {pdf_name}", file=sys.stderr)
            continue

        print(f"  uploading {pdf_path} as {pdf_name} ...")
        uploaded = _upload_pdf(api_url, pdf_path, upload_as=pdf_name)
        doc_ids[pdf_name] = int(uploaded["id"])
        print(f"  uploaded id={uploaded['id']} chunks={uploaded.get('chunk_count')}")

    return doc_ids


def _query(
    api_url: str, question: str, document_id: int, top_k: int
) -> dict[str, Any]:
    resp = requests.post(
        _api_url(api_url, "/query"),
        headers={**_headers(), "Content-Type": "application/json"},
        json={"question": question, "top_k": top_k, "document_ids": [document_id]},
        timeout=300,
    )
    if not resp.ok:
        raise RuntimeError(f"Query failed: {resp.status_code} {resp.text}")
    return resp.json()


def _build_judge() -> GeminiModel | GroqJudge:
    provider = os.environ.get("DEEPEVAL_JUDGE", "groq").strip().lower()
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when DEEPEVAL_JUDGE=gemini.")
        model_name = os.environ.get("DEEPEVAL_GEMINI_MODEL", DEFAULT_JUDGE_MODEL)
        return GeminiModel(model=model_name, api_key=api_key)
    return GroqJudge()


def _run_generation_metrics(
    test_cases: list[LLMTestCase],
    case_meta: list[dict[str, Any]],
    judge: GeminiModel | GroqJudge,
    threshold: float,
) -> list[dict[str, Any]]:
    """Measure Answer Relevancy + Faithfulness sequentially (Groq-friendly)."""
    metric_classes = [AnswerRelevancyMetric, FaithfulnessMetric]
    rows: list[dict[str, Any]] = []

    for tc, meta in zip(test_cases, case_meta):
        row = dict(meta)
        row["metrics"] = {}
        print(f"  scoring {meta['id']} ...")
        for cls in metric_classes:
            metric = cls(threshold=threshold, model=judge, include_reason=True)
            name = cls.__name__
            try:
                metric.measure(tc)
                row["metrics"][name] = {
                    "score": metric.score,
                    "success": metric.success,
                    "reason": metric.reason,
                    "threshold": threshold,
                }
                score_txt = f"{metric.score:.2f}" if metric.score is not None else "n/a"
                mark = "pass" if metric.success else "fail"
                print(f"    {name}: {score_txt} ({mark})")
            except Exception as exc:
                row["metrics"][name] = {
                    "score": None,
                    "success": False,
                    "reason": str(exc),
                    "threshold": threshold,
                }
                print(f"    {name}: error ({exc})")
        rows.append(row)

    return rows


def _summarize_rows(rows: list[dict[str, Any]], skipped: int, threshold: float, judge_name: str) -> dict[str, Any]:
    scores: dict[str, list[float]] = {}
    by_pdf: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        pdf_name = row.get("pdf", "unknown")
        by_pdf.setdefault(pdf_name, {})
        for name, data in row.get("metrics", {}).items():
            score = data.get("score")
            if isinstance(score, (int, float)):
                scores.setdefault(name, []).append(float(score))
                by_pdf[pdf_name].setdefault(name, []).append(float(score))

    def _avg(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    averages = {name: _avg(vals) for name, vals in scores.items()}
    averages_by_pdf = {
        pdf: {name: _avg(vals) for name, vals in metric_map.items()}
        for pdf, metric_map in by_pdf.items()
    }

    return {
        "eval_token": EVAL_USER_TOKEN,
        "case_count": len(rows),
        "skipped": skipped,
        "threshold": threshold,
        "judge_model": judge_name,
        "averages": averages,
        "averages_by_pdf": averages_by_pdf,
        "cases": rows,
    }


def main() -> int:
    # DeepEval/Rich can fail on Windows cp1252 consoles (emoji in progress UI).
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Groq free tier + many judge calls can exceed DeepEval's default timeouts.
    os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")

    parser = argparse.ArgumentParser(description="Run Veritas-RAG DeepEval generation metrics.")
    parser.add_argument("--api-url", default=os.environ.get("VERITAS_API_URL", DEFAULT_API_URL))
    parser.add_argument("--limit", type=int, default=0, help="Max goldens to run (0 = all)")
    parser.add_argument("--threshold", type=float, default=METRIC_THRESHOLD)
    parser.add_argument("--no-save", action="store_true", help="Do not write results.json")
    args = parser.parse_args()

    _load_dotenv()

    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("Warning: GROQ_API_KEY unset — /query generation may fail.", file=sys.stderr)

    print(f"API: {args.api_url}")
    health = _check_health(args.api_url)
    print(f"Health: {health.get('status')} db={health.get('database')} gemini={health.get('gemini')} groq={health.get('groq')}")

    payload = _load_goldens()
    goldens = payload["goldens"]
    default_top_k = int(payload.get("defaults", {}).get("top_k", 5))
    if args.limit > 0:
        goldens = goldens[: args.limit]

    pdf_names = {g["pdf"] for g in goldens}
    path_overrides = _load_pdf_paths()

    print("Ensuring PDFs are indexed...")
    doc_ids = _ensure_documents(args.api_url, pdf_names, path_overrides)

    test_cases: list[LLMTestCase] = []
    case_meta: list[dict[str, Any]] = []
    skipped = 0

    print("Running RAG queries...")
    for golden in goldens:
        gid = golden["id"]
        pdf_name = golden["pdf"]
        if pdf_name not in doc_ids:
            print(f"  [skip] {gid}: no document for {pdf_name}", file=sys.stderr)
            skipped += 1
            continue

        top_k = int(golden.get("top_k", default_top_k))
        try:
            result = _query(
                args.api_url,
                golden["question"],
                doc_ids[pdf_name],
                top_k,
            )
        except RuntimeError as exc:
            print(f"  [fail] {gid}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        retrieval_context = [hit["content"] for hit in result.get("hits", [])]
        test_cases.append(
            LLMTestCase(
                input=golden["question"],
                actual_output=result.get("answer", ""),
                retrieval_context=retrieval_context,
                expected_output=golden.get("expected_output"),
            )
        )
        case_meta.append(
            {
                "id": gid,
                "pdf": pdf_name,
                "document_id": doc_ids[pdf_name],
                "question": golden["question"],
                "answer": result.get("answer", ""),
                "hit_count": len(retrieval_context),
                "tags": golden.get("tags", []),
            }
        )
        print(f"  ok {gid} ({len(retrieval_context)} hits)")

    if not test_cases:
        print("No test cases to evaluate. Add PDFs to tests/eval/fixtures/ or pdf_paths.json.", file=sys.stderr)
        return 1

    judge = _build_judge()

    print(f"\nDeepEval generation metrics: {len(test_cases)} cases, threshold={args.threshold}")
    print(f"Judge: {judge.get_model_name()}\n")
    case_rows = _run_generation_metrics(test_cases, case_meta, judge, args.threshold)

    summary = _summarize_rows(
        case_rows,
        skipped=skipped,
        threshold=args.threshold,
        judge_name=judge.get_model_name(),
    )

    if not args.no_save:
        _RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {_RESULTS_PATH}")

    print("\n--- Overall averages ---")
    for name, avg in summary["averages"].items():
        print(f"  {name}: {avg:.3f}" if avg is not None else f"  {name}: n/a")

    print("\n--- Averages by PDF ---")
    for pdf_name, metric_avgs in summary.get("averages_by_pdf", {}).items():
        parts = []
        for mname, avg in metric_avgs.items():
            if avg is not None:
                parts.append(f"{mname}={avg:.2f}")
        print(f"  {pdf_name}: {', '.join(parts)}")

    print("\n--- Per case ---")
    for row in summary["cases"]:
        parts = []
        for mname, mdata in row.get("metrics", {}).items():
            score = mdata.get("score")
            ok = mdata.get("success")
            if score is not None:
                parts.append(f"{mname}={score:.2f}{'✓' if ok else '✗'}")
        print(f"  {row['id']}: {', '.join(parts)}")

    failed = any(
        not mdata.get("success", True)
        for row in summary["cases"]
        for mdata in row.get("metrics", {}).values()
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
