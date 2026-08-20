"""
Streamlit frontend for the Veritas-RAG FastAPI backend.

Run with:
    streamlit run app.py

The backend is expected to be available at http://localhost:8000 by default.
Override with the VERITAS_API_URL environment variable or the sidebar setting.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import requests
import streamlit as st
import extra_streamlit_components as stx


def _resolve_default_api_url() -> str:
    """
    Resolve the backend URL with this precedence:
      1. Streamlit Cloud secret  (st.secrets["VERITAS_API_URL"])
      2. Environment variable    (VERITAS_API_URL)
      3. Local fallback          (http://localhost:8000)
    """
    try:
        secret_url = st.secrets.get("VERITAS_API_URL") if hasattr(st, "secrets") else None
        if secret_url:
            return str(secret_url)
    except Exception:
        pass
    return os.environ.get("VERITAS_API_URL", "http://localhost:8000")


DEFAULT_API_URL = _resolve_default_api_url()
REQUEST_TIMEOUT = 120  # seconds — embedding + generation can take a while

st.set_page_config(
    page_title="Veritas-RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------- Custom CSS ----------------------------- #


CUSTOM_CSS = """
<style>
    /* ── Hide Streamlit chrome ───────────────────────────────────────── */
    .stAppDeployButton {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
    }

    /* ── Global background gradient ──────────────────────────────────── */
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(139, 92, 246, 0.18), transparent 60%),
            radial-gradient(1000px 500px at 110% 10%, rgba(56, 189, 248, 0.12), transparent 60%),
            linear-gradient(180deg, #0b0d12 0%, #0a0c11 100%) !important;
    }

    /* ── Hero header card ────────────────────────────────────────────── */
    .veritas-hero {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.22), rgba(56, 189, 248, 0.14));
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    .veritas-hero h1 {
        margin: 0 0 4px 0;
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #c4b5fd, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .veritas-hero p {
        margin: 0;
        color: #a8aab8;
        font-size: 0.95rem;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11141d 0%, #0d1018 100%) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(90deg, #c4b5fd, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.4rem;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(139, 92, 246, 0.18) !important;
        margin: 14px 0 !important;
    }

    /* ── Chat message bubbles ────────────────────────────────────────── */
    div[data-testid="stChatMessage"] {
        background: rgba(22, 26, 36, 0.7) !important;
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 14px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        border-color: rgba(56, 189, 248, 0.25);
    }

    /* ── Chat input ──────────────────────────────────────────────────── */
    div[data-testid="stChatInput"] textarea {
        background: rgba(22, 26, 36, 0.9) !important;
        border: 1px solid rgba(139, 92, 246, 0.35) !important;
        border-radius: 12px !important;
        color: #e8e8f0 !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────── */
    .stButton button {
        background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: transform 0.08s ease, box-shadow 0.15s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.35) !important;
    }
    .stButton button:active {
        transform: translateY(0);
    }

    /* ── Expanders ───────────────────────────────────────────────────── */
    div[data-testid="stExpander"] {
        background: rgba(22, 26, 36, 0.5) !important;
        border: 1px solid rgba(139, 92, 246, 0.18) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 500;
    }

    /* ── File uploader dropzone ──────────────────────────────────────── */
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(139, 92, 246, 0.06) !important;
        border: 1.5px dashed rgba(139, 92, 246, 0.45) !important;
        border-radius: 12px !important;
    }

    /* ── Citation card ───────────────────────────────────────────────── */
    .citation-card {
        background: rgba(11, 13, 18, 0.6);
        border-left: 3px solid #8b5cf6;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.9em;
        color: #c8c8d4;
    }
    .citation-meta {
        color: #8b8ea0;
        font-size: 0.8em;
        margin-bottom: 6px;
    }
    .citation-pill {
        display: inline-block;
        background: rgba(139, 92, 246, 0.18);
        color: #c4b5fd;
        padding: 1px 8px;
        border-radius: 999px;
        font-size: 0.75em;
        margin-right: 6px;
        font-weight: 600;
    }

    /* ── Status alerts (success/warning/error) ───────────────────────── */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* ── Scrollbar polish (Chromium) ─────────────────────────────────── */
    ::-webkit-scrollbar {width: 10px; height: 10px;}
    ::-webkit-scrollbar-track {background: transparent;}
    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.35);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {background: rgba(139, 92, 246, 0.55);}
</style>
"""


def _inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------- Session token ----------------------------- #

COOKIE_NAME = "veritas_user_token"
TOKEN_QUERY_PARAM = "token"  # stripped if present; never used for auth
COOKIE_DAYS = 365
MAX_COOKIE_READ_ATTEMPTS = 4


def _cookie_secure() -> bool:
    """Use Secure cookies on HTTPS (Render); allow HTTP for local dev."""
    try:
        if st.context.headers.get("X-Forwarded-Proto") == "https":
            return True
    except Exception:
        pass
    return os.environ.get("VERITAS_COOKIE_SECURE", "").lower() in ("1", "true", "yes")


def _valid_token(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        return None


def _get_cookie_manager() -> stx.CookieManager:
    # One manager per Streamlit session — do NOT cache globally with @st.cache_resource.
    if "cookie_manager" not in st.session_state:
        st.session_state.cookie_manager = stx.CookieManager(key="veritas_cookie_manager")
    return st.session_state.cookie_manager


def _strip_token_from_url() -> None:
    """Drop ?token= from shared links so URLs are not credentials."""
    if TOKEN_QUERY_PARAM in st.query_params:
        del st.query_params[TOKEN_QUERY_PARAM]
        st.rerun()


def _write_user_cookie(cookie_manager: stx.CookieManager, token: str) -> None:
    if st.session_state.get("_cookie_token_written") == token:
        return
    expires = datetime.now(timezone.utc) + timedelta(days=COOKIE_DAYS)
    cookie_manager.set(
        COOKIE_NAME,
        token,
        expires_at=expires,
        path="/",
        same_site="lax",
        secure=_cookie_secure(),
        key="set_veritas_token",
    )
    st.session_state["_cookie_token_written"] = token


def _get_or_create_user_token() -> str:
    """Return a stable per-browser UUID stored in a cookie."""
    if "user_token" in st.session_state:
        return st.session_state["user_token"]

    _strip_token_from_url()

    cookie_manager = _get_cookie_manager()

    cookies = cookie_manager.get_all()
    if cookies is None:
        st.stop()

    cookie_token = _valid_token(cookies.get(COOKIE_NAME))
    if cookie_token:
        st.session_state["user_token"] = cookie_token
        return cookie_token

    attempts = int(st.session_state.get("_cookie_read_attempts", 0))
    if not cookies and attempts < MAX_COOKIE_READ_ATTEMPTS:
        st.session_state["_cookie_read_attempts"] = attempts + 1
        st.stop()

    if "pending_user_token" not in st.session_state:
        st.session_state["pending_user_token"] = str(uuid.uuid4())

    pending = st.session_state["pending_user_token"]

    if st.session_state.get("_cookie_token_written") != pending:
        _write_user_cookie(cookie_manager, pending)
        st.rerun()

    st.session_state["user_token"] = pending
    return pending


# ----------------------------- API helpers ----------------------------- #


class BackendError(Exception):
    """Raised when the FastAPI backend returns an error or is unreachable."""


def _api_url(path: str) -> str:
    return f"{DEFAULT_API_URL.rstrip('/')}{path}"


def _token_headers(token: str) -> dict[str, str]:
    return {"X-User-Token": token}


def _extract_error(resp: requests.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict) and "msg" in first:
            return str(first["msg"])
    if detail:
        return str(detail)
    return f"HTTP {resp.status_code}"


def check_health() -> dict[str, Any] | None:
    """Return health payload, or None if the backend is unreachable."""
    try:
        resp = requests.get(_api_url("/health"), timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def upload_pdf(file_name: str, file_bytes: bytes, token: str) -> dict[str, Any]:
    files = {"file": (file_name, file_bytes, "application/pdf")}
    try:
        resp = requests.post(
            _api_url("/documents/upload"),
            files=files,
            headers=_token_headers(token),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.ConnectionError as exc:
        raise BackendError(
            "Cannot reach the backend. Make sure FastAPI is running."
        ) from exc
    except requests.Timeout as exc:
        raise BackendError("Upload timed out. The PDF may be too large.") from exc
    except requests.RequestException as exc:
        raise BackendError(f"Network error: {exc}") from exc

    if not resp.ok:
        raise BackendError(_extract_error(resp))
    return resp.json()


def list_documents(token: str) -> list[dict[str, Any]]:
    try:
        resp = requests.get(
            _api_url("/documents"),
            headers=_token_headers(token),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])
    except requests.RequestException:
        return []


def renew_documents(token: str) -> None:
    """Best-effort TTL extension for this user's active documents."""
    try:
        requests.post(
            _api_url("/documents/renew"),
            headers=_token_headers(token),
            timeout=10,
        )
    except requests.RequestException:
        pass


def _maybe_renew_document_ttl(token: str) -> None:
    if st.session_state.get("ttl_renewed"):
        return
    renew_documents(token)
    st.session_state["ttl_renewed"] = True


def delete_document(doc_id: int, token: str) -> None:
    try:
        resp = requests.delete(
            _api_url(f"/documents/{doc_id}"),
            headers=_token_headers(token),
            timeout=15,
        )
    except requests.RequestException as exc:
        raise BackendError(f"Could not delete document: {exc}") from exc
    if not resp.ok and resp.status_code != 204:
        raise BackendError(_extract_error(resp))


def query_rag(question: str, top_k: int, token: str) -> dict[str, Any]:
    payload = {"question": question, "top_k": top_k}
    try:
        resp = requests.post(
            _api_url("/query"),
            json=payload,
            headers=_token_headers(token),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.ConnectionError as exc:
        raise BackendError(
            "Cannot reach the backend. Please try again shortly."
        ) from exc
    except requests.Timeout as exc:
        raise BackendError("The query timed out. Try a simpler question.") from exc
    except requests.RequestException as exc:
        raise BackendError(f"Network error: {exc}") from exc

    if not resp.ok:
        raise BackendError(_extract_error(resp))
    return resp.json()


# ----------------------------- Session state ----------------------------- #


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()


_init_state()

# Resolve (or generate) the session token once per script run.
_USER_TOKEN: str = _get_or_create_user_token()
_maybe_renew_document_ttl(_USER_TOKEN)


# ----------------------------- Helpers ----------------------------- #


def _format_expires(expires_at_str: str | None) -> str:
    """Return a human-readable 'Expires in Xh Ym' string, or empty string on failure."""
    if not expires_at_str:
        return ""
    try:
        exp = datetime.fromisoformat(expires_at_str)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        delta = exp - datetime.now(timezone.utc)
        total_seconds = int(delta.total_seconds())
        if total_seconds <= 0:
            return "Expired"
        hours, remainder = divmod(total_seconds, 3600)
        minutes = remainder // 60
        if hours > 0:
            return f"Expires in {hours}h {minutes}m"
        return f"Expires in {minutes}m"
    except Exception:
        return ""


# ----------------------------- Sidebar ----------------------------- #


def _render_health_badge(health: dict[str, Any] | None) -> None:
    if health is None:
        st.error("Backend offline", icon="🔴")
        return

    status = health.get("status", "unknown")
    if status == "healthy":
        st.success(f"Backend healthy · v{health.get('version', '?')}", icon="🟢")
    else:
        st.warning(f"Backend degraded · v{health.get('version', '?')}", icon="🟡")

    with st.expander("Service details", expanded=False):
        st.write(
            {
                "database": health.get("database"),
                "gemini": health.get("gemini"),
                "groq": health.get("groq"),
            }
        )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# 📚 Veritas-RAG")
        st.caption("Ground answers in your PDFs")

        # Session identity pill
        short_token = _USER_TOKEN[:8]
        st.markdown(
            f"<small style='color:#555870;'>Session&nbsp;"
            f"<code style='color:#6d6f84;'>{short_token}…</code></small>",
            unsafe_allow_html=True,
        )

        st.divider()
        health = check_health()
        _render_health_badge(health)

        st.divider()
        st.subheader("Upload PDF")
        uploaded = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if uploaded is not None:
            file_key = f"{uploaded.name}:{uploaded.size}"
            if file_key not in st.session_state.uploaded_files:
                with st.spinner(f"Uploading and indexing '{uploaded.name}'…"):
                    try:
                        result = upload_pdf(uploaded.name, uploaded.getvalue(), _USER_TOKEN)
                    except BackendError as exc:
                        st.error(f"Upload failed: {exc}", icon="⚠️")
                    else:
                        st.session_state.uploaded_files.add(file_key)
                        st.success(
                            f"Indexed '{result['filename']}' — "
                            f"{result['embedded_chunk_count']} chunk(s) embedded.",
                            icon="✅",
                        )
                        st.rerun()

        st.divider()
        st.subheader("Indexed documents")
        docs = list_documents(_USER_TOKEN)
        if not docs:
            st.caption("No documents yet. Upload a PDF to get started.")
        else:
            for doc in docs:
                expiry_str = _format_expires(doc.get("expires_at"))
                expiry_html = (
                    f" · <span style='color:#555870;font-size:0.75em;'>{expiry_str}</span>"
                    if expiry_str
                    else ""
                )
                cols = st.columns([0.78, 0.22])
                with cols[0]:
                    st.markdown(
                        f"**{doc['filename']}**  \n"
                        f"<small style='color:#8b8ea0;'>"
                        f"{doc['chunk_count']} chunks · id {doc['id']}"
                        f"{expiry_html}</small>",
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    if st.button("🗑", key=f"del_{doc['id']}", help="Delete"):
                        try:
                            delete_document(doc["id"], _USER_TOKEN)
                        except BackendError as exc:
                            st.error(str(exc))
                        else:
                            st.rerun()

        st.divider()
        if st.button("🧹 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        with st.expander("⚙️ Advanced", expanded=False):
            st.slider(
                "Top-k passages",
                min_value=1,
                max_value=20,
                key="top_k",
                help=(
                    "How many of the most relevant document chunks to ground "
                    "the answer on. Higher values give the model more context "
                    "but may introduce noise. Default 5 is a good starting point."
                ),
            )


# ----------------------------- Main chat ----------------------------- #


def _render_hero() -> None:
    st.markdown(
        """
        <div class="veritas-hero">
            <h1>📚 Veritas-RAG</h1>
            <p>Ask questions grounded in your indexed PDFs · powered by Gemini embeddings &amp; Groq llama-3.1-8b-instant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"📎 Citations ({len(citations)})", expanded=False):
        for i, cit in enumerate(citations, start=1):
            similarity = cit.get("cosine_similarity")
            sim_str = (
                f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "?"
            )
            filename = cit.get("filename", "unknown")
            chunk_index = cit.get("chunk_index", "?")
            content = cit.get("content", "")
            preview = content if len(content) <= 600 else content[:600] + "…"

            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-meta">
                        <span class="citation-pill">[{i}]</span>
                        <strong>{filename}</strong> · chunk {chunk_index} · similarity {sim_str}
                    </div>
                    {preview}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_message(msg: dict[str, Any]) -> None:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_citations(msg.get("citations", []))


def _render_chat() -> None:
    _render_hero()

    if not st.session_state.messages:
        st.info(
            "💡 Upload a PDF from the sidebar to get started, then ask anything "
            "about its contents.",
            icon="✨",
        )

    for msg in st.session_state.messages:
        _render_message(msg)

    prompt = st.chat_input("Ask a question about your documents…")
    if not prompt:
        return

    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    _render_message(user_msg)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                result = query_rag(prompt, top_k=st.session_state.top_k, token=_USER_TOKEN)
            except BackendError as exc:
                error_text = f"⚠️ {exc}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "citations": []}
                )
                return

        answer = result.get("answer", "(no answer returned)")
        citations = result.get("answer_sources") or result.get("hits") or []
        st.markdown(answer)
        _render_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )


# ----------------------------- Entrypoint ----------------------------- #


def main() -> None:
    _inject_css()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
