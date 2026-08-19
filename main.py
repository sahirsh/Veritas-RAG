import asyncio
import datetime as dt
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from config import (
    APP_VERSION,
    DOCUMENT_TTL_HOURS,
    cors_origins,
    database_configured,
    gemini_configured,
    groq_configured,
)
from database import create_engine_and_sessionmaker, dispose_engine
from embeddings import EmbeddingError, EmbeddingResult, embed_texts
from generation import GenerationError, generate_answer_groq
from models import Document, DocumentChunk
from pdf_text import PdfExtractError, chunk_text, extract_text_from_pdf
from schemas import (
    DocumentDetail,
    DocumentListResponse,
    DocumentRenewResponse,
    DocumentSummary,
    DocumentUploadResponse,
    HealthResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
)
from storage import delete_uploaded_file

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_TEXT_IN_RESPONSE = 100_000
MAX_CHUNKS_TO_EMBED = 200
PDF_MAGIC = b"%PDF-"


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def _extract_user_token(x_user_token: str | None) -> str:
    """
    Validate and return the session token from the X-User-Token header.
    Raises HTTP 400 if the header is missing or not a valid UUID.
    """
    if not x_user_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Token header is required. Your session token was not sent.",
        )
    try:
        uuid.UUID(x_user_token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Token must be a valid UUID.",
        )
    return x_user_token


def _document_expires_at() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=DOCUMENT_TTL_HOURS)


# ---------------------------------------------------------------------------
# Expired-document purge
# ---------------------------------------------------------------------------


async def purge_expired_documents(factory, upload_dir: Path) -> int:
    """
    Delete all documents whose expires_at is in the past.

    Returns the number of documents deleted. Fetches storage_path values
    first so disk files can be removed after the DB rows are gone.
    """
    try:
        async with factory() as session:
            now = dt.datetime.now(dt.timezone.utc)
            expired_rows = (
                await session.execute(
                    select(Document.id, Document.storage_path).where(
                        Document.expires_at < now
                    )
                )
            ).all()

            if not expired_rows:
                return 0

            expired_ids = [r.id for r in expired_rows]
            storage_paths = [r.storage_path for r in expired_rows]

            await session.execute(
                text("DELETE FROM documents WHERE id = ANY(:ids)").bindparams(
                    ids=expired_ids
                )
            )
            await session.commit()

        for sp in storage_paths:
            if sp:
                delete_uploaded_file(upload_dir, sp)

        logger.info("Purged %d expired document(s)", len(expired_ids))
        return len(expired_ids)
    except Exception:
        logger.exception("Error purging expired documents")
        return 0


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "uploads")).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.upload_dir = upload_dir

    engine, session_factory = create_engine_and_sessionmaker()
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.db_ready = False
    app.state.db_error = None

    if engine is not None and session_factory is not None:
        try:
            async with session_factory() as session:
                result = await session.execute(text("SELECT 1"))
                if result.scalar_one() == 1:
                    app.state.db_ready = True
        except Exception as exc:
            logger.exception("Database startup probe failed")
            app.state.db_ready = False
            app.state.db_error = f"{type(exc).__name__}: {exc}"

        if app.state.db_ready:
            await purge_expired_documents(session_factory, upload_dir)

    yield

    await dispose_engine(engine)


app = FastAPI(title="Veritas-RAG", version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _session_factory():
    factory = getattr(app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not configured (set DATABASE_URL)",
        )
    return factory


def _require_db_ready() -> None:
    if not getattr(app.state, "db_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not reachable. Check DATABASE_URL and run migrations.",
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    logger.exception("Unhandled error processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "name": "Veritas-RAG",
        "version": APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Readiness-style health: reports whether required dependencies are configured
    and whether the database answered a probe query at startup.
    """
    db_state = "unconfigured"
    if database_configured():
        db_state = "ready" if getattr(app.state, "db_ready", False) else "unreachable"

    gemini_state = "configured" if gemini_configured() else "unconfigured"
    groq_state = "configured" if groq_configured() else "unconfigured"

    if (
        db_state == "ready"
        and gemini_state == "configured"
        and groq_state == "configured"
    ):
        overall = "healthy"
    else:
        overall = "degraded"

    db_error = getattr(app.state, "db_error", None) if db_state != "ready" else None

    return HealthResponse(
        status=overall,
        version=APP_VERSION,
        database=db_state,
        gemini=gemini_state,
        groq=groq_state,
        uploads_dir=str(getattr(app.state, "upload_dir", Path("uploads"))),
        database_error=db_error,
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    x_user_token: str | None = Header(default=None),
) -> DocumentListResponse:
    token = _extract_user_token(x_user_token)
    factory = _session_factory()
    _require_db_ready()

    now = dt.datetime.now(dt.timezone.utc)

    async with factory() as session:
        total = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.user_token == token, Document.expires_at > now)
                )
            ).scalar_one()
        )
        stmt = (
            select(
                Document,
                func.count(DocumentChunk.id).label("chunk_count"),
            )
            .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
            .where(Document.user_token == token, Document.expires_at > now)
            .group_by(Document.id)
            .order_by(Document.upload_date.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await session.execute(stmt)).all()

    items = []
    for doc, chunk_count in rows:
        items.append(
            DocumentSummary(
                id=doc.id,
                filename=doc.filename,
                upload_date=doc.upload_date,
                expires_at=doc.expires_at,
                user_token=doc.user_token,
                chunk_count=int(chunk_count),
                embedded_chunk_count=int(chunk_count),
            )
        )

    return DocumentListResponse(items=items, total=total, limit=limit, offset=offset)


@app.post("/documents/renew", response_model=DocumentRenewResponse)
async def renew_documents(
    x_user_token: str | None = Header(default=None),
) -> DocumentRenewResponse:
    """Extend expires_at for all non-expired documents owned by this user."""
    token = _extract_user_token(x_user_token)
    factory = _session_factory()
    _require_db_ready()

    now = dt.datetime.now(dt.timezone.utc)
    new_expires = _document_expires_at()

    async with factory() as session:
        result = await session.execute(
            update(Document)
            .where(Document.user_token == token, Document.expires_at > now)
            .values(expires_at=new_expires)
        )
        await session.commit()
        renewed_count = int(result.rowcount or 0)

    return DocumentRenewResponse(renewed_count=renewed_count, expires_at=new_expires)


@app.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int,
    x_user_token: str | None = Header(default=None),
) -> DocumentDetail:
    token = _extract_user_token(x_user_token)
    factory = _session_factory()
    _require_db_ready()

    async with factory() as session:
        doc = await session.get(
            Document,
            document_id,
            options=(selectinload(Document.chunks),),
        )
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )
        if doc.user_token != token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this document.",
            )
        now = dt.datetime.now(dt.timezone.utc)
        if doc.expires_at.replace(tzinfo=dt.timezone.utc) <= now:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} has expired.",
            )
        chunk_count = len(doc.chunks)
        embedding_model = doc.chunks[0].embedding_model if doc.chunks else None

    return DocumentDetail(
        id=doc.id,
        filename=doc.filename,
        upload_date=doc.upload_date,
        expires_at=doc.expires_at,
        user_token=doc.user_token,
        chunk_count=chunk_count,
        embedded_chunk_count=chunk_count,
        embedding_model=embedding_model,
    )


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    x_user_token: str | None = Header(default=None),
) -> None:
    token = _extract_user_token(x_user_token)
    factory = _session_factory()
    _require_db_ready()
    upload_dir: Path = app.state.upload_dir

    async with factory() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )
        if doc.user_token != token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this document.",
            )
        storage_path = doc.storage_path
        await session.delete(doc)
        await session.commit()

    delete_uploaded_file(upload_dir, storage_path)


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to store"),
    x_user_token: str | None = Header(default=None),
):
    token = _extract_user_token(x_user_token)

    if file.content_type:
        ct = file.content_type.split(";")[0].strip().lower()
        if ct not in ("application/pdf", "application/x-pdf"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Content-Type must be application/pdf",
            )

    factory = _session_factory()
    _require_db_ready()

    raw_name = file.filename or ""
    orig = Path(raw_name).name
    if not orig.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must end with .pdf",
        )

    first = await file.read(len(PDF_MAGIC))
    if first != PDF_MAGIC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not a valid PDF",
        )

    upload_dir: Path = app.state.upload_dir
    stored_name = f"{uuid.uuid4().hex}_{orig}"
    dest = upload_dir / stored_name

    total = len(first)
    try:
        with dest.open("wb") as out:
            out.write(first)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    try:
        raw_text = await asyncio.to_thread(extract_text_from_pdf, dest)
    except PdfExtractError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception:
        dest.unlink(missing_ok=True)
        logger.exception("Unexpected error extracting PDF text")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract text from this PDF. The file may be damaged.",
        )

    if not raw_text.strip():
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No extractable text found in this PDF. "
                "Scanned documents may require OCR before upload."
            ),
        )

    chunks = chunk_text(raw_text, words_per_chunk=500, overlap_words=50)
    embedded_chunk_count = 0
    embedding_model: str | None = None
    emb: EmbeddingResult | None = None
    if chunks:
        if len(chunks) > MAX_CHUNKS_TO_EMBED:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Document too large to embed safely ({len(chunks)} chunks > {MAX_CHUNKS_TO_EMBED}).",
            )
        try:
            emb = await embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")
            embedded_chunk_count = len(emb.vectors)
            embedding_model = emb.model
        except EmbeddingError as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    text_len = len(raw_text)
    if text_len <= MAX_EXTRACTED_TEXT_IN_RESPONSE:
        preview = raw_text
        truncated = False
    else:
        preview = raw_text[:MAX_EXTRACTED_TEXT_IN_RESPONSE]
        truncated = True

    expires_at = _document_expires_at()
    db_name = orig[:255]
    try:
        async with factory() as session:
            doc = Document(
                filename=db_name,
                storage_path=stored_name,
                user_token=token,
                expires_at=expires_at,
            )
            session.add(doc)
            await session.flush()
            if emb is not None and emb.vectors:
                for i, (chunk_content, vector) in enumerate(zip(emb.texts, emb.vectors)):
                    session.add(
                        DocumentChunk(
                            document_id=doc.id,
                            chunk_index=i,
                            content=chunk_content,
                            embedding=vector,
                            embedding_model=emb.model,
                        )
                    )
            await session.commit()
            await session.refresh(doc)
    except SQLAlchemyError:
        dest.unlink(missing_ok=True)
        logger.exception("Database error while saving uploaded document")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not save the document. Please try again later.",
        )
    except Exception:
        dest.unlink(missing_ok=True)
        logger.exception("Unexpected error while saving uploaded document")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not save the document. Please try again later.",
        )

    # Best-effort cleanup of any expired documents now that we know DB is healthy.
    asyncio.create_task(purge_expired_documents(factory, upload_dir))

    return DocumentUploadResponse(
        id=doc.id,
        filename=doc.filename,
        upload_date=doc.upload_date,
        expires_at=doc.expires_at,
        user_token=doc.user_token,
        extracted_text_length=text_len,
        extracted_text=preview,
        extracted_text_truncated=truncated,
        chunk_count=len(chunks),
        embedded_chunk_count=embedded_chunk_count,
        embedding_model=embedding_model,
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    x_user_token: str | None = Header(default=None),
) -> QueryResponse:
    token = _extract_user_token(x_user_token)
    factory = _session_factory()
    _require_db_ready()

    q = (request.question or "").strip()
    if not q:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="question must be a non-empty string",
        )
    if request.top_k <= 0 or request.top_k > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="top_k must be between 1 and 50",
        )

    doc_ids = request.document_ids
    if doc_ids is not None:
        if len(doc_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="document_ids must be a non-empty list when provided",
            )
        if len(doc_ids) > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="document_ids may contain at most 100 IDs",
            )

    try:
        emb = await embed_texts([q], task_type="RETRIEVAL_QUERY")
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if not emb.vectors:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding provider returned no vectors",
        )

    query_vec = emb.vectors[0]
    now = dt.datetime.now(dt.timezone.utc)

    async with factory() as session:
        if doc_ids is not None:
            # Validate that every requested ID exists AND belongs to this token.
            existing = (
                await session.execute(
                    select(Document.id).where(
                        Document.id.in_(doc_ids),
                        Document.user_token == token,
                        Document.expires_at > now,
                    )
                )
            ).scalars().all()
            missing = sorted(set(doc_ids) - set(existing))
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Unknown document_ids: {missing}",
                )

        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                Document.filename,
                DocumentChunk.embedding.cosine_distance(query_vec).label("cosine_distance"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.user_token == token, Document.expires_at > now)
            .order_by(text("cosine_distance ASC"))
            .limit(request.top_k)
        )
        if doc_ids is not None:
            stmt = stmt.where(DocumentChunk.document_id.in_(doc_ids))

        rows = (await session.execute(stmt)).all()

    hits: list[QueryHit] = []
    for row in rows:
        cosine_distance = float(row.cosine_distance)
        cosine_similarity = 1.0 - cosine_distance
        hits.append(
            QueryHit(
                document_id=int(row.document_id),
                filename=str(row.filename),
                chunk_id=int(row.id),
                chunk_index=int(row.chunk_index),
                content=str(row.content),
                cosine_similarity=cosine_similarity,
            )
        )

    answer: str
    gen_model: str | None = None
    answer_sources: list[QueryHit] = []
    if not hits:
        answer = (
            "No matching document passages were retrieved; there is no provided text "
            "to answer from."
        )
    else:
        try:
            gen = await generate_answer_groq(q, [h.content for h in hits])
            answer = gen.answer
            gen_model = gen.model
            if gen.citation_passage_numbers:
                picked: list[QueryHit] = []
                for n in gen.citation_passage_numbers:
                    idx = n - 1
                    if 0 <= idx < len(hits):
                        picked.append(hits[idx])
                answer_sources = picked or hits[:1]
            else:
                answer_sources = hits[:1]
        except GenerationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    return QueryResponse(
        question=q,
        top_k=request.top_k,
        document_ids=doc_ids,
        embedding_model=emb.model,
        hits=hits,
        answer=answer,
        answer_sources=answer_sources,
        generation_model=gen_model,
    )
