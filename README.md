# Veritas-RAG

A production-grade Retrieval-Augmented Generation (RAG) API that lets you upload PDF documents and ask natural-language questions about their contents. Answers are grounded in the most semantically relevant passages from your documents and returned with source citations.

**Live demo:** [veritas-rag-frontend.onrender.com](https://veritas-rag-frontend.onrender.com)

> The demo runs on free-tier infrastructure — the first request after a period of inactivity may take up to 60 seconds to wake up. Subsequent requests are fast.

---

## How it works

1. **Upload** — a PDF is text-extracted, split into 500-word overlapping chunks, and embedded into 768-dimensional vectors via the Gemini Embeddings API. Vectors are stored in PostgreSQL with the pgvector extension.
2. **Query** — a natural-language question is embedded using the same model, a cosine similarity search retrieves the most relevant chunks, and Groq (llama-3.1-8b-instant) generates a grounded answer with citations pointing back to the source passages.

**Stack:** FastAPI · PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) · Gemini `gemini-embedding-001` · Groq `llama-3.1-8b-instant` · Streamlit · Docker · Neon · Render

---

## Features

- Semantic search over uploaded PDFs using dense vector embeddings
- Grounded answer generation — responses cite specific passage numbers, no hallucinated facts
- Async FastAPI backend with SQLAlchemy 2.0 and psycopg3
- Automatic schema migrations via Alembic on container start
- Streamlit chat UI with dark theme, session memory, and collapsible citation cards
- Health endpoint reporting status of database, embedding, and generation services
- Docker Compose setup for fully local development

---

## Quick start (Docker)

```bash
cp .env.example .env
# Set GEMINI_API_KEY and GROQ_API_KEY in .env
docker compose up --build
```

The API will be available at `http://localhost:8000`. Run the frontend separately:

```bash
pip install -r requirements-frontend.txt
streamlit run app.py
```

Open `http://localhost:8501` to use the chat interface.

Verify the backend is ready at `http://localhost:8000/health` — expect `database: ready`, `gemini: configured`, and `groq: configured`.

---

## Local development (without Docker)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements-dev.txt

set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/veritas?sslmode=disable
set GEMINI_API_KEY=...
set GROQ_API_KEY=gsk_...

alembic upgrade head
uvicorn main:app --reload
```

Postgres with the pgvector extension must be running. The included `docker-compose.yml` can provide this independently if needed.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service name and version |
| GET | `/health` | Database, Gemini, and Groq status |
| GET | `/documents` | List indexed documents (`limit`, `offset`) |
| GET | `/documents/{id}` | Document metadata and chunk count |
| DELETE | `/documents/{id}` | Delete document, embeddings, and source file |
| POST | `/documents/upload` | Upload and index a PDF (multipart `file`) |
| POST | `/query` | RAG query — returns answer with citations |

Interactive API docs available at `/docs` when running locally.

### Query request body

```json
{
  "question": "What is the main topic?",
  "top_k": 5,
  "document_ids": [1, 2]
}
```

`document_ids` is optional. When provided, retrieval is scoped to those documents only.

---

## Deployment

The backend is containerized via the included `Dockerfile` and deploys on any Docker-capable platform (Render, Fly.io, Railway, Cloud Run, etc.). It requires a managed PostgreSQL instance with the `pgvector` extension enabled.

The Streamlit frontend (`app.py`) is a standalone Python app deployable on any Python host. It reads the backend URL from the `VERITAS_API_URL` environment variable, falling back to `http://localhost:8000`. A lean `requirements-frontend.txt` is provided so the frontend host installs only what it needs.

A `render.yaml` Blueprint is included for one-click backend deployment on Render.

---

## Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DATABASE_URL` | Yes* | — | PostgreSQL connection string |
| `GEMINI_API_KEY` | Yes** | — | Embedding model API key |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Embedding model name |
| `GROQ_API_KEY` | Yes*** | — | Chat inference API key |
| `GROQ_CHAT_MODEL` | No | `llama-3.1-8b-instant` | Generation model name |
| `UPLOAD_DIR` | No | `uploads` | PDF storage directory |
| `CORS_ORIGINS` | No | `*` | Comma-separated allowed origins |
| `PORT` | No | `8000` | API port (read by Docker entrypoint) |
| `VERITAS_API_URL` | No | `http://localhost:8000` | Backend URL (frontend only) |

\* Upload and query return 503 without a configured database.  
\** Embedding steps return 503 without a Gemini key.  
\*** Answer generation returns 503 without a Groq key.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Unit tests cover PDF parsing, chunking, embedding result parsing, and generation response parsing. They run without requiring a live database, Gemini, or Groq.

---

## Migrations

```bash
alembic upgrade head
```

In Docker, `docker_entrypoint.py` runs this automatically before Uvicorn starts.

---

## Roadmap

- **User authentication** — per-user document isolation with scoped retrieval and JWT-based access control
- **Multimodal retrieval** — CLIP-based image embeddings for diagrams and figures extracted from PDFs
- **Inline document viewer** — render PDF pages alongside chat with page-level citation linking
- **Improved vector search** — HNSW parameter tuning and hybrid sparse-dense retrieval
