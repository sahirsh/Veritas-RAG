FROM python:3.12-slim

# Set to "true" (via docker-compose build arg) to install local embedding/LLM
# extras and bake the sentence-transformers model into the image. Defaults to
# "false" so the production/Render build stays small (no PyTorch).
ARG INSTALL_LOCAL_MODELS=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only in local mode: install heavy extras and pre-download the embedding model
# into a world-readable cache so the app (running as appuser) needs no network
# or write access at runtime.
RUN if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then \
        pip install --no-cache-dir -r requirements-local.txt && \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" && \
        chmod -R a+rX /opt/hf-cache; \
    fi

COPY . .

# Entrypoint starts as root (local Docker) to chown uploads, then drops to appuser.
# On Cloud Run the process may already be non-root; entrypoint handles both.
# Cloud Run sets PORT (usually 8080); listen on that value.
EXPOSE 8080

ENTRYPOINT ["python", "/app/docker_entrypoint.py"]
