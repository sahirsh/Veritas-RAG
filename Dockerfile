FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entrypoint starts as root to chown the uploads volume, then drops to appuser.
EXPOSE 8000

ENTRYPOINT ["python", "/app/docker_entrypoint.py"]
