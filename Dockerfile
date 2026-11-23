# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PYTHONPATH=/app:/app/backend \
    PORT=7860

WORKDIR /app

# Install system dependencies for psycopg and audio handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_service/ ./ai_service/
COPY backend/ ./backend/
COPY dashboard/ ./dashboard/
COPY menu/ ./menu/
COPY scripts/ ./scripts/

RUN useradd --create-home --uid 1000 --shell /bin/bash wasla && chown -R wasla:wasla /app
USER wasla

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-7860}/api/ready" || exit 1

CMD ["sh", "-c", "alembic -c backend/alembic.ini upgrade head && uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]
