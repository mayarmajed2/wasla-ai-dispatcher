# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

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

RUN useradd --create-home --shell /bin/bash wasla && chown -R wasla:wasla /app
USER wasla

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/ready || exit 1

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
