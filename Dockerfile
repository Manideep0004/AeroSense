# AaroSense Production Dockerfile (Milestone 5)

# ── Stage 1: Frontend Builder ─────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Requirements Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Install dependencies into a virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Final Production Image ───────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set Python environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p logs/drift_reports logs/drift_alerts mlruns

# Copy the application source code
COPY src/ /app/src/

# Copy the built React frontend
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Note: In production, models/, baselines/, and data/ should be mounted via volumes 
# or pulled from a cloud bucket (S3/GCS) at startup. For this project, we assume 
# they are volume-mounted.

# Expose FastAPI default port
EXPOSE 8000

# Run Uvicorn worker
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
