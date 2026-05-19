# ─── Stage 1: Base image ───────────────────────────────────────────
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies needed by PyMuPDF and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── Stage 2: Install dependencies via uv ──────────────────────────
FROM base AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install all Python dependencies (no dev dependencies)
RUN uv sync --frozen --no-dev

# ─── Stage 3: Final image ──────────────────────────────────────────
FROM base AS final

# Copy the venv from builder stage
COPY --from=builder /app/.venv /app/.venv

# Make sure we use the venv's Python/binaries
ENV PATH="/app/.venv/bin:$PATH"

# Copy the entire application source
COPY . .

# Pre-download the embedding model so first query is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
