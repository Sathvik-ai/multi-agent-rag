FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by PyMuPDF and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install all Python dependencies (no dev dependencies)
RUN uv sync --frozen --no-dev

# Copy the entire application source
COPY . .

# Pre-download the embedding model so first query is fast
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
