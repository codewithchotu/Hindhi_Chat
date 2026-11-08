# Multi-stage production Dockerfile for Cross-Lingual Hindi RAG FastAPI Backend
FROM python:3.11-slim AS base

# System environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models_cache \
    TORCH_HOME=/models_cache

WORKDIR /app

# Install minimal build tools and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root system user and prepare directories with full permissions
RUN groupadd -g 1000 raggroup && \
    useradd -u 1000 -g raggroup -m -s /bin/bash raguser && \
    mkdir -p /models_cache /app/storage/chroma_db /app/data && \
    chown -R raguser:raggroup /models_cache /app

# Copy application source code with non-root ownership
COPY --chown=raguser:raggroup . .

# Switch to non-root user
USER raguser

# Expose FastAPI port
EXPOSE 8000

# Health check with 45s start period probing dual healthcheck endpoints
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || curl -f http://localhost:8000/health || exit 1

# Launch FastAPI via Uvicorn production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
