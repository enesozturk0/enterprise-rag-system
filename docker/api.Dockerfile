# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: Build the application and dependency wheels
# ============================================================
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Native build tools remain only in the builder stage.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# Build the project and all dependencies as reusable wheels.
RUN python -m pip wheel \
    --wheel-dir=/wheels \
    .


# ============================================================
# Stage 2: Minimal production runtime
# ============================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 is required by native numerical dependencies.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels

# Install only artifacts produced by the trusted builder stage.
RUN python -m pip install \
        --no-cache-dir \
        --no-index \
        --find-links=/wheels \
        enterprise-rag-system \
    && rm -rf /wheels

COPY docker/gunicorn.conf.py /app/gunicorn.conf.py

# Fixed UID/GID makes mounted-volume permissions predictable.
RUN groupadd --gid 10001 appgroup \
    && useradd \
        --uid 10001 \
        --gid appgroup \
        --no-create-home \
        --shell /usr/sbin/nologin \
        appuser \
    && mkdir -p /app/data/chroma_db \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD ["python", "-c", "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=4).raise_for_status()"]

CMD ["gunicorn", "src.api.main:app", "--config", "/app/gunicorn.conf.py"]
