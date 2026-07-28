# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser

WORKDIR /app

COPY docker/ui-requirements.txt /tmp/ui-requirements.txt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install \
        --no-cache-dir \
        --requirement /tmp/ui-requirements.txt \
    && rm /tmp/ui-requirements.txt

# UI source and the Pydantic contracts consumed by its API client.
COPY src/__init__.py ./src/
COPY src/ui/ ./src/ui/
COPY src/api/__init__.py ./src/api/
COPY src/api/schemas/ ./src/api/schemas/
COPY .streamlit/ ./.streamlit/

RUN groupadd --gid 10001 appgroup \
    && useradd \
        --uid 10001 \
        --gid appgroup \
        --create-home \
        --home-dir /home/appuser \
        --shell /usr/sbin/nologin \
        appuser \
    && chown -R 10001:10001 /app /home/appuser

# Allows absolute imports such as `src.ui.api_client`.
ENV PYTHONPATH=/app

USER 10001:10001

EXPOSE 8501

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD ["python", "-c", "import src.ui.app; import httpx; httpx.get('http://127.0.0.1:8501/_stcore/health', timeout=4).raise_for_status()"]

CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
