"""Gunicorn configuration for the FastAPI production container."""

from __future__ import annotations

import os


def _positive_integer(name: str, default: int) -> int:
    """Read and validate a positive integer environment value."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")

    return value


# Worker lifecycle
workers = _positive_integer("GUNICORN_WORKERS", 2)
worker_class = "uvicorn_worker.UvicornWorker"

# Container içindeki port sabittir; host portunu Compose yönetecek.
control_socket_disable = True

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# LLM generation normal HTTP isteklerinden daha uzun sürebilir.
timeout = _positive_integer("GUNICORN_TIMEOUT", 120)
graceful_timeout = _positive_integer(
    "GUNICORN_GRACEFUL_TIMEOUT",
    30,
)
keepalive = _positive_integer("GUNICORN_KEEPALIVE", 5)

# Worker heartbeat geçici dosyalarını memory filesystem'de tutar.
worker_tmp_dir = "/dev/shm"

# Container logları stdout/stderr üzerinden toplanır.
accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = os.getenv("LOG_LEVEL", "info").lower()

# Temel HTTP request sınırları.
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190
