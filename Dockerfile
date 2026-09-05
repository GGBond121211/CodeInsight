# syntax=docker/dockerfile:1.7

FROM python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 AS builder

ENV UV_VERSION=0.11.32 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"
COPY backend/pyproject.toml backend/uv.lock ./backend/
COPY backend/src ./backend/src
RUN cd backend && uv sync --locked --no-dev

FROM python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend/src \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --from=builder /build/backend/.venv /opt/venv
COPY backend/src ./backend/src
RUN groupadd --system --gid 10001 codeinsight \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent \
        --shell /usr/sbin/nologin codeinsight \
    && mkdir -p /app/data \
    && chown -R codeinsight:codeinsight /app /opt/venv

USER 10001:10001
EXPOSE 8000 8010
CMD ["python", "-m", "uvicorn", "codeinsight.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
