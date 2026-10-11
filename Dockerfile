# RouteForge gateway image (API + migrations). Multi-stage: dependencies are resolved from
# the committed uv.lock in a builder stage; the runtime stage is slim and runs as non-root.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group bench --no-install-project

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/backend:/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    ROUTEFORGE_EVAL_SANDBOX_BACKEND=subprocess
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY backend/app backend/app
COPY backend/alembic.ini backend/alembic.ini
COPY backend/migrations backend/migrations
COPY config config
COPY router_training/artifacts router_training/artifacts
COPY router_training/datasets/data router_training/datasets/data
RUN mkdir -p benchmarks/results router_training/data && chown -R app:app benchmarks router_training/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1} --no-access-log --timeout-graceful-shutdown 20"]
