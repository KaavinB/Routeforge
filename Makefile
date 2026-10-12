# RouteForge developer workflows. Requires uv (https://docs.astral.sh/uv/) and Docker.
SHELL := /bin/bash
export PYTHONPATH := backend:.
export OMP_NUM_THREADS := 1
UV := uv run --no-sync
COLLECTION ?= latest-2026-10-11
REGISTRY ?= config/models/hosted-latest.yaml

.PHONY: help install env up up-obs down logs lint format typecheck test test-integration coverage \
        audit dataset validate-dataset collect score train replay regression-suite live-benchmark load-test fault-test overhead frontend clean

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install:  ## install Python deps (uv) and dashboard deps (npm)
	uv sync
	@chflags nohidden .venv/lib/python3.12/site-packages/*.pth 2>/dev/null || true
	cd frontend && npm ci

env:  ## create/complete .env with generated secrets
	python3 infrastructure/scripts/init_env.py

up: env  ## start gateway, dashboard, postgres, redis
	docker compose up --build -d
	@echo "gateway http://localhost:8000  dashboard http://localhost:8080"

up-obs: env  ## start everything incl. prometheus/grafana/jaeger
	ROUTEFORGE_OTEL_EXPORTER_ENDPOINT=http://jaeger:4318 docker compose --profile observability up --build -d
	@echo "grafana http://localhost:3000  prometheus http://localhost:9090  jaeger http://localhost:16686"

down:  ## stop the stack
	docker compose --profile observability down

logs:
	docker compose logs -f backend

lint:  ## ruff lint + format check
	$(UV) ruff check backend router_training benchmarks
	$(UV) ruff format --check backend router_training benchmarks

format:
	$(UV) ruff format backend router_training benchmarks
	$(UV) ruff check --fix backend router_training benchmarks

typecheck:  ## mypy --strict on the gateway
	$(UV) mypy backend/app

test:  ## unit + API + failure-injection tests (no external services)
	$(UV) pytest

test-integration:  ## tests against real postgres + redis (docker compose up postgres redis)
	$(UV) pytest -m integration

coverage:
	$(UV) pytest --cov=backend/app --cov-report=term-missing --cov-report=xml

audit:  ## dependency vulnerability scan
	$(UV) pip-audit --strict --progress-spinner off -r <(uv export --frozen --no-dev --no-hashes --no-emit-project)

dataset:  ## build routeforge-bench-v1 from public sources + synthetic JSON tasks
	$(UV) python -m router_training.datasets.build --per-category 120

validate-dataset:  ## oracle-check every answer key and test program (docker sandbox)
	$(UV) python -m router_training.datasets.validate

collect:  ## run all prompts on all models of REGISTRY (paid in hosted mode; budget-capped)
	$(UV) python -m router_training.collect --registry $(REGISTRY)

score:  ## label collected outputs (LLM judge for summaries)
	$(UV) python -m router_training.score --collection $(COLLECTION)

train:  ## train the router artifact for COLLECTION
	$(UV) python -m router_training.training.train --collection $(COLLECTION)

replay:  ## Benchmarks A+F: offline policy replay + Pareto frontier
	$(UV) python -m benchmarks.evaluation.offline_replay --collection $(COLLECTION) --registry $(REGISTRY)

regression-suite:  ## Benchmark G: validate the CI quality gate (replay, zero cost)
	$(UV) python -m benchmarks.evaluation.regression_suite --pool latest

live-benchmark:  ## Benchmark A (live): policies end-to-end through a running hosted gateway (PAID)
	$(UV) python -m benchmarks.evaluation.live_policies --base-url http://localhost:8000

load-test:  ## Benchmark C: Locust concurrency sweep against the running gateway
	$(UV) python -m benchmarks.load_tests.run_load --base-url http://localhost:8000

fault-test:  ## Benchmark D: failure injection (gateway must run with fault injection enabled)
	$(UV) python -m benchmarks.fault_injection.run_faults --base-url http://localhost:8000

overhead:  ## Benchmark E: routing overhead (in-process; add BASE_URL for the replay e2e part)
	$(UV) python -m benchmarks.router_overhead --registry config/models/hosted-latest.yaml \
		--artifact router_training/artifacts/router-latest-2026-10-11 $(if $(BASE_URL),--base-url $(BASE_URL),)

frontend:  ## dashboard dev server (http://localhost:5173, proxies to :8000)
	cd frontend && npm run dev

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache coverage.xml htmlcov
