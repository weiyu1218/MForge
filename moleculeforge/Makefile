.PHONY: install lint test test-unit test-integration test-e2e \
        proto-gen build-images run-dev run-minimal db-migrate clean help

UV ?= uv

help:
	@echo "MoleculeForge Makefile targets:"
	@echo "  install            Install all workspace dependencies"
	@echo "  lint               Run ruff + mypy + import-linter"
	@echo "  test               Run all tests in parallel"
	@echo "  test-unit          Unit tests only"
	@echo "  test-integration   Integration tests (boots docker-compose)"
	@echo "  test-e2e           End-to-end tests"
	@echo "  proto-gen          Regenerate gRPC stubs from protos/"
	@echo "  build-images       Build all Docker base + service images"
	@echo "  run-dev            docker-compose up the dev stack"
	@echo "  run-minimal        Boot minimal demo stack"
	@echo "  db-migrate         Run alembic migrations"
	@echo "  clean              Remove caches and build artefacts"

install:
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy libs/ services/ agents/ models/ || true
	$(UV) run lint-imports || true

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run pytest tests/ -n auto

test-unit:
	$(UV) run pytest tests/unit -n auto -m unit

test-integration:
	docker compose -f infra/docker/docker-compose.test.yml up -d
	$(UV) run pytest tests/integration -m integration; \
	rc=$$?; \
	docker compose -f infra/docker/docker-compose.test.yml down -v; \
	exit $$rc

test-e2e:
	$(UV) run pytest tests/e2e -m e2e

proto-gen:
	@if command -v buf >/dev/null 2>&1; then \
	  echo "Using buf generate…"; cd protos && buf generate; \
	else \
	  echo "buf not found — using grpcio-tools fallback…"; \
	  $(UV) run python tools/dev/generate_protos.py; \
	fi

proto-lint:
	cd protos && buf lint
	cd protos && buf breaking --against '.git#branch=main' || true

build-images:
	bash infra/scripts/build_all_images.sh

run-dev:
	docker compose -f infra/docker/docker-compose.dev.yml up

run-minimal:
	docker compose -f infra/docker/docker-compose.minimal.yml up

db-migrate:
	$(UV) run alembic -c data/alembic/alembic.ini upgrade head

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
	rm -rf dist build *.egg-info
