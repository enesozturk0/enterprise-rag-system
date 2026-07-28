.DEFAULT_GOAL := help

PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: help install-dev lint format type-check test test-all test-cov quality \
	docker-config docker-build docker-up docker-down docker-ps docker-logs

help:
	@echo "Available commands:"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make lint           Run Ruff checks"
	@echo "  make format         Format Python files"
	@echo "  make type-check     Run MyPy"
	@echo "  make test           Run unit tests"
	@echo "  make test-all       Run unit and integration tests"
	@echo "  make test-cov       Run unit tests with coverage gate"
	@echo "  make quality        Run all local quality checks"
	@echo "  make docker-config  Validate Compose configuration"
	@echo "  make docker-build   Build application images"
	@echo "  make docker-up      Start the complete stack"
	@echo "  make docker-down    Stop containers without deleting volumes"
	@echo "  make docker-ps      Show container status"
	@echo "  make docker-logs    Follow service logs"

install-dev:
	$(PYTHON) -m pip install --editable ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format:
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m ruff format src tests scripts

type-check:
	$(PYTHON) -m mypy src --ignore-missing-imports

test:
	$(PYTHON) -m pytest tests --ignore=tests/integration

test-all:
	$(PYTHON) -m pytest tests

test-cov:
	$(PYTHON) -m pytest tests \
		--ignore=tests/integration \
		--cov=src \
		--cov-report=term-missing \
		--cov-fail-under=95

quality: lint type-check test-cov

docker-config:
	$(COMPOSE) config --quiet

docker-build:
	$(COMPOSE) build api ui

docker-up:
	$(COMPOSE) up --detach --build

docker-down:
	$(COMPOSE) down

docker-ps:
	$(COMPOSE) ps

docker-logs:
	$(COMPOSE) logs --follow
