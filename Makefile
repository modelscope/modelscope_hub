.DEFAULT_GOAL := help

.PHONY: help install lint typecheck test test-all test-remote clean build publish

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode with dev deps
	pip install -e ".[dev]"

lint: ## Run ruff linter and formatter check
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code with ruff
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck: ## Run mypy type checking
	mypy src/modelscope_hub/

test: ## Run unit tests only (no remote API calls)
	pytest tests/ -k "not remote" --ignore=tests/integration

test-all: ## Run all tests including integration (needs tests/.env)
	pytest tests/

test-remote: ## Run only remote integration tests
	pytest tests/ -m remote

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build: clean ## Build source and wheel distributions
	python -m build

publish: build ## Upload to PyPI (requires twine)
	twine upload dist/*
