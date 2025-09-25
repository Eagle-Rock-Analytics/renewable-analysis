.PHONY: help install install-dev test test-cov lint format type-check security clean docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install-mamba: ## Install mamba package manager
	@if ! command -v mamba >/dev/null 2>&1; then \
		conda install -y mamba -c conda-forge; \
		eval "$(mamba shell hook --shell ${SHELL##*/})"; \
		mamba activate; \
	else \
		echo "mamba already installed"; \
	fi

install-mamba-env:
	@if ! command -v mamba >/dev/null 2>&1; then \
		echo "mamba is not installed. Please run 'make install-mamba' first."; \
		exit 1; \
	fi
	mamba env create -f environment.yml

install-pkg: ## Install the package
	pip install -e .

install-pkg-dev: ## Install the package with development dependencies
	pip install -e .[dev,test]
	pre-commit install

test: ## Run tests
	python -m pytest

test-cov: ## Run tests with coverage
	python -m pytest --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run all linters
	ruff check src/ tests/ --fix

format: ## Format code
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

security: ## Run security checks
	bandit -r src/
	detect-secrets scan --baseline .secrets.baseline

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

clean: ## Clean up temporary files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -delete

docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

write-env:
	mamba env export --no-builds > environment.yml

setup: install-mamba install-mamba-env
all: clean install-pkg-dev format lint test-cov pre-commit  ## Run full development setup and checks
