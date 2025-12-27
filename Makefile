.PHONY: help install install-dev test test-unit test-integration test-cov lint format type-check check clean run validate ensure-venv

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
UV := uv
PYTEST := $(UV) run pytest
RUFF := $(UV) run ruff
MYPY := $(UV) run mypy
STREAMLIT := $(UV) run streamlit
PYTHON_RUN := $(UV) run python

# Check if virtual environment exists
ensure-venv:
	@if [ ! -d ".venv" ]; then \
		echo "$(YELLOW)⚠ Virtual environment not found. Run 'make install-dev' first.$(NC)"; \
		exit 1; \
	fi

# Source and test directories
SRC_DIR := src/clinical_analytics
TEST_DIR := tests
COV_DIR := htmlcov

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Clinical Analytics Platform - Makefile Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install-dev    # Install all dependencies including dev tools"
	@echo "  make check          # Run all checks (lint, type-check, test)"
	@echo "  make test-cov       # Run tests with coverage report"
	@echo "  make run            # Start the Streamlit application"

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(UV) sync --no-dev --no-group dev

install-dev: ## Install all dependencies including dev tools
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	@echo "$(YELLOW)This will install:$(NC)"
	@echo "  • Dev tools: ruff, mypy, pytest, pytest-cov (from optional-dependencies)"
	@echo "  • Docs tools: mkdocs and related packages (from dependency-groups)"
	$(UV) sync --extra dev --group dev

test: ensure-venv ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not integration"

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "integration"

test-fast: ## Run fast tests (skip slow tests)
	@echo "$(GREEN)Running fast tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not slow"

test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in $(COV_DIR)/index.html$(NC)"

test-cov-term: ensure-venv ## Run tests with terminal coverage only
	@echo "$(GREEN)Running tests with terminal coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing

lint: ensure-venv ## Run ruff linter
	@echo "$(GREEN)Running ruff linter...$(NC)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Run ruff linter and auto-fix issues
	@echo "$(GREEN)Running ruff linter with auto-fix...$(NC)"
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR)

format: ensure-venv ## Format code with ruff
	@echo "$(GREEN)Formatting code with ruff...$(NC)"
	$(RUFF) format $(SRC_DIR) $(TEST_DIR)

format-check: ## Check code formatting without making changes
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(RUFF) format --check $(SRC_DIR) $(TEST_DIR)

type-check: ensure-venv ## Run mypy type checker
	@echo "$(GREEN)Running mypy type checker...$(NC)"
	$(MYPY) $(SRC_DIR)

type-check-strict: ## Run mypy in strict mode
	@echo "$(GREEN)Running mypy in strict mode...$(NC)"
	$(MYPY) --strict $(SRC_DIR)

check: ## Run all checks (lint, format-check, type-check, test)
	@echo "$(GREEN)Running all checks...$(NC)"
	@echo ""
	@echo "$(YELLOW)1. Linting...$(NC)"
	@$(MAKE) lint || (echo "$(RED)❌ Linting failed$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)2. Format check...$(NC)"
	@$(MAKE) format-check || (echo "$(RED)❌ Format check failed$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)3. Type checking...$(NC)"
	@$(MAKE) type-check || (echo "$(RED)❌ Type checking failed$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)4. Running tests...$(NC)"
	@$(MAKE) test || (echo "$(RED)❌ Tests failed$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)✅ All checks passed!$(NC)"

check-fast: ## Run fast checks (lint, format-check, fast tests)
	@echo "$(GREEN)Running fast checks...$(NC)"
	@$(MAKE) lint || exit 1
	@$(MAKE) format-check || exit 1
	@$(MAKE) test-fast || exit 1
	@echo "$(GREEN)✅ All fast checks passed!$(NC)"

run: ensure-venv ## Start the Streamlit application
	@echo "$(GREEN)Starting Streamlit application...$(NC)"
	$(STREAMLIT) run src/clinical_analytics/ui/app.py

validate: ## Run platform validation script
	@echo "$(GREEN)Running platform validation...$(NC)"
	$(PYTHON_RUN) scripts/validate_platform.py

clean: ## Clean generated files and caches
	@echo "$(GREEN)Cleaning generated files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf $(COV_DIR)
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	@echo "$(GREEN)✓ Clean complete$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(GREEN)Cleaning virtual environment...$(NC)"
	rm -rf .venv
	@echo "$(GREEN)✓ Full clean complete$(NC)"

ci: ## Run CI checks (for GitHub Actions)
	@echo "$(GREEN)Running CI checks...$(NC)"
	@$(MAKE) lint
	@$(MAKE) format-check
	@$(MAKE) type-check
	@$(MAKE) test-cov-term

