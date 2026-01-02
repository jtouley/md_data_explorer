.PHONY: help install install-dev test test-serial test-unit test-unit-serial test-integration test-integration-serial test-cov test-cov-serial test-cov-term test-cov-term-serial lint format type-check check check-serial clean run run-app run-app-keep validate ensure-venv diff test-analysis test-analysis-serial test-core test-core-serial test-datasets test-datasets-serial test-e2e test-e2e-serial test-loader test-loader-serial test-storage test-storage-serial test-ui test-ui-serial test-fast-serial test-performance test-performance-serial git-log-first git-log-rest

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
	@echo "  make test-core      # Run tests for core module only"
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

test: ensure-venv ## Run all tests in parallel (default)
	@echo "$(GREEN)Running all tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -n auto -m "not serial"

test-serial: ensure-venv ## Run all tests serially (for debugging or deterministic results)
	@echo "$(GREEN)Running all tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v

test-unit: ## Run unit tests only in parallel (default)
	@echo "$(GREEN)Running unit tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not integration and not serial" -n auto

test-unit-serial: ## Run unit tests serially (for debugging)
	@echo "$(GREEN)Running unit tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not integration"

test-integration: ## Run integration tests in parallel (default)
	@echo "$(GREEN)Running integration tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "integration and not serial" -n auto

test-integration-serial: ## Run integration tests serially (for debugging)
	@echo "$(GREEN)Running integration tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "integration"

test-fast: ## Run fast tests (skip slow tests) in parallel (default)
	@echo "$(GREEN)Running fast tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not slow and not serial" -n auto

test-fast-serial: ## Run fast tests serially (for debugging)
	@echo "$(GREEN)Running fast tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not slow"

# Module-specific test commands (parallel by default)
test-analysis: ensure-venv ## Run analysis module tests in parallel (default)
	@echo "$(GREEN)Running analysis module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/analysis -v -n auto -m "not serial"

test-analysis-serial: ensure-venv ## Run analysis module tests serially (for debugging)
	@echo "$(GREEN)Running analysis module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/analysis -v

test-core: ensure-venv ## Run core module tests in parallel (default)
	@echo "$(GREEN)Running core module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/core -v -n auto -m "not serial"

test-core-serial: ensure-venv ## Run core module tests serially (for debugging)
	@echo "$(GREEN)Running core module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/core -v

test-datasets: ensure-venv ## Run datasets module tests in parallel (default)
	@echo "$(GREEN)Running datasets module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/datasets -v -n auto -m "not serial"

test-datasets-serial: ensure-venv ## Run datasets module tests serially (for debugging)
	@echo "$(GREEN)Running datasets module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/datasets -v

test-e2e: ensure-venv ## Run end-to-end tests in parallel (default)
	@echo "$(GREEN)Running end-to-end tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/e2e -v -n auto -m "not serial"

test-e2e-serial: ensure-venv ## Run end-to-end tests serially (for debugging)
	@echo "$(GREEN)Running end-to-end tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/e2e -v

test-loader: ensure-venv ## Run loader module tests in parallel (default)
	@echo "$(GREEN)Running loader module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/loader -v -n auto -m "not serial"

test-loader-serial: ensure-venv ## Run loader module tests serially (for debugging)
	@echo "$(GREEN)Running loader module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/loader -v

test-storage: ensure-venv ## Run storage module tests in parallel (default)
	@echo "$(GREEN)Running storage module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/storage -v -n auto -m "not serial"

test-storage-serial: ensure-venv ## Run storage module tests serially (for debugging)
	@echo "$(GREEN)Running storage module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/storage -v

test-ui: ensure-venv ## Run UI module tests in parallel (default)
	@echo "$(GREEN)Running UI module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/ui -v -n auto -m "not serial"

test-ui-serial: ensure-venv ## Run UI module tests serially (for debugging)
	@echo "$(GREEN)Running UI module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/ui -v

test-performance: ensure-venv ## Run tests with performance tracking in parallel (default)
	@echo "$(GREEN)Running tests with performance tracking in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --track-performance -n auto -m "not serial"

test-performance-serial: ensure-venv ## Run tests with performance tracking serially (for baseline/deterministic results)
	@echo "$(GREEN)Running tests with performance tracking serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --track-performance

performance-report: ## Generate performance report
	@echo "$(GREEN)Generating performance report...$(NC)"
	$(PYTHON_RUN) scripts/generate_performance_report.py --format markdown

performance-update-docs: ## Update PERFORMANCE.md with current benchmarks
	@echo "$(GREEN)Updating PERFORMANCE.md...$(NC)"
	$(PYTHON_RUN) scripts/generate_performance_report.py --update-docs

performance-baseline: ## Create or update performance baseline
	@echo "$(GREEN)Creating performance baseline...$(NC)"
	@if [ ! -f tests/.performance_data.json ]; then \
		echo "$(RED)Error: No performance data found. Run 'make test-performance' first.$(NC)"; \
		exit 1; \
	fi
	$(PYTHON_RUN) scripts/generate_performance_report.py --create-baseline

performance-regression: ensure-venv ## Run performance regression tests
	@echo "$(GREEN)Running performance regression tests...$(NC)"
	$(PYTEST) tests/test_performance_regression.py -v

test-cov: ## Run tests with coverage report in parallel (default)
	@echo "$(GREEN)Running tests with coverage in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -n auto -m "not serial"
	@echo "$(GREEN)Coverage report generated in $(COV_DIR)/index.html$(NC)"

test-cov-serial: ## Run tests with coverage report serially (for deterministic coverage)
	@echo "$(GREEN)Running tests with coverage serially...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in $(COV_DIR)/index.html$(NC)"

test-cov-term: ensure-venv ## Run tests with terminal coverage only in parallel (default)
	@echo "$(GREEN)Running tests with terminal coverage in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing -n auto -m "not serial"

test-cov-term-serial: ensure-venv ## Run tests with terminal coverage serially (for deterministic coverage)
	@echo "$(GREEN)Running tests with terminal coverage serially...$(NC)"
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

check: ## Run all checks (lint, format-check, type-check, test) - tests run in parallel
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
	@echo "$(YELLOW)4. Running tests (parallel)...$(NC)"
	@$(MAKE) test || (echo "$(RED)❌ Tests failed$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)✅ All checks passed!$(NC)"

check-serial: ## Run all checks serially (for deterministic results)
	@echo "$(GREEN)Running all checks serially...$(NC)"
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
	@echo "$(YELLOW)4. Running tests (serial)...$(NC)"
	@$(MAKE) test-serial || (echo "$(RED)❌ Tests failed$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)✅ All checks passed!$(NC)"

check-fast: ## Run fast checks (lint, format-check, fast tests)
	@echo "$(GREEN)Running fast checks...$(NC)"
	@$(MAKE) lint || exit 1
	@$(MAKE) format-check || exit 1
	@$(MAKE) test-fast || exit 1
	@echo "$(GREEN)✅ All fast checks passed!$(NC)"

run: ensure-venv ## Start the Streamlit application (direct, no Ollama management)
	@echo "$(GREEN)Starting Streamlit application...$(NC)"
	$(STREAMLIT) run src/clinical_analytics/ui/app.py

run-app: ## Start application with bash script (stops Ollama on exit)
	@echo "$(GREEN)Starting application with Ollama lifecycle management...$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop (will stop Ollama)$(NC)"
	@STOP_OLLAMA_ON_EXIT=true bash scripts/run_app.sh

run-app-keep: ## Start application with bash script (keep Ollama running on exit)
	@echo "$(GREEN)Starting application (Ollama will keep running)...$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop (Ollama will stay running)$(NC)"
	@STOP_OLLAMA_ON_EXIT=false bash scripts/run_app.sh

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

diff: ## Generate diff files for tracking changes
	@echo "$(GREEN)📝 Generating diff files...$(NC)"
	@mkdir -p .context/diffs
	@echo "  - Unstaged changes (working dir vs staged/HEAD)..."
	@git diff > .context/diffs/unstaged.diff || true
	@echo "  - Staged changes only..."
	@git diff --cached > .context/diffs/staged.diff || true
	@echo "  - Both unstaged + staged (working dir vs HEAD)..."
	@git diff HEAD > .context/diffs/lastcommit.diff || true
	@echo "  - Current branch vs main (committed only)..."
	@git diff main...HEAD > .context/diffs/currentbranch.diff || true
	@echo "  - Current state vs main (including uncommitted)..."
	@git diff main > .context/diffs/current.diff || true
	@echo "$(GREEN)✅ Diff files generated in .context/diffs/$(NC)"
	@echo "   - unstaged.diff: Unstaged changes"
	@echo "   - staged.diff: Staged changes only"
	@echo "   - lastcommit.diff: All uncommitted changes vs HEAD"
	@echo "   - currentbranch.diff: Current branch vs main (committed only)"
	@echo "   - current.diff: Current state vs main (including uncommitted)"

git-log-first: ## Show first 200 lines of commits since main branch
	@git log main..HEAD --format="%h %s%n%b" | head -200

git-log-rest: ## Show commits since main branch (from line 201 onwards)
	@git log main..HEAD --format="%h %s%n%b" | tail -n +201

