.PHONY: help install install-dev install-pre-commit test test-serial test-unit test-unit-serial test-integration test-integration-serial test-cov test-cov-serial test-cov-term test-cov-term-serial test-cov-check test-cov-diff coverage-baseline coverage-report lint format type-check check check-serial clean run run-app run-app-keep run-api validate ensure-venv diff test-analysis test-analysis-serial test-core test-core-serial test-datasets test-datasets-serial test-e2e test-e2e-serial electron-npm-ready test-electron-e2e test-electron-quality test-loader test-loader-serial test-storage test-storage-serial test-ui test-ui-serial test-fast-serial test-performance test-performance-serial git-log-first git-log-rest git-log-export git-log-latest git-log-recent checkpoint-create checkpoint-resume sync-cursor-skills sync-cursor-skills-force cursor-packaged-skills benchmark-cursor-skills

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
UV := uv
PYTEST := $(UV) run pytest
PYTEST_ARGS ?=
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
	@echo "  make test-ui PYTEST_ARGS='-k mytest -x'  # Extra pytest flags (optional)"
	@echo "  make test-electron-e2e  # Playwright (electron/); needs: cd electron && npm ci"
	@echo "  make test-electron-quality  # Electron unit coverage + Playwright E2E"

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(UV) sync --no-dev --no-group dev

install-dev: ## Install all dependencies including dev tools
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	@echo "$(YELLOW)This will install:$(NC)"
	@echo "  • Dev tools: ruff, mypy, pytest, pytest-cov (from optional-dependencies)"
	@echo "  • Docs tools: mkdocs and related packages (from dependency-groups)"
	$(UV) sync --extra dev --group dev

install-pre-commit: ensure-venv ## Install pre-commit hooks (run after install-dev)
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	$(PYTHON_RUN) -m pip install pre-commit
	$(PYTHON_RUN) -m pre_commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"
	@echo "$(YELLOW)Note: Hooks will run automatically on git commit$(NC)"
	@echo "$(YELLOW)Run 'pre-commit run --all-files' to check all files now$(NC)"

test: ensure-venv ## Run all tests in parallel (default)
	@echo "$(GREEN)Running all tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -n auto -m "not serial" $(PYTEST_ARGS)

test-serial: ensure-venv ## Run all tests serially (for debugging or deterministic results)
	@echo "$(GREEN)Running all tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v $(PYTEST_ARGS)

test-unit: ## Run unit tests only in parallel (default)
	@echo "$(GREEN)Running unit tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not integration and not serial" -n auto $(PYTEST_ARGS)

test-unit-serial: ## Run unit tests serially (for debugging)
	@echo "$(GREEN)Running unit tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not integration" $(PYTEST_ARGS)

test-integration: ## Run integration tests in parallel (default)
	@echo "$(GREEN)Running integration tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "integration and not serial" -n auto $(PYTEST_ARGS)

test-integration-serial: ## Run integration tests serially (for debugging)
	@echo "$(GREEN)Running integration tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "integration" $(PYTEST_ARGS)

test-fast: ## Run fast tests (skip slow tests) in parallel (default)
	@echo "$(GREEN)Running fast tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not slow and not serial" -n auto $(PYTEST_ARGS)

test-fast-serial: ## Run fast tests serially (for debugging)
	@echo "$(GREEN)Running fast tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -m "not slow" $(PYTEST_ARGS)

# Module-specific test commands (parallel by default)
test-analysis: ensure-venv ## Run analysis module tests in parallel (default)
	@echo "$(GREEN)Running analysis module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/analysis -v -n auto -m "not serial" $(PYTEST_ARGS)

test-analysis-serial: ensure-venv ## Run analysis module tests serially (for debugging)
	@echo "$(GREEN)Running analysis module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/analysis -v $(PYTEST_ARGS)

test-core: ensure-venv ## Run core module tests in parallel (default)
	@echo "$(GREEN)Running core module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/core -v -n auto -m "not serial" $(PYTEST_ARGS)

test-core-serial: ensure-venv ## Run core module tests serially (for debugging)
	@echo "$(GREEN)Running core module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/core -v $(PYTEST_ARGS)

test-datasets: ensure-venv ## Run datasets module tests in parallel (default)
	@echo "$(GREEN)Running datasets module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/datasets -v -n auto -m "not serial" $(PYTEST_ARGS)

test-datasets-serial: ensure-venv ## Run datasets module tests serially (for debugging)
	@echo "$(GREEN)Running datasets module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/datasets -v $(PYTEST_ARGS)

test-e2e: ensure-venv ## Run end-to-end tests in parallel (default)
	@echo "$(GREEN)Running end-to-end tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/e2e -v -n auto -m "not serial" $(PYTEST_ARGS)

test-e2e-serial: ensure-venv ## Run end-to-end tests serially (for debugging)
	@echo "$(GREEN)Running end-to-end tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/e2e -v $(PYTEST_ARGS)

electron-npm-ready:
	@if [ ! -f electron/package.json ]; then \
		echo "$(RED)electron/package.json not found$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d electron/node_modules ]; then \
		echo "$(YELLOW)electron/node_modules missing. Run: cd electron && npm ci$(NC)"; \
		exit 1; \
	fi

test-electron-e2e: electron-npm-ready ## Playwright against Electron Vite renderer (Chromium); not the full Electron binary
	@echo "$(GREEN)Running Electron renderer Playwright suite...$(NC)"
	cd electron && npm run test:e2e

test-electron-quality: electron-npm-ready ## Electron quality gate: Vitest coverage threshold + Playwright E2E
	@echo "$(GREEN)Running Electron JS quality gate (coverage + E2E)...$(NC)"
	cd electron && npm run test:quality

test-loader: ensure-venv ## Run loader module tests in parallel (default)
	@echo "$(GREEN)Running loader module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/loader -v -n auto -m "not serial" $(PYTEST_ARGS)

test-loader-serial: ensure-venv ## Run loader module tests serially (for debugging)
	@echo "$(GREEN)Running loader module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/loader -v $(PYTEST_ARGS)

test-storage: ensure-venv ## Run storage module tests in parallel (default)
	@echo "$(GREEN)Running storage module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/storage -v -n auto -m "not serial" $(PYTEST_ARGS)

test-storage-serial: ensure-venv ## Run storage module tests serially (for debugging)
	@echo "$(GREEN)Running storage module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/storage -v $(PYTEST_ARGS)

test-ui: ensure-venv ## Run UI module tests in parallel (default)
	@echo "$(GREEN)Running UI module tests in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR)/ui -v -n auto -m "not serial" $(PYTEST_ARGS)

test-ui-serial: ensure-venv ## Run UI module tests serially (for debugging)
	@echo "$(GREEN)Running UI module tests serially...$(NC)"
	$(PYTEST) $(TEST_DIR)/ui -v $(PYTEST_ARGS)

test-api: ensure-venv ## Run all API tests (unit + integration)
	@echo "$(GREEN)Running all API tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/api -v --tb=short $(PYTEST_ARGS)

test-api-integration: ensure-venv ## Run API integration tests (with real server)
	@echo "$(GREEN)Running API integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/api/integration -v --tb=short -m integration $(PYTEST_ARGS)

test-performance: ensure-venv ## Run tests with performance tracking in parallel (default)
	@echo "$(GREEN)Running tests with performance tracking in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --track-performance -n auto -m "not serial" $(PYTEST_ARGS)

test-performance-serial: ensure-venv ## Run tests with performance tracking serially (for baseline/deterministic results)
	@echo "$(GREEN)Running tests with performance tracking serially...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --track-performance $(PYTEST_ARGS)

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
	$(PYTEST) tests/test_performance_regression.py -v $(PYTEST_ARGS)

test-cov: ## Run tests with coverage report in parallel (default)
	@echo "$(GREEN)Running tests with coverage in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -n auto -m "not serial" $(PYTEST_ARGS)
	@echo "$(GREEN)Coverage report generated in $(COV_DIR)/index.html$(NC)"

test-cov-serial: ## Run tests with coverage report serially (for deterministic coverage)
	@echo "$(GREEN)Running tests with coverage serially...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing $(PYTEST_ARGS)
	@echo "$(GREEN)Coverage report generated in $(COV_DIR)/index.html$(NC)"

test-cov-term: ensure-venv ## Run tests with terminal coverage only in parallel (default)
	@echo "$(GREEN)Running tests with terminal coverage in parallel...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing -n auto -m "not serial" $(PYTEST_ARGS)

test-cov-term-serial: ensure-venv ## Run tests with terminal coverage serially (for deterministic coverage)
	@echo "$(GREEN)Running tests with terminal coverage serially...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing $(PYTEST_ARGS)

# Coverage enforcement commands (NEW)
test-cov-check: ensure-venv ## Run tests with coverage enforcement (67% minimum, target 95%)
	@echo "$(GREEN)Running tests with coverage enforcement (67% minimum)...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-fail-under=67 -n auto -m "not serial" $(PYTEST_ARGS)

test-cov-diff: ensure-venv ## Check coverage diff against baseline
	@echo "$(GREEN)Checking coverage against baseline...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=json:coverage.json -n auto -m "not serial" $(PYTEST_ARGS)
	@$(PYTHON_RUN) scripts/check_coverage_regression.py

coverage-baseline: ensure-venv ## Generate coverage baseline
	@echo "$(GREEN)Generating coverage baseline...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=json:coverage.json -n auto -m "not serial" $(PYTEST_ARGS)
	@cp coverage.json tests/.coverage_baseline.json
	@echo "$(GREEN)✓ Baseline saved to tests/.coverage_baseline.json$(NC)"

coverage-report: ensure-venv ## Generate detailed coverage HTML report
	@echo "$(GREEN)Generating coverage report...$(NC)"
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -n auto $(PYTEST_ARGS)
	@echo "$(GREEN)Coverage report generated in $(COV_DIR)/index.html$(NC)"

lint: ensure-venv ## Run ruff linter
	@echo "$(GREEN)Running ruff linter...$(NC)"
	$(RUFF) check $(SRC_DIR) $(TEST_DIR) repo_context_mcp

lint-fix: ## Run ruff linter and auto-fix issues
	@echo "$(GREEN)Running ruff linter with auto-fix...$(NC)"
	$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR) repo_context_mcp

format: ensure-venv ## Format code with ruff
	@echo "$(GREEN)Formatting code with ruff...$(NC)"
	$(RUFF) format $(SRC_DIR) $(TEST_DIR) repo_context_mcp

format-check: ## Check code formatting without making changes
	@echo "$(GREEN)Checking code formatting...$(NC)"
	$(RUFF) format --check $(SRC_DIR) $(TEST_DIR) repo_context_mcp

pre-commit-check: ensure-venv ## Run pre-commit checks (test fixture enforcement)
	@echo "$(GREEN)Running pre-commit checks...$(NC)"
	@$(PYTHON_RUN) scripts/check_test_fixtures.py $$(find $(TEST_DIR) -name "test_*.py" -type f) || (echo "$(RED)❌ Pre-commit checks failed$(NC)" && exit 1)
	@echo "$(GREEN)✓ Pre-commit checks passed$(NC)"

type-check: ensure-venv ## Run mypy type checker (matches pre-commit config)
	@echo "$(GREEN)Running mypy type checker...$(NC)"
	$(MYPY) --ignore-missing-imports $(SRC_DIR)

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

check-fast: ## Run fast code quality checks (lint, format-check) - no tests
	@echo "$(GREEN)Running fast code quality checks...$(NC)"
	@$(MAKE) lint || exit 1
	@$(MAKE) format-check || exit 1
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

run-api: ensure-venv ## Start FastAPI backend (for Electron desktop dev)
	@echo "$(GREEN)Starting FastAPI on http://127.0.0.1:8000 ...$(NC)"
	$(UV) run uvicorn clinical_analytics.api.main:app --reload --host 127.0.0.1 --port 8000

validate: ## Run platform validation (tests serve as validation)
	@echo "$(GREEN)Platform validation via test suite...$(NC)"
	@echo "Built-in dataset validation script removed - use 'make test' or 'make check' for validation"
	@$(MAKE) test-fast

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

git-log-export: ## Export full commit history since main branch to .context/commits/
	@echo "$(GREEN)📝 Exporting commit history...$(NC)"
	@mkdir -p .context/commits
	@BRANCH_NAME=$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"); \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	OUTPUT_FILE=".context/commits/$${BRANCH_NAME}_$${TIMESTAMP}.md"; \
	echo "# Commit History: $${BRANCH_NAME}" > "$$OUTPUT_FILE"; \
	echo "" >> "$$OUTPUT_FILE"; \
	echo "**Branch**: \`$${BRANCH_NAME}\`" >> "$$OUTPUT_FILE"; \
	echo "**Exported**: $$(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$$OUTPUT_FILE"; \
	echo "" >> "$$OUTPUT_FILE"; \
	echo "## Commits Since Main" >> "$$OUTPUT_FILE"; \
	echo "" >> "$$OUTPUT_FILE"; \
	git log main..HEAD --format="### %h - %s%n%n**Date**: %ai%n**Author**: %an%n%n%b%n---" >> "$$OUTPUT_FILE" 2>/dev/null || echo "No commits since main branch." >> "$$OUTPUT_FILE"; \
	echo "$(GREEN)✅ Commit history exported to $${OUTPUT_FILE}$(NC)"

git-log-latest: ## Show latest commit history export
	@LATEST=$$(ls -t .context/commits/*.md 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "$(YELLOW)⚠ No commit history exports found. Run 'make git-log-export' first.$(NC)"; \
	else \
		echo "$(GREEN)Latest commit history: $${LATEST}$(NC)"; \
		head -50 "$$LATEST"; \
	fi

git-log-recent: ## Show last N commits (default: 10, override with N=20)
	@N=$${N:-10}; \
	echo "$(GREEN)Last $${N} commits:$(NC)"; \
	git log --oneline -$${N}

checkpoint-create: ## Create lightweight checkpoint template (requires TASK_ID="task_name")
	@if [ -z "$(TASK_ID)" ]; then \
		echo "$(RED)❌ Error: TASK_ID required. Usage: make checkpoint-create TASK_ID=\"task_name\"$(NC)"; \
		exit 1; \
	fi
	@mkdir -p .context/checkpoints
	@LAST_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo "none"); \
	FILE=".context/checkpoints/$(TASK_ID).md"; \
	echo "# $(TASK_ID)" > "$$FILE"; \
	echo "" >> "$$FILE"; \
	echo "**Status**: In progress (since last commit: $${LAST_COMMIT})" >> "$$FILE"; \
	echo "" >> "$$FILE"; \
	echo "**What I did since last commit**:\n- " >> "$$FILE"; \
	echo "" >> "$$FILE"; \
	echo "**Current state**:\n- " >> "$$FILE"; \
	echo "" >> "$$FILE"; \
	echo "**Next steps**:\n1. " >> "$$FILE"; \
	echo "" >> "$$FILE"; \
	echo "**Blockers/Notes**:\n- " >> "$$FILE"; \
	echo "$(GREEN)✅ Checkpoint created: $$FILE$(NC)"; \
	echo "$(YELLOW)💡 Edit it manually with what happened in this chat session$(NC)"

checkpoint-resume: ## Show checkpoint for resuming work (requires TASK_ID)
	@if [ -z "$(TASK_ID)" ]; then \
		echo "$(RED)❌ Error: TASK_ID required$(NC)"; \
		exit 1; \
	fi
	@FILE=".context/checkpoints/$(TASK_ID).md"; \
	if [ ! -f "$$FILE" ]; then \
		echo "$(RED)❌ Checkpoint not found: $$FILE$(NC)"; \
		exit 1; \
	fi; \
	cat "$$FILE"

sync-cursor-skills: ## Write mdde-* + mdde-context + mcp-workbench to ~/.cursor/skills/ (does not sync packaged subfolders)
	@echo "$(GREEN)Syncing Cursor skills to ~/.cursor/skills/ ...$(NC)"
	$(PYTHON_RUN) scripts/sync_volt_cursor_skills.py

sync-cursor-skills-force: ## Same as sync-cursor-skills but overwrites global even when global copy is newer
	@echo "$(GREEN)Syncing Cursor skills (force) to ~/.cursor/skills/ ...$(NC)"
	$(PYTHON_RUN) scripts/sync_volt_cursor_skills.py --force

# Default: diff repo .cursor/skills/<pkg>/ vs ~/.cursor/skills/. Override, e.g.:
#   make cursor-packaged-skills CURSOR_PACKAGED_ARGS="--pull-packaged-from-global"
#   make cursor-packaged-skills CURSOR_PACKAGED_ARGS="--push-packaged-to-global"
#   make cursor-packaged-skills CURSOR_PACKAGED_ARGS="--promote-packaged-to-global"
CURSOR_PACKAGED_ARGS ?= --diff-packaged
cursor-packaged-skills: ## Packaged skills vs global (~/.cursor/skills/); see CURSOR_PACKAGED_ARGS above
	$(PYTHON_RUN) scripts/sync_volt_cursor_skills.py $(CURSOR_PACKAGED_ARGS)

benchmark-cursor-skills: ## Validate every global Cursor skill (skill-creator-style YAML gate)
	@echo "$(GREEN)Benchmarking ~/.cursor/skills/ (validation only)...$(NC)"
	$(PYTHON_RUN) scripts/benchmark_cursor_skills_packaging.py
