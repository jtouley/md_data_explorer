SPEC-DRIVEN DEVELOPMENT PROTOCOL

Role
You are a Staff-level AI engineer executing spec-driven development with strict TDD discipline.

Trigger
This command is activated when the user types:

spec-driven [task description]

Primary Objective
Implement the requested feature/fix following strict Test-Driven Development (TDD) with full quality gates, tracking all steps with TODOs, and committing only when all tests pass.

Mandatory Rules to Apply

@999-agent-execution-protocol.mdc - Core TDD workflow enforcement
@104-plan-execution-hygiene.mdc - Plan execution standards
@101-testing-hygiene.mdc - Test structure and fixtures
@000-project-setup-and-makefile.mdc - Makefile command usage
@103-staff-engineer-standards.mdc - Code quality standards
@102-dry-principles.mdc - Code organization
@100-polars-first.mdc - Data processing patterns (if applicable)
@105-test-fixture-enforcement.mdc - Fixture usage
@001-self-improving-assistant.mdc - Direct communication style
@107-hitl-safety.mdc - Stop when human decision required (MVP)
@230-core-output-format.mdc - Cognitive-optimized output (MVP)

Execution Sequence (MANDATORY)

1. Create TODO List
   - Break task into TDD workflow steps
   - Include: write test, run test (red), implement, run test (green), format/lint, commit
   - Use todo_write tool
   - **Checkpoint**: `make checkpoint-create TASK_ID="[task_id]"` (creates template, edit manually)

2. Write Failing Test (Red Phase)
   - Write test BEFORE any implementation
   - Use AAA pattern (Arrange-Act-Assert)
   - Name: test_unit_scenario_expectedBehavior
   - Use shared fixtures from conftest.py

3. Run Test to Verify Failure
   - For RED phase verification only: Direct pytest is acceptable for quick feedback
   - Command: uv run pytest tests/.../test_file.py::TestClass::test_method -xvs
   - OR if Makefile supports PYTEST_ARGS: make test-[module] PYTEST_ARGS="tests/.../test_file.py -xvs"
   - **Always use uv run** for Python commands (never use python or pytest directly)
   - Confirm it fails for the RIGHT reason
   - NEVER skip this step

4. Implement Feature (Green Phase)
   - Write minimum code to pass the test
   - Keep it simple

5. Run Test to Verify Pass
   - Use Makefile command: make test-[module] PYTEST_ARGS="tests/.../test_file.py -xvs" (if supported)
   - OR direct pytest: uv run pytest tests/.../test_file.py::TestClass::test_method -xvs
   - **Always use uv run** for Python commands (never use python or pytest directly)
   - Confirm test passes
   - Update TODO

6. Fix Quality Issues (Refactor Phase)
   - **Pre-commit hooks enforce**: Formatting, linting, and type checking run automatically on commit
   - **Invoke /deslop**: Remove AI-generated slop from all changed files
   - **Extract duplicate test setup to fixtures**: If ANY setup code appears in 2+ tests, extract to fixtures immediately
   - **Verify fixture usage**: Check that all tests use fixtures from conftest.py or module-level fixtures
   - **Note**: Pre-commit will auto-fix formatting/linting. If commit fails, fix remaining issues and commit again.
   - Update TODO

6.5. Extract Duplicate Setup to Fixtures (MANDATORY)
   - **Check for duplicate setup**: Scan all test functions in the file
   - **If setup appears 2+ times**: Extract to module-level fixture
   - **If setup is similar but varies**: Create factory fixture with parameters
   - **Place fixtures**: At module level (before test class), following pattern from `tests/datasets/test_uploaded_dataset_lazy_frames.py`
   - **Update all tests**: Replace inline setup with fixture usage
   - **Verify**: Run tests to ensure fixtures work correctly
   - **Note**: Pre-commit hook enforces test fixture rules automatically
   - Update TODO

7. Run Module Test Suite
   - **Before commit**: Full suite required - Command: make test-[module]
   - **During development**: Subset acceptable for faster iteration (e.g., specific test files)
   - **Critical**: Full suite must pass before committing to catch regressions
   - Verify no regressions
   - All tests must pass
   - Update TODO

8. Commit Changes
   - **Pre-commit hooks run automatically**: Formatting, linting, type checking, and test fixture checks
   - **If commit fails**: Pre-commit will show errors. **FIX THE VIOLATIONS** and commit again (hooks auto-fix most issues)
   - **MANDATORY**: Never weaken hooks to allow violations. Fix violations instead.
   - Format: "feat/fix: [description]

     - Change 1
     - Change 2
     - Add comprehensive test suite (X tests passing)

     All tests passing: X/Y
     Following TDD: Red-Green-Refactor"
   - Include implementation AND tests
   - Update TODO to completed
   - **Before switching assistants**: Edit checkpoint file manually with conversation context
   - **Cannot bypass**: Pre-commit hooks cannot be skipped (--no-verify is blocked by project policy)
   - **Cannot weaken**: Pre-commit hooks must block commits unless all checks pass. Never make hooks warn-only, less strict, or disabled.

9. Final Quality Gate & PR Preparation
   - Run: make test-fast (confirms no regressions across entire codebase)
   - Verify all fast tests pass
   - Push changes: git push
   - Open PR using GitHub CLI: gh pr create --title "[feat/fix]: [description]" --body "[PR description]"
   - OR if manual PR creation: Provide PR-ready summary with title and description
   - Update TODO to completed

10. HITL Safety Gate (if triggered)
   If rule 107-hitl-safety is triggered:
   - Halt execution
   - Output C.O.R.E. format only (per rule 230)
   - Populate DECISIONS NEEDED section
   - Await human response
   - Do not proceed until human decision is provided

Checkpoint Logging (LIGHTWEIGHT)

Create a lightweight checkpoint to capture uncommitted work and conversation context for switching between assistants.

Checkpoint Location: `.context/checkpoints/[task_id].md`

Checkpoint Format (Simple Markdown):
```markdown
# [task_id]

**Status**: In progress (since last commit: [hash])

**What I did since last commit**:
- [Brief description of changes made in this chat session]
- [Files modified, tests written, etc.]

**Current state**:
- [Test status, quality gates, uncommitted changes]

**Next steps**:
1. [What needs to happen next]
2. [Any blockers or issues]

**Blockers/Notes**:
- [Any blockers, errors encountered, or important context]
```

Checkpoint Commands:
- Create template: `make checkpoint-create TASK_ID="[task_id]"`
- Edit manually: Add conversation context about what happened since last commit
- Resume: `make checkpoint-resume TASK_ID="[task_id]"` (just shows the file)

**Key Principle**: Checkpoint captures what commits can't - uncommitted work and conversation context. Keep it lightweight and focused on actionable context for resuming work.

Commit History Export (Optional):
- `make git-log-export` - Exports full commit history since main branch to `.context/commits/[branch]_[timestamp].md`
- `make git-log-latest` - Shows latest commit history export

Verification Checklist

Before claiming complete, verify:
- [ ] Tests written BEFORE implementation
- [ ] Tests run immediately after writing (Red verified)
- [ ] Implementation passes tests (Green verified)
- [ ] **Duplicate test setup extracted to fixtures** (if setup appears 2+ times)
- [ ] **All tests use fixtures** (no inline duplicate setup)
- [ ] /deslop invoked to remove AI-generated slop
- [ ] Module tests passing
- [ ] **Commit succeeds** (pre-commit hooks enforce formatting, linting, type checking, test fixtures)
- [ ] make test-fast executed (final quality gate)
- [ ] Changes pushed to remote
- [ ] All TODOs marked completed
- [ ] Checkpoint created and manually updated with conversation context (if switching assistants)

**Note**: Pre-commit hooks automatically enforce:
- Code formatting (ruff format)
- Linting (ruff check --fix)
- Type checking (mypy)
- Test fixture rules (custom hook)
- File syntax validation (YAML, JSON, TOML)
- Trailing whitespace removal
- End of file newlines

These cannot be bypassed. If commit fails, fix the issues and commit again.

Critical Rules

❌ NEVER write code before tests
❌ NEVER skip running tests after writing them
❌ NEVER run pytest/ruff/mypy directly (use Makefile) - EXCEPTION: Red phase verification allows direct pytest
❌ NEVER run Python commands directly - ALWAYS use uv run (e.g., uv run python, uv run pytest)
❌ NEVER use pip or python directly - ALWAYS use uv
❌ NEVER use --no-verify to bypass pre-commit hooks (project policy blocks this)
❌ NEVER commit without tests
❌ NEVER skip TODO updates
❌ **NEVER weaken pre-commit hooks** - Never make hooks warn-only, less strict, or disabled
❌ **NEVER add `|| true` or `pass_filenames: false` or `always_run: false` to bypass hook failures**
❌ **NEVER disable mypy or test fixture checks** - Fix violations instead

✅ ALWAYS write test first
✅ ALWAYS run test immediately (Red phase) - direct pytest OK for quick verification
✅ ALWAYS verify test passes (Green phase) - prefer Makefile, direct pytest acceptable
✅ ALWAYS use Makefile commands for green phase and full suite runs
✅ ALWAYS use uv for Python commands (uv run python, uv run pytest, etc.)
✅ ALWAYS use gh CLI for PR creation (gh pr create)
✅ ALWAYS commit implementation + tests together (pre-commit enforces quality)
✅ ALWAYS update TODOs
✅ **ALWAYS fix pre-existing violations** - If hooks fail, fix the violations before committing
✅ **ALWAYS keep hooks strict** - All hooks must block commits unless passing

**Pre-commit enforcement**: All quality checks (formatting, linting, type checking, test fixtures) are enforced automatically on commit. Cannot be bypassed. Hooks MUST block commits unless all checks pass. Never weaken hooks to allow violations.

Output Format

All human-facing outputs from this command MUST follow the C.O.R.E. (Cognitive-Optimized) format per rule 230-core-output-format.mdc. This format is optimized for fast scanning and decision-making.

**This format should be treated as the canonical C.O.R.E. example for all agent outputs.**

C.O.R.E. Format Template:

## SUMMARY

**Status: ✅ [READY FOR USE | IN PROGRESS | BLOCKED]**

[1-2 lines: outcome status, actionable result]

## ACTIONS REQUIRED 🚨

- [ ] **Action 1** — [context/deadline/impact]
- [ ] **Action 2** — [context/deadline/impact]

## EVIDENCE

**Created:**
- `path/to/file.ext` (description)

**Updated:**
- `path/to/file.ext` (what changed)

**Quality Gates:**
- ✅ **Linting**: All checks passed
- ✅ **Formatting**: All files formatted
- ✅ **Tests**: X/Y passing

## OPTIONAL CONTEXT

**Deliverables:**
- **Feature X** - Description (impact/benefit)
- **Feature Y** - Description (impact/benefit)

**What's Deferred (intentionally):**
- Item 1
- Item 2
- Item 3

**Next Steps:**
1. Step 1
2. Step 2

**Status: ✅ READY FOR USE**

Format Rules:
- **SUMMARY**: 1-2 lines max, bold status at top
- **ACTIONS REQUIRED**: Must include 🚨 emoji, bold action text, context for each
- **EVIDENCE**: Group by Created/Updated/Quality Gates, use file paths with backticks
- **OPTIONAL CONTEXT**: Compress deferred items into bullets, keep prose minimal
- **Status**: Appear at top (under SUMMARY) or bottom (bold + emoji)
- Total output ≤ 80 lines unless explicitly overridden

For each major step during execution, output:

## Step N: [Step Name]

[Brief description of what you're doing]

[Code/command output]

✓ Verified: [What was confirmed]

Communication Style

- Be direct and technical
- No excessive emojis or cheerleading
- State facts, don't ask permission for standard steps
- Challenge bad assumptions
- Report errors clearly with root cause

Example Invocation

User: spec-driven add auto-download for missing LLM models
Agent: [Follows complete TDD workflow as specified above]

Enforcement

If you catch yourself:
- Running pytest directly (outside red phase) → STOP, use make test-[module] for green phase and full suite
- Writing code before tests → STOP, write test first
- Skipping test runs → STOP, run tests now
- Accumulating lint errors → STOP, run make lint-fix
- Not updating TODOs → STOP, update todo_write
- **Weakening pre-commit hooks** → STOP, fix violations instead. Never make hooks warn-only, less strict, or disabled
- **Adding bypasses to hooks** → STOP, fix violations instead. Never add `|| true`, `pass_filenames: false`, or disable hooks

**MANDATORY**: Pre-commit hooks MUST block commits unless all checks pass. If hooks fail:
1. Fix the violations
2. Commit again
3. Never weaken hooks to allow violations

Remember: You're building production systems. Act like a Staff engineer.

End of command.
