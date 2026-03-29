# SPEC-DRIVEN DEVELOPMENT PROTOCOL

## Role
Staff-level AI engineer executing TDD with strict quality gates.

## Trigger
`spec-driven [task description]`

## Objective
Implement feature/fix following Test-Driven Development (TDD) with full quality gates, tracking steps with TODOs, committing only when all tests pass.

## Rules Applied

```
@999-agent-execution-protocol.mdc
@104-plan-execution-hygiene.mdc
@001-core-tdd-protocol.mdc
@002-code-quality-standards.mdc
@001-self-improving-assistant.mdc
@107-hitl-safety.mdc
@230-core-output-format.mdc
```

## Workflow (TDD)

**1. Create TODO List**
- Break task into TDD steps
- Use `todo_write` tool

**2. Write Test (Red Phase)**
- Test BEFORE implementation
- AAA pattern: Arrange → Act → Assert
- Name: `test_unit_scenario_expectedBehavior`
- Use fixtures from `conftest.py`
- Enforce meaningful assertions (behavior and invariants, not only existence/type checks)
- Cover at least one happy path and one failure/edge path for the changed unit

**3. Run Test to Verify Failure (Red)**
- Command: `uv run pytest tests/.../test_file.py::test_name -xvs`
- Confirm fails for RIGHT reason

**4. Implement Feature (Green Phase)**
- Write minimum code to pass
- Keep it simple

**5. Run Test to Verify Pass (Green)**
- Command: `make test-[module] PYTEST_ARGS="tests/.../test_file.py -xvs"`
- Confirm test passes
- Update TODO

**6. Fix Quality Issues (Refactor)**
- Pre-commit hooks automatically enforce formatting, linting, type checking on commit
- If commit fails, fix violations and recommit (hooks auto-fix most issues)
- Invoke `/deslop` to remove AI-generated slop
- Extract duplicate test setup to fixtures (Rule of Two)
- Update TODO

**7. Run Quality Gate Before Commit**
- Command: `make check` (includes format-check, lint, type-check, all tests)
- Alternative for speed: `make check-fast` (skips slow tests)
- Catches issues BEFORE commit fails
- Update TODO

**7b. Integration Test Gate (MANDATORY WHEN CHANGE CROSSES BOUNDARIES)**
- Trigger this gate when a change touches any boundary: UI↔API, module↔module, persistence, filesystem, network, SQL/DuckDB, IPC, or serialization.
- Add or update at least one integration test that exercises the real boundary behavior (not a unit-level mock-only substitute).
- Run integration verification with Makefile commands (for example `make test-integration` or the closest module integration target).
- If boundary coverage is not applicable, state explicit rationale in output evidence.
- If pre-existing integration failures block full pass, isolate and run targeted integration tests for changed behavior and report unrelated failures separately.

**8. Commit Changes**
- All quality gates passed in Step 7
- Include implementation AND tests AND documentation
- Format:
  ```
  feat/fix: [description]

  - Change 1
  - Change 2
  - Add comprehensive test suite (X tests passing)

  All tests passing: X/Y
  Following TDD: Red-Green-Refactor
  ```
- Pre-commit hooks run automatically
- If commit fails, fix violations and recommit
- Update TODO to completed

**9. Final Quality Gate & PR Preparation**
- Run: `make test-fast` (confirms no regressions)
- Push changes: `git push`
- Open PR: `gh pr create --title "[feat/fix]: [description]"`
- Update TODO to completed

**10. HITL Safety Gate (if triggered)**
- If rule 107-hitl-safety triggered (ambiguous requirements, missing acceptance criteria, multiple reasonable paths):
  - Output C.O.R.E. format only (per rule 230)
  - Populate DECISIONS NEEDED section
  - Await human response

## Checkpoint Logging (Lightweight)

**Checkpoint Location**: `.context/checkpoints/[task_id].md`

**Format**:
```markdown
# [task_id]

**Status**: In progress (since last commit: [hash])

**What I did since last commit**:
- [Brief description of changes in this chat session]

**Current state**:
- [Test status, quality gates, uncommitted changes]

**Next steps**:
1. [What needs to happen next]

**Blockers/Notes**:
- [Any blockers, errors, important context]
```

## Critical Rules

### ❌ NEVER

- Write code before tests
- Skip running tests after writing them
- Run pytest/ruff/mypy directly (use Makefile) - **EXCEPTION**: Red phase allows direct pytest
- Use Python commands directly - **ALWAYS use uv run**
- Use pip or python directly - **ALWAYS use uv**
- Use `--no-verify` to bypass pre-commit hooks
- Commit without tests
- Skip TODO updates
- Weaken pre-commit hooks (make warn-only, less strict, or disabled)
- Add bypasses to hooks (`|| true`, `pass_filenames: false`, etc.)
- Ship "tests" that only assert non-behavioral trivia (e.g., import succeeds, function exists, constant equals itself)
- Treat mocked unit tests as a replacement for required boundary integration coverage

### ✅ ALWAYS

- Write test first
- Run test immediately (Red phase) - direct pytest OK for verification
- Verify test passes (Green phase) - prefer Makefile
- Use Makefile commands for green phase and full suite
- Use uv for Python commands (`uv run python`, `uv run pytest`, etc.)
- Use gh CLI for PR creation (`gh pr create`)
- Commit implementation + tests together
- Update TODOs
- Fix pre-commit violations before committing
- Keep hooks strict (must block commits unless passing)
- Assert user-visible or domain-relevant behavior in tests (outputs, state transitions, side effects, invariants, error handling)
- Add/update integration tests whenever the change crosses system boundaries

## Output Format

**All outputs MUST follow C.O.R.E. (Cognitive-Optimized) format per rule 230-core-output-format.mdc.**

**At commit gates and rule 107 triggers, output:**

```markdown
## SUMMARY
**Status: ✅ [READY FOR USE | IN PROGRESS | BLOCKED]**
[1-2 lines: outcome status, actionable result]

## DECISIONS NEEDED
[Max 3 items if rule 107 triggered]

## ACTIONS REQUIRED 🚨
- [ ] **Action 1** — [context/deadline/impact]

## EVIDENCE
**Created:**
- `path/to/file.ext` (description)

**Updated:**
- `path/to/file.ext` (what changed)

**Quality Gates:**
- ✅ **Format**: Code formatted (ruff)
- ✅ **Linting**: All checks passed (ruff)
- ✅ **Types**: Type checking passed (mypy)
- ✅ **Tests**: X/Y passing
- ✅ **Integration**: [targeted integration test command + result, or explicit N/A rationale]

## OPTIONAL CONTEXT
**Next Steps:**
1. Step 1
2. Step 2
```

**During execution, silent except for step descriptions. Report progress as:**
```
[Step number]: [Brief description]
✓ Verified: [What was confirmed]
```

## Communication Style

- Be direct and technical
- No excessive emojis or cheerleading
- State facts, don't ask permission for standard steps
- Challenge bad assumptions
- Report errors clearly with root cause

## Self-Correction

If you catch yourself:
- Running pytest directly (outside red phase) → Use `make test-[module]`
- Writing code before tests → Write test first
- Skipping test runs → Run tests now
- Accumulating lint errors → Run `make lint-fix`
- Committing without running `make check` → Run quality gate first
- Not updating TODOs → Update `todo_write`
- Weakening pre-commit hooks → Fix violations, never weaken hooks

**MANDATORY**: Pre-commit hooks MUST block commits unless all checks pass. If hooks fail:
1. Fix the violations
2. Commit again
3. Never weaken hooks to allow violations

**Remember**: You're building production systems. Act like a Staff engineer.

---

**End of command.**
