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

**3. Run Test to Verify Failure (Red)**
- Command: `uv run pytest tests/.../test_file.py::test_name -xvs`
- Confirm fails for RIGHT reason
- NEVER skip this step

**4. Implement Feature (Green Phase)**
- Write minimum code to pass
- Keep it simple

**5. Run Test to Verify Pass (Green)**
- Command: `make test-[module] PYTEST_ARGS="tests/.../test_file.py -xvs"`
- Confirm test passes
- Update TODO

**6. Fix Quality Issues (Refactor)**
- **Pre-commit hooks enforce**: Formatting, linting, type checking (auto-run on commit)
- **Invoke `/deslop`**: Remove AI-generated slop
- **Extract duplicate setup**: If setup appears in 2+ tests, extract to fixtures immediately
- **Note**: Pre-commit auto-fixes most issues. If commit fails, fix remaining violations and commit again.
- Update TODO

**7. Run Module Test Suite**
- **Before commit**: Full suite required - `make test-[module]`
- **During development**: Subset acceptable (e.g., specific test files)
- **Critical**: Full suite must pass before committing
- Update TODO

**8. Commit Changes**
- **Pre-commit hooks run automatically**: Formatting, linting, type checking, test fixtures
- **If commit fails**: Fix violations and commit again (hooks auto-fix most issues)
- **Format**:
  ```
  feat/fix: [description]

  - Change 1
  - Change 2
  - Add comprehensive test suite (X tests passing)

  All tests passing: X/Y
  Following TDD: Red-Green-Refactor
  ```
- Include implementation AND tests
- Update TODO to completed
- **Cannot bypass**: Pre-commit hooks cannot be skipped (`--no-verify` blocked by policy)
- **Cannot weaken**: Hooks MUST block commits unless all checks pass

**9. Final Quality Gate & PR Preparation**
- Run: `make test-fast` (confirms no regressions)
- Verify all fast tests pass
- Push changes: `git push`
- Open PR: `gh pr create --title "[feat/fix]: [description]" --body "[PR description]"`
- Update TODO to completed

**10. HITL Safety Gate (if triggered)**
- If rule 107-hitl-safety triggered:
  - Halt execution
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

**Commands**:
- Create: `make checkpoint-create TASK_ID="[task_id]"` (edit manually with conversation context)
- Resume: `make checkpoint-resume TASK_ID="[task_id]"`

**Key Principle**: Checkpoint captures what commits can't - uncommitted work and conversation context.

## Verification Checklist

Before claiming complete:
- [ ] Tests written BEFORE implementation
- [ ] Tests run immediately after writing (Red verified)
- [ ] Implementation passes tests (Green verified)
- [ ] Duplicate test setup extracted to fixtures (if setup appears 2+ times)
- [ ] All tests use fixtures (no inline duplicate setup)
- [ ] `/deslop` invoked to remove AI slop
- [ ] Module tests passing
- [ ] Commit succeeds (pre-commit hooks enforce quality)
- [ ] `make test-fast` executed (final quality gate)
- [ ] Changes pushed to remote
- [ ] All TODOs marked completed
- [ ] Checkpoint created and updated (if switching assistants)

**Pre-commit hooks automatically enforce:**
- Code formatting (ruff format)
- Linting (ruff check --fix)
- Type checking (mypy)
- Test fixture rules (custom hook)
- File syntax validation
- Trailing whitespace removal
- End-of-file newlines

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

### ✅ ALWAYS

- Write test first
- Run test immediately (Red phase) - direct pytest OK for verification
- Verify test passes (Green phase) - prefer Makefile
- Use Makefile commands for green phase and full suite
- Use uv for Python commands (`uv run python`, `uv run pytest`, etc.)
- Use gh CLI for PR creation (`gh pr create`)
- Commit implementation + tests together (pre-commit enforces quality)
- Update TODOs
- Fix pre-commit violations before committing
- Keep hooks strict (must block commits unless passing)

**Pre-commit enforcement**: All quality checks enforced automatically on commit. Cannot be bypassed. Hooks MUST block commits unless all checks pass.

## Output Format

**All outputs MUST follow C.O.R.E. (Cognitive-Optimized) format per rule 230-core-output-format.mdc.**

**Format:**
```
## SUMMARY
**Status: ✅ [READY FOR USE | IN PROGRESS | BLOCKED]**
[1-2 lines: outcome status, actionable result]

## ACTIONS REQUIRED 🚨
- [ ] **Action 1** — [context/deadline/impact]

## EVIDENCE
**Created:**
- `path/to/file.ext` (description)

**Updated:**
- `path/to/file.ext` (what changed)

**Quality Gates:**
- ✅ **Linting**: All checks passed
- ✅ **Tests**: X/Y passing

## OPTIONAL CONTEXT
**Next Steps:**
1. Step 1
2. Step 2

**Status: ✅ READY FOR USE**
```

**For each major step during execution:**
```
## Step N: [Step Name]
[Brief description]
[Code/command output]
✓ Verified: [What was confirmed]
```

## Communication Style

- Be direct and technical
- No excessive emojis or cheerleading
- State facts, don't ask permission for standard steps
- Challenge bad assumptions
- Report errors clearly with root cause

## Example Invocation

```
User: spec-driven add auto-download for missing LLM models
Agent: [Follows complete TDD workflow as specified above]
```

## Enforcement

If you catch yourself:
- Running pytest directly (outside red phase) → STOP, use `make test-[module]`
- Writing code before tests → STOP, write test first
- Skipping test runs → STOP, run tests now
- Accumulating lint errors → STOP, run `make lint-fix`
- Not updating TODOs → STOP, update `todo_write`
- Weakening pre-commit hooks → STOP, fix violations. Never weaken hooks
- Adding bypasses to hooks → STOP, fix violations. Never add bypasses

**MANDATORY**: Pre-commit hooks MUST block commits unless all checks pass. If hooks fail:
1. Fix the violations
2. Commit again
3. Never weaken hooks to allow violations

**Remember**: You're building production systems. Act like a Staff engineer.

---

**End of command.**
