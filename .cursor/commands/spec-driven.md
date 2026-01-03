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
   - Confirm it fails for the RIGHT reason
   - NEVER skip this step

4. Implement Feature (Green Phase)
   - Write minimum code to pass the test
   - Keep it simple

5. Run Test to Verify Pass
   - Use Makefile command: make test-[module] PYTEST_ARGS="tests/.../test_file.py -xvs" (if supported)
   - OR direct pytest: uv run pytest tests/.../test_file.py::TestClass::test_method -xvs
   - Confirm test passes
   - Update TODO

6. Fix Quality Issues (Refactor Phase)
   - Run: make format
   - Run: make lint-fix
   - Fix any remaining issues manually
   - **Invoke /deslop**: Remove AI-generated slop from all changed files
   - Update TODO

7. Run Module Test Suite
   - Command: make test-[module]
   - Verify no regressions
   - All tests must pass
   - Update TODO

8. Commit Changes
   - Format: "feat/fix: [description]
     
     - Change 1
     - Change 2
     - Add comprehensive test suite (X tests passing)
     
     All tests passing: X/Y
     Following TDD: Red-Green-Refactor"
   - Include implementation AND tests
   - Update TODO to completed
   - **Before switching assistants**: Edit checkpoint file manually with conversation context

9. HITL Safety Gate (if triggered)
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
- [ ] make format executed
- [ ] make lint-fix executed
- [ ] /deslop invoked to remove AI-generated slop
- [ ] Zero NEW linting errors in changed files
- [ ] Module tests passing
- [ ] Changes committed with tests
- [ ] All TODOs marked completed
- [ ] Checkpoint created and manually updated with conversation context (if switching assistants)

Critical Rules

❌ NEVER write code before tests
❌ NEVER skip running tests after writing them
❌ NEVER run pytest/ruff/mypy directly (use Makefile) - EXCEPTION: Red phase verification allows direct pytest
❌ NEVER accumulate quality issues
❌ NEVER commit without tests
❌ NEVER skip TODO updates

✅ ALWAYS write test first
✅ ALWAYS run test immediately (Red phase) - direct pytest OK for quick verification
✅ ALWAYS verify test passes (Green phase) - prefer Makefile, direct pytest acceptable
✅ ALWAYS use Makefile commands for green phase and full suite runs
✅ ALWAYS fix quality issues immediately
✅ ALWAYS commit implementation + tests together
✅ ALWAYS update TODOs

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

Remember: You're building production systems. Act like a Staff engineer.

End of command.

