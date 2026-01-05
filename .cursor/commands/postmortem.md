# POSTMORTEM COMMAND

## Role
Generate an actionable forensic artifact that results in verifiable rule/command updates.

## Trigger
This command is activated when the user types:

/postmortem <incident_id>

Examples:
- /postmortem test_incident
- /postmortem pr32_failure
- /postmortem governance_framework_issue

## Primary Objective
Analyze an incident (failed plan, regression, spec drift) and produce actionable fixes that improve governance infrastructure (rules, commands, checkpoints).

## Execution Sequence

1. Parse Incident ID
   - Extract incident identifier from input
   - Normalize to identifier format

2. Load Incident Artifacts
   - Plan artifact for incident (if available): `.cursor/plans/{incident_id}.plan.md` or `.context/plans/{incident_id}.md`
   - Git diff for incident (if available): Use `git diff` or `.context/diffs/{incident_id}.diff`
   - Evaluations (if available): `.context/evaluations/{incident_id}_*.md`
   - Checkpoints (if available): `.context/checkpoints/{incident_id}.md`

3. Perform Analysis
   - Identify what happened (3–5 structured bullets)
   - Quantify impact (regressions, drift events)
   - Identify root causes (atomic sentences)
   - Identify missed guards (rules/commands that should have caught this)
   - Generate actionable fixes (spec deltas, rule modifications, command modifications)

4. Output Postmortem
   - Follow C.O.R.E. output format (per rule 230-core-output-format.mdc)
   - Write artifact to: `.context/postmortems/{incident_id}.md`
   - Create `.context/postmortems/` directory if it doesn't exist

## Output Format

**All outputs MUST follow C.O.R.E. (Cognitive-Optimized) format per rule 230-core-output-format.mdc.**

### Chat Output (C.O.R.E. Format)

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
- `.context/postmortems/{incident_id}.md` (postmortem artifact)

**Updated:**
- [list any files modified]

## OPTIONAL CONTEXT

[Only if explicitly requested]
```

### Artifact Template

The postmortem artifact written to `.context/postmortems/{incident_id}.md` MUST follow this structure:

```markdown
# Postmortem: <incident_id>
Date: <ISO8601>

## What Happened
- <3–5 structured bullets; machine-parsable>

## Quantified Impact
- unexpected_regressions: <count>
- drift_events: <count>

## Root Causes
- <list of root causes, atomic sentences>

## Missed Guards
- rule_refs: <list of .mdc rules>
- command_refs: <list of .md files>

## Generated Fixes
- fixes: [list of actionable changes]
  - Spec delta templates (if applicable)
  - Rule modifications (if applicable)
  - Command modifications (if applicable)

## Verification Artifacts
- tests_added: [list]
- evaluations_added: [list]
```

## Processing Rules

- Postmortem MUST produce at least one actionable structural change
- If postmortem does not create a delta, it MUST be invalidated and remediated
- Output MUST follow C.O.R.E. format (rule 230-core-output-format.mdc)
- Artifact MUST be written to `.context/postmortems/{incident_id}.md`

## Rules Applied

```
@230-core-output-format.mdc - Cognitive-optimized output (MVP)
@107-hitl-safety.mdc - Human-in-the-loop safety (MVP)
```

## Critical Rules

### ❌ NEVER

- Produce postmortems without actionable fixes
- Skip artifact creation
- Use narrative prose instead of structured format
- Exceed 80 lines in chat output (use C.O.R.E. format)

### ✅ ALWAYS

- Follow C.O.R.E. output format
- Produce at least one actionable fix
- Write artifact to `.context/postmortems/`
- Reference rule 230 in output
