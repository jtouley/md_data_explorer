PLAN UPDATE COMMAND

Role
You are a Staff-level AI engineer updating a plan file with feedback from a review.

Trigger
This command is activated when the user types:

/plan-update <plan-identifier>
/plan-update <plan-name>
/plan-update <plan-file-path>

Examples:
- /plan-update adr004_phase_4_proactive_question_generation_864660ab
- /plan-update adr004_phase_4_proactive_question_generation_864660ab.plan.md
- /plan-update .cursor/plans/adr004_phase_4_proactive_question_generation_864660ab.plan.md

Primary Objective
Update the specified plan file with feedback from its corresponding review file, using the spec-driven workflow.

Execution Sequence

1. Parse Plan Identifier
   - Extract plan identifier from input
   - If only name provided (e.g., "adr004_phase_4_proactive_question_generation_864660ab"), construct paths:
     * Plan file: `.cursor/plans/{plan-name}.plan.md`
     * Review file: `.context/reviews/plan_{plan-name}.md`
   - If filename provided (e.g., "adr004_phase_4_proactive_question_generation_864660ab.plan.md"), extract base name and construct both paths
   - If full path provided (e.g., ".cursor/plans/xxx.plan.md"), extract plan name and construct review path
   - Normalize to full path format

2. Verify Files Exist
   - Check if plan file exists at constructed path
   - Check if review file exists at constructed path
   - If either file doesn't exist, report error with expected paths and stop

3. Load Review File
   - Read review file from: `.context/reviews/plan_{plan-name}.md`
   - Extract key feedback sections:
     * Blocking Issues
     * Non-Blocking Feedback
     * Update Instructions
   - Prepare feedback summary for spec-driven command

4. Execute Spec-Driven Update
   - Construct spec-driven command:
     `/spec-driven update {plan-file-path} with feedback here {review-file-path}`
   - Execute the spec-driven workflow to update the plan
   - The spec-driven command will:
     * Read the plan file
     * Read the review file
     * Apply feedback to update the plan
     * Follow TDD workflow if code changes are needed

5. Output Summary
   - Display brief summary of update operation
   - Confirm which files were processed
   - Report any issues encountered

File Path Resolution Logic

Input: "adr004_phase_4_proactive_question_generation_864660ab"
→ Plan: `.cursor/plans/adr004_phase_4_proactive_question_generation_864660ab.plan.md`
→ Review: `.context/reviews/plan_adr004_phase_4_proactive_question_generation_864660ab.md`

Input: "adr004_phase_4_proactive_question_generation_864660ab.plan.md"
→ Plan: `.cursor/plans/adr004_phase_4_proactive_question_generation_864660ab.plan.md`
→ Review: `.context/reviews/plan_adr004_phase_4_proactive_question_generation_864660ab.md`

Input: ".cursor/plans/adr004_phase_4_proactive_question_generation_864660ab.plan.md"
→ Plan: `.cursor/plans/adr004_phase_4_proactive_question_generation_864660ab.plan.md`
→ Review: `.context/reviews/plan_adr004_phase_4_proactive_question_generation_864660ab.md`

Error Handling

- If plan file not found: Report error with expected path and stop
- If review file not found: Report error with expected path and stop
- If plan identifier cannot be parsed: Report error and stop
- If review file is empty: Report warning but proceed

Output Format

Chat Summary:
```
## Plan Update: {plan-name}

**Plan File**: {plan-file-path}
**Review File**: {review-file-path}

Updating plan with feedback from review...

[Execute spec-driven update workflow]

✓ Plan updated successfully
```

Communication Style

- Be direct and technical
- No fluff, no excessive explanations
- State facts clearly
- Report errors with full context

Integration with /spec-driven

This command is a convenience wrapper around /spec-driven that:
- Automatically resolves file paths
- Reduces typing for the common workflow:
  1. User runs /plan-review to get feedback
  2. User runs /plan-update to apply feedback
  3. User runs /spec-driven to execute updated plan

Example Invocation

User: /plan-update adr004_phase_4_proactive_question_generation_864660ab
Agent: 
- Resolves: `.cursor/plans/adr004_phase_4_proactive_question_generation_864660ab.plan.md`
- Resolves: `.context/reviews/plan_adr004_phase_4_proactive_question_generation_864660ab.md`
- Executes: `/spec-driven update .cursor/plans/adr004_phase_4_proactive_question_generation_864660ab.plan.md with feedback here .context/reviews/plan_adr004_phase_4_proactive_question_generation_864660ab.md`

User: /plan-update .cursor/plans/config_migration_to_yaml_ed79c904.plan.md
Agent:
- Uses provided plan path
- Resolves: `.context/reviews/plan_config_migration_to_yaml_ed79c904.md`
- Executes: `/spec-driven update .cursor/plans/config_migration_to_yaml_ed79c904.plan.md with feedback here .context/reviews/plan_config_migration_to_yaml_ed79c904.md`

End of command.

