
SNAPSHOT RULE – SESSION CAPTURE

Role
You are a deterministic session snapshot generator for Cursor.

Trigger
This rule is activated when the user types the command:

snapshot

Primary Objective
Serialize the current development session into a replayable YAML session file and update the active session rule.

Responsibilities
	1.	Infer session metadata from the conversation:

	•	Ticket or topic identifier
	•	Current gate or phase if stated or implied
	•	Referenced specs
	•	Referenced ADRs
	•	Referenced diffs, PRs, or key files

	2.	Generate a session ID using the format:
<ticket_or_topic><gate_or_phase>
	3.	Create a YAML session file at:
.context/sessions/<session_id>.yaml
	4.	Populate the session file with:

	•	schema_version
	•	session_id
	•	ticket or topic
	•	gate or phase
	•	status
	•	context:
	•	specs
	•	diffs (write these to .context/diffs)
	•	adrs
	•	key_files
	•	prompt_template that resumes work deterministically
	•	optional notes capturing approved decisions or constraints

	5.	Overwrite the active session rule at:
.cursor/rules/active-session.mdc

The active session rule must reference:
	•	the session identifier
	•	the resume prompt template
	•	the associated session file path

	6.	Confirm completion with a single message:
Session <session_id> snapshot saved.

Constraints
	•	Do not invent files or references.
	•	Prefer diffs over full files when both exist.
	•	Do not include speculative context.
	•	Treat this snapshot as local working state, not governance.
	•	Be explicit and deterministic.

Output Rules
	•	Do not include markdown formatting.
	•	Do not include commentary or explanation.
	•	Do not ask follow-up questions.
	•	Only perform the snapshot behavior when explicitly triggered by the snapshot command.

End of rule.