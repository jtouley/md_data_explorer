#!/usr/bin/env python3
"""Analyze parse outcome metrics from logs."""
import json
import sys
from collections import Counter
from pathlib import Path


def analyze_logs(log_file: Path) -> None:
    """Parse structlog output and compute metrics with granular checkpoints."""
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return

    outcomes = []
    parse_outcome_found = False

    with open(log_file) as f:
        for line in f:
            if "parse_outcome" in line:
                parse_outcome_found = True
                try:
                    entry = json.loads(line)
                    outcomes.append(entry)
                except json.JSONDecodeError:
                    continue

    if not parse_outcome_found:
        print(f"Warning: No 'parse_outcome' events found in log file: {log_file}")
        print("  Ensure instrumentation is enabled and logs contain parse_outcome events")
        return

    if not outcomes:
        print("No parse outcomes found in logs (all JSON parse attempts failed)")
        return

    total = len(outcomes)
    tier_counts = Counter(o.get("tier") for o in outcomes if "tier" in o)

    tier3_outcomes = [o for o in outcomes if o.get("tier") == "tier3"]
    tier3_total = len(tier3_outcomes)

    # Granular checkpoints
    llm_called = sum(1 for o in tier3_outcomes if o.get("llm_called"))
    llm_http_success = sum(1 for o in tier3_outcomes if o.get("llm_http_success"))
    json_parse_success = sum(1 for o in tier3_outcomes if o.get("json_parse_success"))
    schema_validate_success = sum(1 for o in tier3_outcomes if o.get("schema_validate_success"))
    final_returned = sum(1 for o in tier3_outcomes if o.get("final_returned_from_tier3"))

    print("Parse Outcome Analysis")
    print("=" * 50)
    print(f"Total parses: {total}")
    print("\nTier Distribution:")
    tier1_count = tier_counts.get("tier1", 0)
    tier2_count = tier_counts.get("tier2", 0)
    print(f"  Tier 1 (pattern): {tier1_count} ({tier1_count/total*100:.1f}%)")
    print(f"  Tier 2 (semantic): {tier2_count} ({tier2_count/total*100:.1f}%)")
    print(f"  Tier 3 (LLM): {tier3_total} ({tier3_total/total*100:.1f}%)")

    if tier3_total > 0:
        print("\nTier 3 Pipeline (Granular Checkpoints):")
        print(f"  LLM called: {llm_called}/{tier3_total} ({llm_called/tier3_total*100:.1f}%)")
        print(f"  LLM HTTP success: {llm_http_success}/{tier3_total} ({llm_http_success/tier3_total*100:.1f}%)")
        print(f"  JSON parse success: {json_parse_success}/{tier3_total} ({json_parse_success/tier3_total*100:.1f}%)")
        schema_success_pct = schema_validate_success / tier3_total * 100
        print(f"  Schema validate success: {schema_validate_success}/{tier3_total} ({schema_success_pct:.1f}%)")
        final_returned_pct = final_returned / tier3_total * 100
        print(f"  Final returned: {final_returned}/{tier3_total} ({final_returned_pct:.1f}%)")

    print("\nDiagnostics:")
    print("  Tier 3 rate <10%: LLM path bypassed (lower tier thresholds)")
    print("  LLM HTTP <80%: Ollama unavailable/timing out")
    print("  JSON parse <80%: LLM returning invalid JSON (check model size)")
    print("  Schema validate <80%: Invalid intents (model hallucinating, need 8b+)")


if __name__ == "__main__":
    log_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/nl_query.log")
    analyze_logs(log_file)
