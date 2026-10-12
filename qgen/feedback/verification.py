"""Report criterion violations without selecting or rewriting a candidate."""

import json

from .criteria import CRITERIA_GROUPS, format_criteria


VERIFICATION_OUTPUT_FORMAT = '{"violations": ["A1", "C2"], "explanation": "One paragraph explaining the violations."}'


def format_verification_prompt(question, source_document):
    return f"""**Task:** Check the forecasting question against every numbered criterion below using the source document. First list all violated criterion IDs in violations. Then explain each violation in the explanation field. If none are violated, return an empty violations list while still including the explanation field which could be empty as well.

**Criteria:**
{format_criteria()}

**Document:**
{source_document}

**Question:**
{question}

Return only a JSON object in this format:
{VERIFICATION_OUTPUT_FORMAT}
"""


def parse_verification(response):
    report = json.loads(response)
    if not isinstance(report, dict) or not isinstance(report.get("violations"), list):
        raise ValueError("Expected a JSON object containing a violations list")
    known_ids = {f"{letter}{number}" for letter, _, criteria in CRITERIA_GROUPS
                 for number in range(1, len(criteria) + 1)}
    explanation = report.get("explanation")
    if not isinstance(explanation, str) or (report["violations"] and not explanation.strip()):
        raise ValueError("Expected an explanation string, nonempty when violations are listed")
    seen = set()
    for criterion_id in report["violations"]:
        if not isinstance(criterion_id, str) or criterion_id not in known_ids or criterion_id in seen:
            raise ValueError(f"Unknown or duplicate criterion ID: {criterion_id!r}")
        seen.add(criterion_id)
    return report
