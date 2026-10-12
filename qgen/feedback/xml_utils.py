"""Parse the XML-style fields required by the upstream prompts."""

import re

from qgen.utils.filtering_utils import is_valid_date_format

QUESTION_PATTERN = r'<question\s+id="(\d+)"\s*>\s*(.*?)\s*</question>'

QUESTION_FIELDS = ("question_title", "start_date", "background", "resolution_date", "resolution_criteria", "answer")


def extract_tag_content(text, tag):
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return matches[-1].strip() if matches else ""



def extract_final_question(response):
    # Parse one complete block, never assemble fields from different candidates.
    blocks = re.findall(QUESTION_PATTERN, response, re.DOTALL)
    if not blocks:
        return ""
    number, content = blocks[-1]
    fields = re.findall(r"<(question_title|start_date|background|resolution_date|resolution_criteria|answer)>\s*(.*?)\s*</\1>", content, re.DOTALL)
    if tuple(field for field, _ in fields) != QUESTION_FIELDS:
        return ""
    values = dict(fields)
    if any(not value.strip() for field, value in fields if field != "background"):
        return ""
    if any(not is_valid_date_format(values[field].strip()) for field in ("start_date", "resolution_date")):
        return ""
    return f'<question id="{number}">\n{content}\n</question>'


def parse_generated_questions(response, max_questions):
    pattern = QUESTION_PATTERN
    blocks = list(re.finditer(pattern, response, re.DOTALL))
    if not blocks or len(blocks) > max_questions or re.sub(pattern, "", response, flags=re.DOTALL).strip():
        raise ValueError("Expected between one and the requested maximum number of question blocks")
    candidates = []
    for number, block in enumerate(blocks, 1):
        question = extract_final_question(block.group(0))
        if block.group(1) != str(number) or not question:
            raise ValueError(f"Malformed or out-of-order question {number} generation")
        candidates.append({"candidate_id": number, "question": question})
    return candidates
