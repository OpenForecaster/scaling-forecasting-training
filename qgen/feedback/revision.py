"""One feedback-driven revision, continuing each model's own conversation."""

import json

from .xml_utils import parse_generated_questions
from .question_generator import question_output_format
from .verification import VERIFICATION_OUTPUT_FORMAT, parse_verification


def continue_conversation(prompt, response, followup):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
        {"role": "user", "content": followup},
    ]


def revision_prompt(candidate):
    return f"""Revise question {candidate['candidate_id']} using this verifier feedback:
{json.dumps(candidate['verification'], ensure_ascii=False)}

Satisfy all the original criteria. If the current question is difficult to repair, replace it with a completely different question grounded in the same document. Return exactly one complete question, using id="1" for this response. Use YYYY-MM-DD for both dates. Return only the following XML format, without commentary:
{question_output_format(1)}"""


def reverification_prompt(question):
    return f"""Check this revised question against every original numbered criterion using the source document already provided. Reassess all criteria, including any new violations. List all violated IDs and explain the violations in one paragraph. Return an empty violations list if none are violated.

Question:
{question}

Return only a JSON object in this format:
{VERIFICATION_OUTPUT_FORMAT}"""


async def revise_question(row, generator_engine, verifier_engine, on_update=None):
    if any('verification' not in candidate for candidate in row['candidates']):
        return
    if any(not c['verification']['violations'] for c in row['candidates']):
        return
    flagged = [c for c in row['candidates'] if c['verification']['violations']]
    if not flagged:
        return
    candidate = min(flagged, key=lambda c: (len(c['verification']['violations']), c['candidate_id']))
    if 'revision' not in candidate:
        chat = continue_conversation(row['generation_prompt'], row['generated_questions'],
                                     revision_prompt(candidate))
        try:
            response = (await generator_engine.generate([chat]))[0]["response"]
        except Exception as error:
            candidate['revision'] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            if on_update:
                on_update()
            return
        revision = {'generation_messages': chat, 'generation_response': response}
        candidate['revision'] = revision
        try:
            revision['question'] = parse_generated_questions(response, 1)[0]['question'].replace(
                '<question id="1">', f'<question id="{candidate["candidate_id"]}">', 1)
            revision['status'] = 'generated'
        except (ValueError, TypeError) as error:
            revision.update(status='failed', error=str(error))
        if on_update:
            on_update()
    revision = candidate['revision']
    if revision['status'] != 'generated':
        return
    chat = continue_conversation(candidate['verification_prompt'], candidate['verification_response'],
                                 reverification_prompt(revision['question']))
    try:
        response = (await verifier_engine.generate([chat]))[0]["response"]
        revision.update(verification_messages=chat, verification_response=response)
        revision['verification'] = parse_verification(response)
        revision['status'] = 'completed'
    except Exception as error:
        revision.update(status='failed', error=f'{type(error).__name__}: {error}')
    if on_update:
        on_update()
