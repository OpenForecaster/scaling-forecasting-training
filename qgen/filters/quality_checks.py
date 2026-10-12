"""Optional document grounding, searched date correction, and difficulty checks."""

import json
import math
from datetime import date, datetime, timedelta, timezone

from qgen.utils.xml_utils import extract_tag_content


def question_text(question):
    # Labels, source metadata, and hidden dates are not forecasting inputs.
    return (f"Question: {question['question_title']}\n"
            f"Background: {question.get('background', '')}\n"
            f"Resolution criteria: {question['resolution_criteria']}")


async def complete(engine, prompt, trace, stage):
    result = (await engine.generate([prompt], batch_size=1))[0]
    trace.append({'stage': stage, 'prompt': prompt, 'result': result})
    if result is None or not result.get('response'):
        raise ValueError('Inference returned no response')
    return result['response']


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


def parse_outcomes(text):
    outcomes = parse_json(text)['outcomes']
    if not isinstance(outcomes, list) or not 1 <= len(outcomes) <= 5:
        raise ValueError('Expected one to five outcomes')
    seen = set()
    for outcome in outcomes:
        answer, probability = outcome['answer'], outcome['probability']
        if not isinstance(answer, str) or not answer.strip() or answer.strip().casefold() in seen:
            raise ValueError('Expected distinct, nonempty answer strings')
        seen.add(answer.strip().casefold())
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError('Expected numeric probabilities')
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Probabilities must be finite and between zero and one')
    if sum(outcome['probability'] for outcome in outcomes) > 1 + 1e-6:
        raise ValueError('Outcome probabilities must sum to at most one')
    return outcomes


async def answer_matches(question, answer, judge, trace):
    prompt = f"""Decide whether the proposed answer has the same meaning as the reference answer for this question. Accept paraphrases, abbreviations, and equivalent names. Reject distinct outcomes and extra claims that change the answer.

{question_text(question)}
Reference answer: {json.dumps(question['answer'])}
Proposed answer: {json.dumps(answer)}

Return only <answer>1</answer> for a match or <answer>0</answer> otherwise."""
    text = await complete(judge, prompt, trace, 'answer_match')
    verdict = extract_tag_content(text, 'answer')
    if verdict not in {'0', '1'}:
        raise ValueError('Answer matcher did not return a binary verdict')
    return verdict == '1'


async def forecast(question, evidence, instructions, engine, judge, trace):
    prompt = f"""{instructions}

{question_text(question)}

Evidence:
{evidence}

Return up to five distinct possible answers with probabilities. Use concise answers and probabilities between zero and one that sum to at most one; any remaining probability represents unlisted outcomes. Return only JSON:
{{"outcomes": [{{"answer": "...", "probability": 0.5}}]}}"""
    outcomes = parse_outcomes(await complete(engine, prompt, trace, 'forecast'))
    matches = [await answer_matches(question, outcome['answer'], judge, trace) for outcome in outcomes]
    top_index = max(range(len(outcomes)), key=lambda index: outcomes[index]['probability'])
    return {'status': 'completed', 'outcomes': outcomes, 'matches': matches,
            'top1_correct': matches[top_index],
            'top1_probability': outcomes[top_index]['probability'],
            'p_correct': max((outcome['probability'] for outcome, match in zip(outcomes, matches) if match), default=0.0)}


async def document_grounding(question, engine, judge, trace):
    document = question['article_maintext']
    published = question['article_publish_date']
    if not document or not published:
        raise ValueError('Document grounding requires article_maintext and article_publish_date')
    evidence = f"Published date: {published}\n{document}"
    result = await forecast(question, evidence,
                            'Answer using only information explicitly stated in the provided document. Do not use outside knowledge or search.',
                            engine, judge, trace)
    result['passed'] = result['top1_correct'] and result['top1_probability'] > 0.5
    return result


DATE_INSTRUCTIONS = """Search public sources to establish both dates.
start_date: the earliest date when the premise made sense and the information in the title, background, and resolution criteria was publicly available, so this question could reasonably have been asked as a forecast.
resolution_date: the earliest date when the answer became publicly known with certainty under the resolution criteria.
Use valid YYYY-MM-DD dates, with start_date earlier than resolution_date.
Return only JSON with answer, start_date, and resolution_date."""


def parse_dates(text):
    result = parse_json(text)
    if set(result) != {'answer', 'start_date', 'resolution_date'}:
        raise ValueError('Expected only answer, start_date, and resolution_date')
    if not isinstance(result['answer'], str) or not result['answer'].strip():
        raise ValueError('Expected a nonempty answer')
    for field in ('start_date', 'resolution_date'):
        value = result[field]
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError(f'{field} must be a valid YYYY-MM-DD date')
    if result['start_date'] >= result['resolution_date']:
        raise ValueError('start_date must be earlier than resolution_date')
    return result


async def correct_dates(question, engine, judge, trace):
    messages = [{'role': 'user', 'content': f"Find the answer and forecasting dates.\n\n{question_text(question)}\n\n{DATE_INSTRUCTIONS}"}]
    errors = []
    for turn in range(2):
        while True:
            text = await complete(engine, list(messages), trace, 'date_research')
            messages.append({'role': 'assistant', 'content': text})
            try:
                result = parse_dates(text)
                break
            except (ValueError, TypeError) as error:
                errors.append(str(error))
                if len(errors) > 1:
                    return {'status': 'invalid_response', 'passed': False, 'errors': errors}
                messages.append({'role': 'user', 'content': f"Your response failed validation: {error}. Correct it using the evidence already gathered.\n{DATE_INSTRUCTIONS}"})
        matched = await answer_matches(question, result['answer'], judge, trace)
        if matched:
            return {'status': 'completed', 'passed': True, **result, 'errors': errors,
                    'original_start_date': question.get('question_start_date'),
                    'original_resolution_date': question.get('resolution_date')}
        if turn == 0:
            messages.append({'role': 'user', 'content': f"The correct answer is {json.dumps(question['answer'])}. Recheck both dates using this answer.\n{DATE_INSTRUCTIONS}"})
    return {'status': 'unresolved', 'passed': False, **result, 'errors': errors}


def calendar_date(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).date()
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.date()


def historical_evidence(question, cutoff):
    """Read the existing embedding-retrieval output, keeping only dated evidence."""
    if 'relevant_articles_sorted_by_docs' not in question and 'relevant_docs' not in question:
        raise ValueError('Difficulty scoring requires relevant_articles_sorted_by_docs or relevant_docs')
    documents = question.get('relevant_articles_sorted_by_docs', question.get('relevant_docs', []))
    evidence = []
    for item in documents:
        document = item if isinstance(item, dict) else item[2]
        if not document.get('max_date'):
            continue
        dates = [calendar_date(document['max_date'])]
        dates.extend(calendar_date(document[field]) for field in ('date_publish', 'date_modify') if document.get(field))
        if max(dates) > cutoff:
            continue
        text = document.get('relevant_passage') or document.get('summary')
        if text:
            evidence.append(f"Title: {document.get('title', '')}\nDate: {max(dates).isoformat()}\n{text}")
    return '\n\n'.join(evidence), len(evidence), len(documents) - len(evidence)


async def difficulty(question, engine, judge, trace, cutoff_days=7, threshold=0.5):
    cutoff = calendar_date(question['resolution_date']) - timedelta(days=cutoff_days)
    evidence, included, excluded = historical_evidence(question, cutoff)
    result = await forecast(question, evidence,
                            f"Forecast from the perspective of {cutoff.isoformat()}. Use only information available by this date and the historical evidence below. Do not search or use later information.",
                            engine, judge, trace)
    result.update(passed=result['p_correct'] <= threshold, cutoff=cutoff.isoformat(),
                  evidence_count=included, excluded_evidence_count=excluded)
    return result
