import asyncio
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from qgen.filters import quality_checks as checks
from qgen.scripts.check_questions import check_question


QUESTION = {'question_id': 1, 'question_title': 'Who will win the award in 2025?',
            'background': '', 'resolution_criteria': 'Resolve to the officially announced winner.',
            'answer': 'Example Person', 'question_start_date': '2025-01-01',
            'resolution_date': '2025-03-01', 'article_publish_date': '2025-03-02',
            'article_maintext': 'Example Person was officially announced as the winner.'}


class Engine:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    async def generate(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return [{'response': response} if response is not None else None]


def outcomes(*items):
    return json.dumps({'outcomes': [{'answer': answer, 'probability': probability} for answer, probability in items]})


def dates(answer='Example Person', start='2025-01-01', resolution='2025-03-01'):
    return json.dumps({'answer': answer, 'start_date': start, 'resolution_date': resolution})


@pytest.mark.parametrize('probability,matched,passed', [(0.5, True, False), (0.51, True, True), (0.9, False, False)])
def test_grounding_requires_confident_correct_top_answer(probability, matched, passed):
    engine = Engine(outcomes(('Winner', probability)))
    judge = Engine(f'<answer>{int(matched)}</answer>')
    result = asyncio.run(checks.document_grounding(QUESTION, engine, judge, []))
    assert result['passed'] is passed
    assert 'Reference answer:' not in engine.prompts[0]
    assert QUESTION['article_maintext'] in engine.prompts[0]


def test_grounding_rejects_correct_lower_ranked_answer():
    engine = Engine(outcomes(('Other Person', 0.7), ('Example Person', 0.2)))
    result = asyncio.run(checks.document_grounding(QUESTION, engine, Engine('<answer>0</answer>', '<answer>1</answer>'), []))
    assert result['p_correct'] == 0.2
    assert not result['passed']


def test_date_followup_keeps_history_and_reveals_answer_only_after_mismatch():
    engine = Engine(dates('Wrong Person'), dates(resolution='2025-02-28'))
    result = asyncio.run(checks.correct_dates(QUESTION, engine, Engine('<answer>0</answer>', '<answer>1</answer>'), []))
    assert result['passed']
    assert result['resolution_date'] == '2025-02-28'
    assert result['original_resolution_date'] == '2025-03-01'
    assert 'Example Person' not in engine.prompts[0][0]['content']
    assert engine.prompts[1][1]['role'] == 'assistant'
    assert 'Example Person' in engine.prompts[1][2]['content']
    assert QUESTION['resolution_criteria'] in engine.prompts[0][0]['content']


def test_date_format_retry_is_bounded():
    engine = Engine(dates(start='invalid'), dates(start='2025-03-02'))
    result = asyncio.run(checks.correct_dates(QUESTION, engine, Engine(), []))
    assert result['status'] == 'invalid_response'
    assert not result['passed']
    assert len(engine.prompts) == 2


def test_date_retry_can_recover_and_no_publication_upper_bound_is_imposed():
    engine = Engine(dates(start='invalid'), dates(resolution='2025-03-10'))
    result = asyncio.run(checks.correct_dates(QUESTION, engine, Engine('<answer>1</answer>'), []))
    assert result['passed']
    assert result['resolution_date'] == '2025-03-10'


def test_date_answer_mismatch_after_followup_is_unresolved():
    result = asyncio.run(checks.correct_dates(QUESTION, Engine(dates('Wrong'), dates('Still Wrong')),
                                            Engine('<answer>0</answer>', '<answer>0</answer>'), []))
    assert result['status'] == 'unresolved'
    assert not result['passed']


def test_historical_evidence_excludes_later_modified_and_undated_documents():
    timestamp = datetime(2025, 2, 22, tzinfo=timezone.utc).timestamp()
    question = {**QUESTION, 'relevant_articles_sorted_by_docs': [
        ['0.9', 'early', {'max_date': timestamp, 'relevant_passage': 'EARLY'}],
        ['0.8', 'late', {'max_date': '2025-02-23', 'relevant_passage': 'LATE'}],
        ['0.7', 'modified', {'max_date': '2025-02-01', 'date_modify': '2025-02-25', 'relevant_passage': 'MODIFIED'}],
        ['0.6', 'undated', {'relevant_passage': 'UNDATED'}]]}
    text, included, excluded = checks.historical_evidence(question, date(2025, 2, 22))
    assert 'EARLY' in text
    assert all(word not in text for word in ('LATE', 'MODIFIED', 'UNDATED'))
    assert (included, excluded) == (1, 3)


def test_evidence_dates_are_compared_in_utc():
    question = {**QUESTION, 'relevant_docs': [
        {'max_date': '2025-02-22T23:00:00-05:00', 'relevant_passage': 'TOO LATE IN UTC'}]}
    assert checks.historical_evidence(question, date(2025, 2, 22)) == ('', 0, 1)


def test_missing_retrieval_is_an_error_before_forecasting():
    args = SimpleNamespace(grounding=False, fix_dates=False, difficulty=True, cutoff_days=7, difficulty_threshold=0.5)
    engine = Engine()
    _, results = asyncio.run(check_question(QUESTION, args, engine, Engine(), None))
    assert results['difficulty']['status'] == 'error'
    assert not engine.prompts


@pytest.mark.parametrize('probability,passed', [(0.5, True), (0.51, False)])
def test_difficulty_boundary_and_no_label_or_source_leakage(probability, passed):
    question = {**QUESTION, 'relevant_docs': []}
    engine = Engine(outcomes(('Winner', probability)))
    result = asyncio.run(checks.difficulty(question, engine, Engine('<answer>1</answer>'), []))
    assert result['cutoff'] == '2025-02-22'
    assert result['passed'] is passed
    assert 'Example Person' not in engine.prompts[0]
    assert '2025-03-02' not in engine.prompts[0]
    assert result['evidence_count'] == 0


def test_successful_forecast_with_no_matching_answer_has_zero_probability():
    result = asyncio.run(checks.difficulty({**QUESTION, 'relevant_docs': []}, Engine(outcomes(('Other', 0.9))),
                                         Engine('<answer>0</answer>'), []))
    assert result['p_correct'] == 0.0
    assert result['passed']


@pytest.mark.parametrize('response', [None, RuntimeError('transport failure'), 'invalid JSON'])
def test_failures_are_errors_not_zero_probability(response):
    args = SimpleNamespace(grounding=False, fix_dates=False, difficulty=True, cutoff_days=7, difficulty_threshold=0.5)
    _, results = asyncio.run(check_question({**QUESTION, 'relevant_docs': []}, args, Engine(response), Engine(), None))
    assert results['difficulty']['status'] == 'error'
    assert 'p_correct' not in results['difficulty']
    assert not results['difficulty']['passed']


def test_grounding_rejection_stops_later_checks():
    args = SimpleNamespace(grounding=True, fix_dates=True, difficulty=True)
    date_engine = Engine()
    _, results = asyncio.run(check_question(QUESTION, args, Engine(outcomes(('Other', 0.9))),
                                           Engine('<answer>0</answer>'), date_engine))
    assert list(results) == ['grounding']
    assert not date_engine.prompts


def test_difficulty_uses_corrected_dates():
    args = SimpleNamespace(grounding=False, fix_dates=True, difficulty=True, cutoff_days=7, difficulty_threshold=0.5)
    engine = Engine(outcomes(('Other', 0.9)))
    corrected, results = asyncio.run(check_question({**QUESTION, 'relevant_docs': []}, args, engine,
                                                    Engine('<answer>1</answer>', '<answer>0</answer>'),
                                                    Engine(dates(start='2025-01-02', resolution='2025-03-10'))))
    assert corrected['question_start_date'] == '2025-01-02'
    assert results['difficulty']['cutoff'] == '2025-03-03'


@pytest.mark.parametrize('response', [outcomes(('A', 1.1)), outcomes(('A', float('nan'))),
                                     outcomes(('A', 0.7), ('B', 0.7)), outcomes(('A', True)), outcomes(('A', 0.3), ('a', 0.3))])
def test_invalid_probabilities_or_duplicate_outcomes_are_rejected(response):
    with pytest.raises(ValueError):
        checks.parse_outcomes(response)


def test_optional_runner_writes_corrected_questions_and_error_reports(tmp_path, monkeypatch):
    from qgen.inference import openrouter_inference
    from qgen.scripts.check_questions import parser, run

    class Client:
        def __init__(self, model, **kwargs):
            self.model = model

        async def generate(self, prompts, **kwargs):
            prompt = prompts[0]
            if self.model.endswith(':online'):
                response = dates(start='2025-01-02', resolution='2025-03-10')
            elif 'Reference answer:' in prompt:
                response = '<answer>1</answer>' if 'Proposed answer: "Example Person"' in prompt else '<answer>0</answer>'
            elif 'DOCUMENT FAILURE' in prompt:
                raise RuntimeError('provider unavailable')
            elif 'explicitly stated in the provided document' in prompt:
                response = outcomes(('Example Person', 0.8))
            else:
                response = outcomes(('Other', 0.8))
            return [{'response': response}]

    monkeypatch.setattr(openrouter_inference, 'OpenRouterInference', Client)
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({**QUESTION, 'relevant_docs': []}) + '\n' +
                      json.dumps({**QUESTION, 'question_id': 2, 'article_maintext': 'DOCUMENT FAILURE'}) + '\n')
    output = tmp_path / 'checks'
    args = parser().parse_args(['--input', str(source), '--output_dir', str(output),
                               '--model', 'example/offline', '--date_model', 'example/offline:online',
                               '--grounding', '--fix_dates', '--difficulty'])
    asyncio.run(run(args))
    accepted = [json.loads(line) for line in (output / 'questions.jsonl').read_text().splitlines()]
    reports = [json.loads(line) for line in (output / 'checks.jsonl').read_text().splitlines()]
    assert len(accepted) == 1
    assert accepted[0]['question_id'] == 1
    assert accepted[0]['question_start_date'] == '2025-01-02'
    assert accepted[0]['resolution_date'] == '2025-03-10'
    assert reports[0]['checks']['difficulty']['cutoff'] == '2025-03-03'
    assert reports[1]['checks']['grounding']['status'] == 'error'
    assert (output / 'config.json').exists()
