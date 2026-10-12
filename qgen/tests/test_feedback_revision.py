import asyncio
import json

import pytest

from qgen.feedback.io import read_jsonl, write_jsonl
from qgen.feedback.question_generator import FeedbackQuestionGenerator
from qgen.feedback.revision import revise_question
from qgen.feedback.verification import parse_verification
from qgen.feedback.xml_utils import parse_generated_questions
from qgen.scripts.generate_with_feedback import parser


ARTICLE = {'title': 'Prize awarded', 'maintext': 'Ada Vale won the 2024 Aurora Prize.',
           'url': 'https://example.org/prize', 'date_publish': '2024-06-15'}
QUESTION = '''<question id="1">
<question_title>Who will win the 2024 Aurora Prize?</question_title>
<start_date>2024-06-01</start_date>
<background>The Aurora Prize is awarded annually.</background>
<resolution_date>2024-06-15</resolution_date>
<resolution_criteria>The official winner will be announced on June 15, 2024.</resolution_criteria>
<answer>Ada Vale</answer>
</question>'''
CLEAN = '{"violations": [], "explanation": ""}'
REJECTED = '{"violations": ["A1"], "explanation": "The question concerns a minor detail."}'


class Engine:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    async def generate(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        results = []
        for _ in prompts:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            results.append({'response': response})
        return results


def rejected_document(ids_and_violations=((1, ['A1']),)):
    candidates = []
    for number, violations in ids_and_violations:
        report = {'violations': violations, 'explanation': 'Fix these'}
        candidates.append({'candidate_id': number,
                           'question': QUESTION.replace('id="1"', f'id="{number}"'),
                           'verification': report, 'verification_prompt': f'Original check {number}',
                           'verification_response': json.dumps(report)})
    return {**ARTICLE, 'generation_prompt': 'Original document and criteria',
            'generated_questions': QUESTION, 'candidates': candidates}


def test_revision_preserves_both_conversations_and_original_candidate():
    row = rejected_document(((3, ['A1']), (1, ['A1', 'A2']), (2, ['A1'])))
    revised = QUESTION.replace('Who will win', 'Which person will win')
    generation, verification = Engine(revised), Engine(CLEAN)
    saved_statuses = []
    asyncio.run(revise_question(row, generation, verification,
                               lambda: saved_statuses.append(row['candidates'][2]['revision']['status'])))
    candidate = row['candidates'][2]
    revision = candidate['revision']
    assert [c['candidate_id'] for c in row['candidates'] if 'revision' in c] == [2]
    assert saved_statuses == ['generated', 'completed']
    assert revision['question'].startswith('<question id="2">')
    assert 'Which person will win' in revision['question']
    assert revision['verification']['violations'] == []
    assert candidate['verification']['violations'] == ['A1']
    assert candidate['question'] == QUESTION.replace('id="1"', 'id="2"')
    assert generation.prompts[0][:2] == [
        {'role': 'user', 'content': row['generation_prompt']},
        {'role': 'assistant', 'content': row['generated_questions']},
    ]
    assert 'Revise question 2' in generation.prompts[0][2]['content']
    assert 'Fix these' in generation.prompts[0][2]['content']
    assert verification.prompts[0][:2] == [
        {'role': 'user', 'content': candidate['verification_prompt']},
        {'role': 'assistant', 'content': candidate['verification_response']},
    ]
    asyncio.run(revise_question(row, Engine(), Engine()))


@pytest.mark.parametrize('report', [CLEAN, None])
def test_accepted_candidate_or_incomplete_verification_skips_revision(report):
    row = rejected_document(((1, ['A1']), (2, ['A2'])))
    if report is None:
        row['candidates'][1].pop('verification')
        row['candidates'][1]['verification_error'] = 'Transport failed'
    else:
        row['candidates'][1]['verification'] = json.loads(report)
    asyncio.run(revise_question(row, Engine(), Engine()))
    assert all('revision' not in c for c in row['candidates'])


@pytest.mark.parametrize('response', ['', 'malformed XML', RuntimeError('Transport failed')])
def test_failed_revision_is_recorded_and_not_repeated(response):
    row = rejected_document()
    asyncio.run(revise_question(row, Engine(response), Engine()))
    candidate = row['candidates'][0]
    assert candidate['revision']['status'] == 'failed'
    assert candidate['question'] == QUESTION
    asyncio.run(revise_question(row, Engine(), Engine()))


@pytest.mark.parametrize('response,status', [(REJECTED, 'completed'), ('invalid JSON', 'failed')])
def test_reverification_rejects_or_records_error_without_second_attempt(response, status):
    row = rejected_document()
    asyncio.run(revise_question(row, Engine(QUESTION), Engine(response)))
    candidate = row['candidates'][0]
    assert candidate['revision']['status'] == status
    assert candidate['verification']['violations'] == ['A1']
    if status == 'completed':
        assert candidate['revision']['verification']['violations'] == ['A1']
    asyncio.run(revise_question(row, Engine(), Engine()))


def test_interrupted_revision_resumes_only_reverification(tmp_path):
    row = rejected_document()
    row['candidates'][0]['revision'] = {'status': 'generated', 'question': QUESTION}
    path = tmp_path / 'documents.jsonl'
    write_jsonl(path, [row])
    generator = FeedbackQuestionGenerator(Engine(), criteria_engine=Engine(CLEAN))
    result = asyncio.run(generator.run_pipeline([ARTICLE], path, revise=True))
    assert result[0]['candidates'][0]['revision']['status'] == 'completed'
    assert len(generator.inference_engine.prompts) == 0
    assert len(generator.criteria_engine.prompts) == 1


def test_workflow_can_enable_revision_after_initial_run_and_resume_without_calls(tmp_path):
    creator, verifier = Engine(QUESTION, QUESTION), Engine(REJECTED, CLEAN)
    generator = FeedbackQuestionGenerator(creator, criteria_engine=verifier, num_questions=3)
    path = tmp_path / 'feedback' / 'documents.jsonl'
    initial = asyncio.run(generator.run_pipeline([ARTICLE], path))
    assert 'revision' not in initial[0]['candidates'][0]
    result = asyncio.run(generator.run_pipeline([ARTICLE], path, revise=True))
    assert result[0]['candidates'][0]['revision']['verification']['violations'] == []
    assert result[0]['candidates'][0]['question'] == QUESTION
    asyncio.run(generator.run_pipeline([ARTICLE], path, revise=True))
    assert len(creator.prompts) == len(verifier.prompts) == 2
    assert len(read_jsonl(path)) == 1


def test_failed_document_does_not_cancel_other_documents(tmp_path):
    class Creator:
        async def generate(self, prompts, **kwargs):
            if 'Broken' in prompts[0]:
                raise RuntimeError('Transport failed')
            return [{'response': QUESTION}]

    articles = [{**ARTICLE, 'title': 'Broken', 'url': 'https://example.org/broken'}, ARTICLE]
    generator = FeedbackQuestionGenerator(Creator(), criteria_engine=Engine(CLEAN))
    rows = asyncio.run(generator.run_pipeline(articles, tmp_path / 'documents.jsonl', revise=True))
    assert 'Transport failed' in rows[0]['generation_error']
    assert rows[1]['candidates'][0]['verification']['violations'] == []
    assert 'revision' not in rows[1]['candidates'][0]


def test_document_episodes_progress_independently_with_bounded_concurrency(tmp_path):
    async def exercise():
        fast_finished = asyncio.Event()
        events = []
        active = 0
        peak = 0

        class EpisodeEngine:
            async def generate(self, prompts, **kwargs):
                nonlocal active, peak
                prompt = prompts[0]
                if prompt.startswith('check '):
                    name = prompt.removeprefix('check ')
                    events.append(('finished', name))
                    active -= 1
                    if name == 'fast':
                        fast_finished.set()
                    return [{'response': CLEAN}]
                active += 1
                peak = max(peak, active)
                events.append(('started', prompt))
                if prompt == 'slow':
                    await asyncio.wait_for(fast_finished.wait(), timeout=2)
                return [{'response': QUESTION}]

        class Generator(FeedbackQuestionGenerator):
            def format_prompt(self, row):
                return row['title']

            def format_verification_prompt(self, question, row):
                return 'check ' + row['title']

        articles = [{**ARTICLE, 'title': name, 'url': 'https://example.org/' + name}
                    for name in ['slow', 'fast', 'third']]
        rows = await Generator(EpisodeEngine()).run_pipeline(
            articles, tmp_path / 'documents.jsonl', concurrency=2, revise=True)
        assert peak == 2 and active == 0
        assert events.index(('finished', 'fast')) < events.index(('started', 'third'))
        assert all(row['candidates'][0]['verification']['violations'] == [] for row in rows)

    asyncio.run(exercise())


@pytest.mark.parametrize('report', ['{}', '{"violations": ["Z1"], "explanation": "Bad"}',
                                  '{"violations": ["A1", "A1"], "explanation": "Bad"}',
                                  '{"violations": ["A1"], "explanation": ""}'])
def test_invalid_criterion_reports_are_errors(report):
    with pytest.raises(ValueError):
        parse_verification(report)


def test_generation_parser_keeps_questions_separate_and_checks_dates():
    with pytest.raises(ValueError):
        parse_generated_questions(QUESTION.replace('<answer>Ada Vale</answer>', '') +
                                  QUESTION.replace('id="1"', 'id="2"'), 2)
    with pytest.raises(ValueError):
        parse_generated_questions(QUESTION.replace('2024-06-15', '2024-02-30'), 1)


def test_feedback_prompt_and_cli_use_original_numbered_criteria():
    from qgen.feedback.criteria import format_criteria

    generator = FeedbackQuestionGenerator(None, num_questions=3, fewshot=False)
    prompt = generator.format_prompt(ARTICLE)
    verification = generator.format_verification_prompt(QUESTION, ARTICLE)
    assert format_criteria() in prompt and format_criteria() in verification
    assert 'between 1 and 3' in prompt
    assert '<question id="1">' in prompt
    assert 'Bob Dylan' not in prompt
    args = parser().parse_args(['--article_path', 'articles.jsonl', '--output_path', 'feedback.jsonl',
                               '--creator_model', 'creator', '--verifier_model', 'verifier', '--revise'])
    assert args.revise and args.fewshot
