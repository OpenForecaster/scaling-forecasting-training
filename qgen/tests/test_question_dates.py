import asyncio

from qgen.qgen_core.question_generator import ForecastingQuestionGenerator
from qgen.utils.conversion_utils import convert_question_format
from qgen.utils.xml_utils import extract_final_question


QUESTION = """<q1>
<question_id>0</question_id>
<question_title>Who will win the award in 2025?</question_title>
<start_date>2025-01-01</start_date>
<background></background>
<resolution_date>2025-03-01</resolution_date>
<resolution_criteria>Resolve to the winner named in the official announcement.</resolution_criteria>
<answer>Example Person</answer>
<answer_type>string (name)</answer_type>
</q1>"""


def test_dates_survive_selection_extraction_and_conversion():
    generator = ForecastingQuestionGenerator(None, use_freeq=True)
    final = generator.extract_final_question(QUESTION)
    converted = convert_question_format({'final_question': final, 'date_publish': '2025-03-02'})
    assert converted['question_start_date'] == '2025-01-01'
    assert converted['resolution_date'] == '2025-03-01'
    assert converted['background'] == ''
    assert converted['article_publish_date'] == '2025-03-02'


def test_legacy_question_without_date_tags_still_parses():
    old = QUESTION.replace('<start_date>2025-01-01</start_date>', '').replace('<resolution_date>2025-03-01</resolution_date>', '')
    converted = convert_question_format({'final_question': extract_final_question(old)})
    assert converted['answer'] == 'Example Person'
    assert 'question_start_date' not in converted
    assert 'resolution_date' not in converted


def test_last_question_fields_are_not_combined_with_previous_question():
    incomplete = QUESTION.replace('<answer>Example Person</answer>', '')
    assert extract_final_question(QUESTION + incomplete) == ''


def test_invalid_or_incomplete_explicit_dates_are_rejected():
    assert extract_final_question(QUESTION.replace('2025-03-01', '2025-02-30')) == ''
    assert extract_final_question(QUESTION.replace('2025-03-01', '2025-01-01')) == ''
    assert extract_final_question(QUESTION.replace('<start_date>2025-01-01</start_date>', '')) == ''
    assert extract_final_question(QUESTION.replace('</background>', '')) == ''


def test_generation_uses_publication_date_and_one_template():
    article = {'title': 'Award announced', 'maintext': 'The winner was named.',
               'date_publish': '2025-03-02', 'date_modify': '2025-03-03', 'date_download': '2025-03-04'}
    for count in (1, 3):
        generator = ForecastingQuestionGenerator(None, use_freeq=True, num_questions=count)
        prompt = generator.format_prompt(article)
        assert f'between 1 and {count}' in prompt
        assert 'Published Date: 2025-03-02' in prompt
        assert '2025-03-03' not in prompt and '2025-03-04' not in prompt
        assert '<start_date>[YYYY-MM-DD]</start_date>' in prompt
        assert '<resolution_date>[YYYY-MM-DD]</resolution_date>' in prompt
        assert 'Bob Dylan' not in prompt
        assert len(generator._extract_questions_from_text(QUESTION)) == 1
        generator.fewshot = True
        assert 'Bob Dylan' in generator.format_prompt(article)


def test_validation_stays_binary_and_rewriters_preserve_dates():
    generator = ForecastingQuestionGenerator(None, use_freeq=True)
    validation = generator._format_question_validation_prompt(QUESTION, {'maintext': 'Example Person won.'})
    assert '<answer>0/1</answer>' in validation
    assert '<start_date>' in generator._format_choose_best_prompt(QUESTION, {})
    assert '<resolution_date>' in generator._format_freeq_leakage_prompt(QUESTION, '')


def test_existing_pipeline_accepts_fewer_candidates_and_resumes_without_calls(tmp_path):
    class Engine:
        def __init__(self, *responses):
            self.responses = iter(responses)
            self.prompts = []

        async def generate(self, prompts, **kwargs):
            self.prompts.extend(prompts)
            return [{'response': next(self.responses)} for _ in prompts]

    generation = Engine(QUESTION)
    selection = Engine('<answer>1</answer>', QUESTION)
    leakage = Engine(QUESTION)
    generator = ForecastingQuestionGenerator(generation, use_freeq=True, num_questions=3,
                                             validate_questions=True, choose_best=True,
                                             check_leakage=True, choose_engine=selection,
                                             leakage_engine=leakage)
    articles = [{'title': 'Award announced', 'maintext': 'Example Person won.',
                 'date_publish': '2025-03-02', 'url': 'https://example.org/award'}]
    path = str(tmp_path / 'generated.jsonl')
    result = asyncio.run(generator.run_pipeline(articles, path))
    assert result[0]['q1_valid'] == 1
    assert result[0]['individual_validation_done'] == 1
    assert result[0]['choose_best'] == 1
    assert result[0]['leakage_check'] == 1
    converted = convert_question_format(result[0])
    assert converted['question_start_date'] == '2025-01-01'
    assert converted['resolution_date'] == '2025-03-01'
    asyncio.run(generator.run_pipeline(articles, path))
    assert [len(engine.prompts) for engine in (generation, selection, leakage)] == [1, 2, 1]
