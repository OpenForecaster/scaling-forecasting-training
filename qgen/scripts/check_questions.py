"""Run optional quality checks on standardized questions without regenerating."""

import argparse
import asyncio
import json
from pathlib import Path

from qgen.filters import quality_checks
from qgen.utils.conversion_utils import standardize_entry_format


async def check_question(question, args, engine, judge, date_engine):
    question = standardize_entry_format(question)
    checks = {}
    corrected = dict(question)
    for name, enabled in [('grounding', args.grounding), ('dates', args.fix_dates), ('difficulty', args.difficulty)]:
        if not enabled:
            continue
        trace = []
        try:
            if name == 'grounding':
                result = await quality_checks.document_grounding(corrected, engine, judge, trace)
            elif name == 'dates':
                result = await quality_checks.correct_dates(corrected, date_engine, judge, trace)
                if result['passed']:
                    corrected.update(question_start_date=result['start_date'], resolution_date=result['resolution_date'])
            else:
                result = await quality_checks.difficulty(corrected, engine, judge, trace,
                                                         args.cutoff_days, args.difficulty_threshold)
        except Exception as error:
            result = {'status': 'error', 'passed': False, 'error': f'{type(error).__name__}: {error}'}
        checks[name] = {**result, 'trace': trace}
        if not result['passed']:
            break
    return corrected, checks


async def run(args):
    from qgen.inference.openrouter_inference import OpenRouterInference

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty output directory to avoid overwriting earlier results')
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    engine = OpenRouterInference(model=args.model, max_tokens=args.max_tokens, concurrency=args.concurrency)
    judge = OpenRouterInference(model=args.judge_model or args.model, max_tokens=args.max_tokens, concurrency=args.concurrency)
    date_engine = None
    if args.fix_dates:
        date_engine = OpenRouterInference(model=args.date_model, max_tokens=args.max_tokens, concurrency=args.concurrency)
    questions = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    slots = asyncio.Semaphore(args.concurrency)

    async def episode(index, question):
        async with slots:
            corrected, checks = await check_question(question, args, engine, judge, date_engine)
            passed = all(result['passed'] for result in checks.values())
            record = {'input_index': index, 'question_id': question.get('question_id'),
                      'passed': passed, 'checks': checks}
            with (output / 'checks.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            return corrected if passed else None

    results = await asyncio.gather(*(episode(index, question) for index, question in enumerate(questions)))
    kept = [question for question in results if question is not None]
    with (output / 'questions.jsonl').open('w') as stream:
        for question in kept:
            stream.write(json.dumps(question) + '\n')
    print(f'Kept {len(kept)}/{len(questions)} questions; results: {output}')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--input', type=Path, required=True)
    result.add_argument('--output_dir', type=Path, required=True)
    result.add_argument('--model', required=True, help='OpenRouter model without live search, for grounding/difficulty')
    result.add_argument('--judge_model', help='Semantic answer judge; defaults to --model')
    result.add_argument('--date_model', help='Search-enabled OpenRouter model slug ending in :online')
    result.add_argument('--grounding', action='store_true')
    result.add_argument('--fix_dates', action='store_true')
    result.add_argument('--difficulty', action='store_true')
    result.add_argument('--cutoff_days', type=int, default=7)
    result.add_argument('--difficulty_threshold', type=float, default=0.5)
    result.add_argument('--concurrency', type=int, default=8)
    result.add_argument('--max_tokens', type=int, default=16384)
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if not (args.grounding or args.fix_dates or args.difficulty):
        cli.error('Enable at least one of --grounding, --fix_dates, or --difficulty')
    if args.fix_dates and (not args.date_model or not args.date_model.endswith(':online')):
        cli.error('--fix_dates requires --date_model with an :online search-enabled slug')
    if ':online' in args.model or ':online' in (args.judge_model or ''):
        cli.error('Grounding, difficulty, and answer judging require models without live search')
    if args.concurrency < 1 or args.max_tokens < 1 or args.cutoff_days < 1:
        cli.error('Concurrency, token budget, and cutoff days must be positive')
    if not 0 <= args.difficulty_threshold <= 1:
        cli.error('--difficulty_threshold must be between zero and one')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
