"""Generate questions with criterion reports and an optional feedback revision."""

import argparse
import asyncio
from pathlib import Path

from qgen.feedback.io import load_articles
from qgen.feedback.question_generator import FeedbackQuestionGenerator


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--article_path', type=Path, required=True)
    result.add_argument('--output_path', type=Path, required=True,
                        help='JSONL document records with candidates, reports, and revisions')
    result.add_argument('--creator_model', required=True, help='OpenRouter model slug')
    result.add_argument('--verifier_model', required=True, help='OpenRouter model slug')
    result.add_argument('--num_q_per_article', type=int, default=1)
    result.add_argument('--concurrency', type=int, default=8)
    result.add_argument('--max_tokens', type=int, default=16384)
    result.add_argument('--no_fewshot', action='store_false', dest='fewshot',
                        help='Omit the worked example from generation')
    result.add_argument('--revise', action='store_true',
                        help='Attempt one feedback revision when every candidate has violations')
    return result


async def run(args):
    from qgen.inference.openrouter_inference import OpenRouterInference

    articles = load_articles(args.article_path)
    creator = OpenRouterInference(args.creator_model, max_tokens=args.max_tokens,
                                 concurrency=args.concurrency)
    verifier = OpenRouterInference(args.verifier_model, max_tokens=args.max_tokens,
                                  concurrency=args.concurrency, require_answer_tags=False)
    generator = FeedbackQuestionGenerator(creator, criteria_engine=verifier,
                                          num_questions=args.num_q_per_article, fewshot=args.fewshot)
    rows = await generator.run_pipeline(articles, args.output_path,
                                        concurrency=args.concurrency, revise=args.revise)
    print(f'Saved {len(rows)} document records to {args.output_path}')


def main():
    cli = parser()
    args = cli.parse_args()
    if min(args.num_q_per_article, args.concurrency, args.max_tokens) < 1:
        cli.error('Question count, concurrency, and token budget must be positive')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
