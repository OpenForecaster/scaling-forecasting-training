#!/usr/bin/env python3
"""
Generate Forecasting Questions - CLI entry point for question generation.

This script provides a command-line interface for generating forecasting questions
from news articles. 

Usage:
    python scripts/generate_questions.py \\
        --article_path /path/to/articles.jsonl \\
        --output_path generated_questions.jsonl \\
        --freeq --check_leakage --choose_best --validate \\
        --use_openrouter --openrouter_model deepseek/deepseek-chat-v3-0324
"""

import os
import argparse
import asyncio
import logging
import sys

# Add parent of qgen directory to path so we can import qgen package
qgen_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(qgen_dir)
sys.path.insert(0, parent_dir)

from qgen.qgen_core.article_processor import ArticleProcessor
from qgen.qgen_core.question_generator import ForecastingQuestionGenerator
from qgen.inference.openrouter_inference import OpenRouterInference

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Generate forecasting questions from news articles")
    
    # Article input
    parser.add_argument(
        "--article_path", 
        type=str, 
        required=True,
        help="Path to news article file or directory"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to local HuggingFace model for vLLM inference"
    )
    parser.add_argument(
        "--use_openrouter", 
        action="store_true",
        help="Use OpenRouter for inference"
    )
    parser.add_argument(
        "--openrouter_model", 
        type=str, 
        default="deepseek/deepseek-chat-v3-0324",
        # default="deepseek/deepseek-v3.2",
        help="OpenRouter model to use"
    )
    
    # Processing flags
    parser.add_argument(
        "--check_leakage", 
        action="store_true",
        help="Check leakage of answer in the generated questions"
    )
    parser.add_argument(
        "--choose_best", 
        action="store_true",
        help="Choose the best question from the generated questions"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate the generated questions"
    )
    parser.add_argument(
        "--freeq",
        action="store_true",
        help="Generate free-form short answer questions instead of MCQ"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_q", 
        type=int, 
        default=1,
        help="Number of questions to generate per article"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=16384,
        help="Maximum tokens for generation"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1000,
        help="Batch size for parallel processing"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save generated questions (default: same location as article_path with '_generated_questions.jsonl' suffix)"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Ignore existing results and regenerate all questions"
    )
    
    args = parser.parse_args()
    
    # If no output path specified, generate it from article_path
    if args.output_path is None:
        article_dir = os.path.dirname(args.article_path)
        article_basename = os.path.basename(args.article_path)
        article_name, _ = os.path.splitext(article_basename)
        output_filename = f"{article_name}_generated_questions.jsonl"
        args.output_path = os.path.join(article_dir, output_filename) if article_dir else output_filename
    
    # Load articles
    processor = ArticleProcessor(args.article_path)
    articles = processor.load_articles()
    
    # Initialize inference engines
    if args.use_openrouter:
        inference_engine = OpenRouterInference(
            model=args.openrouter_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        leakage_engine = OpenRouterInference(
            model="meta-llama/llama-4-maverick",
            max_tokens=args.max_tokens,
            temperature=0.6
        )
        choose_engine = OpenRouterInference(
            model="meta-llama/llama-4-maverick",
            max_tokens=args.max_tokens,
            temperature=0.6
        )
    else:
        if not args.model_path:
            raise ValueError("model_path must be specified when not using OpenRouter")
        from qgen.inference.vllm_inference import VLLMInference
        inference_engine = VLLMInference(
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        leakage_engine = inference_engine
        choose_engine = inference_engine
    
    # Initialize question generator
    generator = ForecastingQuestionGenerator(
        inference_engine=inference_engine,
        use_freeq=args.freeq,
        check_leakage=args.check_leakage,
        leakage_engine=leakage_engine,
        choose_engine=choose_engine,
        choose_best=args.choose_best,
        num_questions=args.num_q,
        validate_questions=args.validate
    )
    
    logger.info(f"Output path: {args.output_path}")
    
    # Generate questions
    results = await generator.run_pipeline(
        articles,
        output_path=args.output_path,
        batch_size=args.batch_size,
        regenerate=args.regenerate
    )
    
    logger.info("Question generation process complete.")


if __name__ == "__main__":
    asyncio.run(main())


