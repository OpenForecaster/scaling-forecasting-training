import os
import argparse
import asyncio
import logging
from typing import List, Dict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from qgen.article_processor import ArticleProcessor
from qgen.question_generator import ForecastingQuestionGenerator
# from qgen.inference.vllm_inference import VLLMInference
from qgen.inference.openrouter_inference import OpenRouterInference

async def main():
    parser = argparse.ArgumentParser(description="Generate forecasting questions from news articles")
    
    # Article input
    parser.add_argument(
        "--article_path", 
        type=str, 
        # default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.reuters.com_selected100.jsonl",
        default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.dw.com_2025-05_selected12.jsonl",
        # default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.dw.com_2025-05_selected30.jsonl",
        # default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.theguardian.com_2025-05_selected18.jsonl",
        # default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.reuters.com_2025-05_selected24.jsonl",
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
        "--openrouter_model", 
        type=str, 
        default="deepseek/deepseek-chat-v3-0324",
        help="OpenRouter model to use"
    )
    
    # Generation parameters
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
        default="debug/generated_questions.jsonl",
        help="Path to save generated questions"
    )
    
    # Add regenerate argument
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Ignore existing results in output_path and regenerate all questions."
    )
    
    parser.add_argument(
        "--num_q", 
        type=int, 
        default=1,
        help="Number of questions to generate per article"
    )
    # Add freeq argument
    parser.add_argument(
        "--freeq",
        action="store_true",
        help="Generate free-form short answer questions instead of multiple choice questions."
    )
    
    args = parser.parse_args()
    
    model_requested = args.openrouter_model if args.use_openrouter else args.model_path
    if args.use_openrouter:
        if "v3" in model_requested:
            model_requested = "deepseek/deepseek-chat-v3-0324"
        elif "r1" in model_requested:
            model_requested = "deepseek/deepseek-r1-0528"
        elif "o4-mini" in model_requested:
            model_requested = "openai/o4-mini-high"
        elif "grok3-mini" in model_requested:
            model_requested = "x-ai/grok-3-mini"
        elif "k2" in model_requested:
            model_requested = "moonshotai/kimi-k2"
        elif "qwen3" in model_requested:
            model_requested = "qwen/qwen3-32b"
        elif "llama-4-maverick" in model_requested:
            model_requested = "meta-llama/llama-4-maverick"
        elif "llama-4-scout" in model_requested:
            model_requested = "meta-llama/llama-4-scout"
    
    # Load articles
    processor = ArticleProcessor(args.article_path)
    articles = processor.load_articles()
    num_questions = args.num_q
    
    # Initialize inference engine
    if args.use_openrouter:
        inference_engine = OpenRouterInference(
            model=model_requested,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        leakage_engine = OpenRouterInference(
            model="meta-llama/llama-4-scout",
            max_tokens=args.max_tokens,
            temperature=0.6
        )
        choose_engine = OpenRouterInference(
            # model="deepseek/deepseek-r1-0528",
            # model="qwen/qwen3-32b",
            model="meta-llama/llama-4-maverick",
            max_tokens=args.max_tokens,
            temperature=0.6
        )
        leakage_engine = choose_engine
    else:
        if not args.model_path:
            raise ValueError("model_path must be specified when not using OpenRouter")
        
        inference_engine = VLLMInference(
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
    
    # Initialize question generator
    generator = ForecastingQuestionGenerator(
        inference_engine=inference_engine,
        use_freeq=args.freeq,
        check_leakage=args.check_leakage,
        leakage_engine=leakage_engine,
        choose_engine=choose_engine,
        choose_best=args.choose_best,
        num_questions=num_questions,
        validate_questions=args.validate
    )
    
    output_path = args.output_path
    # Check if default 
    if output_path == "debug/generated_questions.jsonl":
        prefix = "/fast/sgoel/forecasting/news/tokenized_data/news/deduped/recent/"
        # prefix = "/fast/nchandak/forecasting/newsdata/theguardian/"
        prefix = "/fast/nchandak/forecasting/newsdata/theguardian/2025/tillMarch/"
        prefix = "/fast/nchandak/forecasting/newsdata/testset/"
        
        # if prefix not in args.article_path:
        #     prefix = "debug/"
        # else :
        #     prefix += "qgen/"
            
        prefix += "qgen/"
        
        model_name = model_requested.split("/")[-1]
        news_source = args.article_path.split("/")[-1].split(".")[1]
        num_articles_selected = args.article_path.split(".")[-2].split("selected")[1]
        qtype = "free" if args.freeq else "mcq"
        output_path = f"{prefix}{model_name}_{news_source}_{num_articles_selected}_{qtype}_{num_questions}.jsonl"

    logger.info(f"Output path: {output_path}")

    # Generate questions, passing output_path and regenerate flag
    # The results are saved incrementally within this function now.
    results = await generator.run_pipeline(
        articles,
        output_path=output_path,
        batch_size=args.batch_size,
        regenerate=args.regenerate
    )
    
    # Optional: Final save call (can be removed as saving is incremental)
    # generator.save_results(results, args.output_path)
    logger.info("Question generation process complete.")


if __name__ == "__main__":
    asyncio.run(main())