#!/usr/bin/env python3
import re
import json
import os
import sys
import logging
import asyncio
import argparse
import glob
import time
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from datetime import datetime

# Import the existing OpenRouter inference engine
sys.path.append('/home/nchandak/forecasting')
from qgen.inference.openrouter_inference import OpenRouterInference

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def extract_probability(completion: str) -> Optional[float]:
    """
    Extracts the probability from the LLM's output.
    Returns the probability as a float.
    """
    # Check if completion is None or not a string
    if completion is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(completion, str):
        completion = str(completion)
    
    # Check if completion is empty after conversion
    if not completion.strip():
        return None
    
    # Handle thinking tags
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    
    try:
        matches = re.finditer(r"<probability>(.*?)<\/probability>", completion, re.DOTALL)
        matches_list = list(matches)
    except:
        return None

    if not matches_list:
        return None
    
    # Get the last match
    last_match = matches_list[-1]
    probability_text = last_match.group(1).strip()

    # Try to parse probability as float
    try:
        probability = float(probability_text)
        return probability
    except (ValueError, TypeError):
        return None

def format_forecasting_prompt_binary(
    question_title: str,
    resolution_criteria: str,
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a binary forecasting question.  You have to come up with the best probability estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {question_title}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""

    return prompt

def load_paleka_questions_from_jsonl(file_path: str) -> List[dict]:
    """
    Load questions from a Paleka JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries with question data
    """
    questions_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    # Store the original line data and add an index
                    question_entry = {
                        'idx': line_idx,
                        'original_data': data,
                        'file_path': file_path
                    }
                    questions_data.append(question_entry)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx} in {file_path}: {e}")
                    continue
    
    logger.info(f"Loaded {len(questions_data)} questions from {file_path}")
    return questions_data

def load_existing_results(output_file: str) -> Dict[str, Dict]:
    """Load existing results from JSONL file if it exists."""
    existing_results = {}
    
    if os.path.exists(output_file):
        logger.info(f"Found existing results file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line.strip())
                        # Create a unique key for each question component
                        file_idx = result.get('idx', '')
                        component_key = result.get('component_key', '')
                        result_key = f"{file_idx}_{component_key}"
                        existing_results[result_key] = result
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(existing_results)} existing results")
    
    return existing_results

def save_results_incrementally(results: List[Dict], output_file: str):
    """Save results to JSONL file."""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

async def evaluate_paleka_questions(
    model_name: str,
    questions_data: List[dict],
    output_file: str,
    max_tokens: int = 8192,
    num_generations: int = 1,
    batch_size: int = 5,
) -> List[dict]:
    """
    Run inference on Paleka questions using OpenRouter API and return results in the required format
    """
    # Load existing results
    existing_results = load_existing_results(output_file)
    
    # Initialize the inference engine
    inference_engine = OpenRouterInference(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.6  # Match the original implementation
    )
    
    # First, prepare all prompts and metadata
    all_prompts = []
    prompt_metadata = []  # Store info about each prompt for later mapping
    
    # Process each question's original data structure
    for question_entry in questions_data:
        original_data = question_entry['original_data']
        file_path = question_entry['file_path']
        file_idx = question_entry['idx']
        
        # Process each component in the original data
        for component_key, component_data in original_data.items():
            if isinstance(component_data, dict) and 'title' in component_data:
                question_title = component_data.get('title', '')
                resolution_criteria = component_data.get('body', 'N/A')
                
                # Check if this component already has results
                result_key = f"{file_idx}_{component_key}"
                if result_key in existing_results:
                    existing_result = existing_results[result_key]
                    existing_responses = existing_result.get("response", [])
                    
                    # Check if we have all required generations
                    if len(existing_responses) >= num_generations:
                        valid_responses = sum(1 for resp in existing_responses if resp and resp.strip() and "<probability" in resp)
                        if valid_responses >= num_generations:
                            continue  # Skip this component, it's already complete
                
                if question_title:
                    # Create prompt
                    prompt = format_forecasting_prompt_binary(
                        question_title=question_title, 
                        resolution_criteria=resolution_criteria
                    )
                    
                    # Add for each generation
                    for gen_idx in range(num_generations):
                        all_prompts.append(prompt)
                        
                        # Store metadata for this prompt
                        prompt_metadata.append({
                            'question_entry_idx': question_entry['idx'],
                            'component_key': component_key,
                            'question_title': question_title,
                            'resolution_criteria': resolution_criteria,
                            'file_path': file_path,
                            'original_data_idx': questions_data.index(question_entry),
                            'gen_idx': gen_idx,
                            'result_key': result_key
                        })
    
    if not all_prompts:
        logger.warning("No new prompts to process!")
        # Return existing results in the required format
        results_by_question = {}
        for result_key, existing_result in existing_results.items():
            file_idx, component_key = result_key.split('_', 1)
            file_idx = int(file_idx)
            
            if file_idx not in results_by_question:
                results_by_question[file_idx] = {
                    "line": {},
                    "original_file": existing_result.get("original_file", ""),
                    "idx": file_idx
                }
            
            # Reconstruct the forecast data
            responses = existing_result.get("response", [])
            if responses:
                prob = extract_probability(responses[0])  # Use first generation
                if prob is None:
                    prob = 0.5
                
                results_by_question[file_idx]["line"][component_key] = {
                    "question": {
                        "title": existing_result.get("question_title", "")
                    },
                    "forecast": {
                        "prob": prob
                    }
                }
        
        return list(results_by_question.values())
    
    logger.info(f"Starting generation with OpenRouter for {len(all_prompts)} prompts")
    start_time = time.time()
    
    # Group results by question component
    component_results = {}
    
    # Initialize component_results with existing data
    for result_key, existing_result in existing_results.items():
        component_results[result_key] = {
            "responses": existing_result.get("response", []),
            "prompt_tokens": existing_result.get("prompt_tokens", []),
            "completion_tokens": existing_result.get("completion_tokens", []),
            "reasoning": existing_result.get("reasoning", []),
            "metadata": {
                'question_title': existing_result.get("question_title", ""),
                'original_file': existing_result.get("original_file", ""),
                'component_key': existing_result.get("component_key", ""),
                'file_idx': existing_result.get("idx", 0)
            }
        }
    
    # Process missing prompts in batches
    for batch_start in tqdm(range(0, len(all_prompts), batch_size), desc=f"Processing {model_name}"):
        batch_end = min(batch_start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_metadata = prompt_metadata[batch_start:batch_end]
        
        # Generate completions for this batch
        batch_completions = await inference_engine.generate(
            prompts=batch_prompts,
            batch_size=batch_size
        )
        
        # Process batch results
        for metadata, completion in zip(batch_metadata, batch_completions):
            result_key = metadata['result_key']
            gen_idx = metadata['gen_idx']
            
            if result_key not in component_results:
                component_results[result_key] = {
                    "responses": [],
                    "prompt_tokens": [],
                    "completion_tokens": [],
                    "reasoning": [],
                    "metadata": {
                        'question_title': metadata['question_title'],
                        'original_file': os.path.basename(metadata['file_path']),
                        'component_key': metadata['component_key'],
                        'file_idx': metadata['question_entry_idx']
                    }
                }
            
            # Handle None completions (failed requests)
            if completion is None:
                response = ""
                prompt_tokens = 0
                completion_tokens = 0
                reasoning = ""
            else:
                response = completion['response']
                prompt_tokens = completion['prompt_tokens']
                completion_tokens = completion['completion_tokens']
                reasoning = completion['reasoning']
            
            # Ensure we have the right number of slots
            while len(component_results[result_key]["responses"]) <= gen_idx:
                component_results[result_key]["responses"].append("")
                component_results[result_key]["prompt_tokens"].append(0)
                component_results[result_key]["completion_tokens"].append(0)
                component_results[result_key]["reasoning"].append("")
                
            # Store the result at the correct generation index
            component_results[result_key]["responses"][gen_idx] = response
            component_results[result_key]["prompt_tokens"][gen_idx] = prompt_tokens
            component_results[result_key]["completion_tokens"][gen_idx] = completion_tokens
            component_results[result_key]["reasoning"][gen_idx] = reasoning
        
        # Save progress after each batch (following binary_evals.py pattern)
        current_results = []
        for result_key, data in component_results.items():
            metadata = data["metadata"]
            
            result = {
                "model": model_name,
                "split": "eval",
                "data_type": "paleka",
                "idx": metadata['file_idx'],
                "component_key": metadata['component_key'],
                "original_file": metadata['original_file'],
                "question_title": metadata['question_title'],
                "response": data["responses"],
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "reasoning": data["reasoning"],
            }
            
            current_results.append(result)
        
        # Save incremental results to the same file
        save_results_incrementally(current_results, output_file)
        logger.info(f"Saved progress: {len(current_results)} results to {output_file}")
        
        # Small delay between batches
        await asyncio.sleep(1)
    
    end_time = time.time()
    logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Group results by original question entry for final output format
    results_by_question = {}
    extraction_success = 0
    total_outputs = 0
    
    # Process all component results
    for result_key, data in component_results.items():
        metadata = data["metadata"]
        file_idx = metadata['file_idx']
        component_key = metadata['component_key']
        question_title = metadata['question_title']
        original_file = metadata['original_file']
        
        # Extract probability from the first generation
        responses = data["responses"]
        if responses and responses[0]:
            generated_text = responses[0]
            total_outputs += 1
            
            # Process the response to extract probability
            if "</think>" in generated_text:
                generated_text = generated_text.split("</think>")[1]
            
            prob = extract_probability(generated_text)
            if prob is None:
                prob = 0.5  # Default probability if extraction fails
            else:
                extraction_success += 1
        else:
            prob = 0.5
            total_outputs += 1
            
        # Initialize result structure for this question if not exists
        if file_idx not in results_by_question:
            results_by_question[file_idx] = {
                "line": {},
                "original_file": original_file,
                "idx": file_idx
            }
        
        # Add forecast for this component
        results_by_question[file_idx]["line"][component_key] = {
            "question": {
                "title": question_title
            },
            "forecast": {
                "prob": prob
            }
        }
    
    # Convert to list and filter out empty results
    all_results = []
    for file_idx in sorted(results_by_question.keys()):
        result = results_by_question[file_idx]
        if result["line"]:  # Only add if we have at least one component
            all_results.append(result)
    
    if total_outputs > 0:
        logger.info(f"Extraction success rate: {extraction_success / total_outputs * 100:.2f}%")
    
    return all_results

async def process_paleka_directory(
    data_dir: str,
    model_name: str,
    output_dir: str,
    max_tokens: int = 8192,
    num_generations: int = 1,
    batch_size: int = 5,
):
    """
    Process all JSONL files in a directory
    """
    # Find all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    
    if not jsonl_files:
        logger.error(f"No JSONL files found in {data_dir}")
        return
    
    logger.info(f"Found {len(jsonl_files)} JSONL files to process")
    
    for jsonl_file in jsonl_files:
        logger.info(f"Processing file: {jsonl_file}")
        
        # Load questions from this file
        questions_data = load_paleka_questions_from_jsonl(jsonl_file)
        
        if not questions_data:
            logger.warning(f"No questions found in {jsonl_file}")
            continue
        
        # Create a subdirectory for the model name
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Create output filename (do not include model name in filename)
        input_filename = os.path.basename(jsonl_file)
        output_path = os.path.join(model_output_dir, input_filename)
        
        # Generate forecasts
        results = await evaluate_paleka_questions(
            model_name=model_name,
            questions_data=questions_data,
            output_file=output_path,
            max_tokens=max_tokens,
            num_generations=num_generations,
            batch_size=batch_size,
        )
        
        # Save final results in the required format
        final_output_path = output_path.replace('.jsonl', '_final.jsonl')
        with open(final_output_path, 'w') as f:
            for result in results:
                # Only write the line part in the required format
                output_line = {"line": result["line"]}
                f.write(json.dumps(output_line) + '\n')
        
        logger.info(f"Saved {len(results)} final results to {final_output_path}")

async def main():
    parser = argparse.ArgumentParser(description="Paleka forecasting evaluation using OpenRouter API")
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/paleka", 
                       help="Base directory to save outputs")
    
    parser.add_argument('--max_tokens', type=int, default=16384, 
                       help="Maximum number of tokens for generation")
    
    parser.add_argument('--data_dir', type=str, 
                       default="/fast/nchandak/forecasting/datasets/paleka/tuples_2028",
                       help="Directory containing Paleka JSONL files")
    
    parser.add_argument('--num_generations', type=int, default=1, 
                       help="Number of generations to use per prompt")
    
    parser.add_argument('--models', nargs='+', default=[None],
                       help="List of models to evaluate")
    
    parser.add_argument('--batch_size', type=int, default=5,
                       help="Batch size for API requests")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.base_save_dir, exist_ok=True)
    logger.info(f"Output directory: {args.base_save_dir}")

    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Available models on OpenRouter for Paleka evaluation
    models = [
        "openai/gpt-4o",
        "deepseek/deepseek-chat-v3-0324",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-2.5-pro-preview",
        "meta-llama/llama-4-maverick",
        "qwen/qwen3-32b",
        "deepseek/deepseek-r1",
        "x-ai/grok-4",
    ]
    
    # Handle models list
    if args.models == [None] or not args.models or args.models[0] is None:
        args.models = models
    
    logger.info(f"Models to evaluate: {args.models}")
    
    # Process each model
    for model_name in args.models:
        logger.info(f"Evaluating model: {model_name}")
        
        # Process all files in the directory
        await process_paleka_directory(
            data_dir=args.data_dir,
            model_name=model_name,
            output_dir=args.base_save_dir,
            max_tokens=args.max_tokens,
            num_generations=args.num_generations,
            batch_size=args.batch_size,
        )
        
        # Small delay between models
        await asyncio.sleep(2)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())
