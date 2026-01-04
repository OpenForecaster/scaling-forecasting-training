#!/usr/bin/env python3
import re
import json
import os
import sys
import logging
import asyncio
import argparse
import numpy as np
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from datasets import Dataset, load_dataset

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


def extract_answer(completion: str) -> Optional[str]:
    """Extract the final answer from the LLM's output."""
    # Check if completion is None or not a string
    if completion is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(completion, str):
        completion = str(completion)
    
    # Check if completion is empty after conversion
    if not completion.strip():
        return None
    
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    
    try:
        matches = re.finditer(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
        matches_list = list(matches)
    except:
        return None
    
    if not matches_list:
        return None 
    
    # Get the last match
    last_match = matches_list[-1]
    answer_text = last_match.group(1).strip()
    
    return answer_text


def extract_probability(completion: str) -> Optional[float]:
    """Extract the probability from the LLM's output."""
    # Check if completion is None or not a string
    if completion is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(completion, str):
        completion = str(completion)
    
    # Check if completion is empty after conversion
    if not completion.strip():
        return None
    
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


def format_futurebench_prompt(
    question: str,
    event_type: str = "",
    open_to_bet_until: str = ""
) -> str:
    """Format the prompt for FutureBench dataset with YES/NO answer."""
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best guess for the final answer (YES/NO). Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question: {question}

{f"Event Type: {event_type}" if event_type else ""}
{f"Open to bet until: {open_to_bet_until}" if open_to_bet_until else ""}

Think step by step about the information provided, reason about uncertainty and put your final answer (YES or NO) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. Thus, the range of the score is [-1, 0]. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be either YES or NO and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""

    return prompt


def format_futurebench_prompt_binary(
    question: str,
) -> str:
    """Format the prompt without article context (probability-only version)."""
    
    prompt = f"""You will be asked a BINARY forecasting question.  You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your probability estimate of it resolving YES).
        
Question: {question}

Think step by step about the information provided, reason about uncertainty and put your final probability for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""

    return prompt


def add_idx_column(dataset: Dataset) -> Dataset:
    """Adds an 'idx' column to the dataset, storing the original row index."""
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)


def load_existing_results(output_file: str) -> Dict[int, Dict]:
    """Load existing results from JSONL file if it exists."""
    existing_results = {}
    
    if os.path.exists(output_file):
        logger.info(f"Found existing results file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line.strip())
                        idx = result.get('idx')
                        if idx is not None:
                            existing_results[idx] = result
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(existing_results)} existing results")
    
    return existing_results


def save_results_incrementally(results: List[Dict], output_file: str):
    """Save results to JSONL file."""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


async def evaluate_model(
    model_name: str,
    dataset: List[dict],
    output_file: str,
    num_generations: int = 1,
    max_tokens: int = 8192,
    use_answer_format: bool = False,
    batch_size: int = 5,
):
    """Run inference using the existing OpenRouterInference engine with incremental saving."""
    
    # Load existing results
    existing_results = load_existing_results(output_file)
    
    # Initialize the inference engine
    inference_engine = OpenRouterInference(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.6  # Will be adjusted automatically based on model
    )
    
    # Determine what needs to be processed
    missing_prompts = []
    missing_metadata = []
    
    for i, row in enumerate(dataset):
        question_idx = row["idx"]
        
        # Check if this question already has complete results
        if question_idx in existing_results:
            existing_result = existing_results[question_idx]
            existing_responses = existing_result.get("response", [])
            existing_answers = existing_result.get("extracted_answer", [])
            
            # Check if we have all required generations
            if len(existing_responses) >= num_generations and len(existing_answers) >= num_generations:
                # Check if responses are valid (not empty/None)
                valid_responses = sum(1 for resp in existing_responses if resp and resp.strip() and "<probability" in resp)
                
                extracted_answers = existing_result.get("extracted_answer", [])
                num_extracted_answers = sum(1 for ans in extracted_answers if ans and len(ans) > 0)
                if valid_responses >= num_generations and num_extracted_answers >= num_generations:
                    continue  # Skip this question, it's already complete
        
        # This question needs processing - add all generations for it
        for gen_idx in range(num_generations):
            # Format the prompt
            if use_answer_format:
                prompt = format_futurebench_prompt(
                    question=row["question"],
                    event_type=row.get("event_type", ""),
                    open_to_bet_until=row.get("open_to_bet_until", "")
                )
            else:
                prompt = format_futurebench_prompt_binary(
                    question=row["question"]
                )
            
            if i == 0 and gen_idx == 0:
                logger.info(f"Sample prompt: {prompt}")
            
            missing_prompts.append(prompt)
            missing_metadata.append((row, gen_idx))
    
    logger.info(f"Found {len(missing_prompts)} prompts to process (out of {len(dataset) * num_generations} total)")

    if not missing_prompts:
        logger.info("All results already exist, nothing to process")
        # Convert existing results to list format
        all_results = list(existing_results.values())
        return all_results
    
    # Process in batches
    question_results = {}
    
    # Initialize question_results with existing data
    for idx, existing_result in existing_results.items():
        if idx in question_results:
            continue
        
        # Find the corresponding row
        row = None
        for r in dataset:
            if r["idx"] == idx:
                row = r
                break
        
        if row:
            question_results[idx] = {
                "row": row,
                "responses": existing_result.get("response", []),
                "final_answers": existing_result.get("extracted_answer", []),
                "prompt_tokens": existing_result.get("prompt_tokens", []),
                "completion_tokens": existing_result.get("completion_tokens", []),
                "reasoning": existing_result.get("reasoning", []),
            }
    
    # Process missing prompts in batches
    for batch_start in tqdm(range(0, len(missing_prompts), batch_size), desc=f"Processing {model_name}"):
        batch_end = min(batch_start + batch_size, len(missing_prompts))
        batch_prompts = missing_prompts[batch_start:batch_end]
        batch_metadata = missing_metadata[batch_start:batch_end]
        
        # Generate completions for this batch
        batch_completions = await inference_engine.generate(
            prompts=batch_prompts,
            batch_size=batch_size
        )
        
        # Process batch results
        for (row, gen_idx), completion in zip(batch_metadata, batch_completions):
            question_idx = row["idx"]
            
            if question_idx not in question_results:
                question_results[question_idx] = {
                    "row": row,
                    "responses": [],
                    "final_answers": [],
                    "prompt_tokens": [],
                    "completion_tokens": [],
                    "reasoning": [],
                }
            
            response = None
            
            # Handle None completions (failed requests)
            if completion is None:
                completion = ""
                final_ans = {}
                prompt_tokens = 0
                completion_tokens = 0
                reasoning = ""
            else:
                response = completion['response']
                prompt_tokens = completion['prompt_tokens']
                completion_tokens = completion['completion_tokens']
                reasoning = completion['reasoning']
                
                # Extract answer and probability
                if use_answer_format:
                    last_ans = extract_answer(response)
                else:
                    last_ans = "YES"  # Default for probability-only format
                    
                final_prob = extract_probability(response)
                final_ans = {last_ans: final_prob} if last_ans else {}
            
            # Ensure we have the right number of slots
            while len(question_results[question_idx]["responses"]) <= gen_idx:
                question_results[question_idx]["responses"].append("")
                question_results[question_idx]["final_answers"].append({})
                question_results[question_idx]["prompt_tokens"].append(0)
                question_results[question_idx]["completion_tokens"].append(0)
                question_results[question_idx]["reasoning"].append("")
                
            # Store the result at the correct generation index
            question_results[question_idx]["responses"][gen_idx] = response
            question_results[question_idx]["final_answers"][gen_idx] = final_ans
            question_results[question_idx]["prompt_tokens"][gen_idx] = prompt_tokens
            question_results[question_idx]["completion_tokens"][gen_idx] = completion_tokens
            question_results[question_idx]["reasoning"][gen_idx] = reasoning
            
        # Save progress after each batch
        current_results = []
        for question_idx, data in question_results.items():
            row = data["row"]
            
            result = {
                "model": model_name,
                "split": "eval",
                "data_type": "futurebench",
                "idx": question_idx,
                "response": data["responses"],
                "extracted_answer": data["final_answers"],
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "reasoning": data["reasoning"],
                "use_answer_format": use_answer_format,
                # FutureBench specific fields
                "event_id": row.get("event_id", ""),
                "question": row.get("question", ""),
                "event_type": row.get("event_type", ""),
                "open_to_bet_until": row.get("open_to_bet_until", ""),
                "result": row.get("result", ""),
                "source": row.get("source", ""),
                # Convert result to binary label (1 for Yes, 0 for No/None of the above)
                "resolution": 1 if row.get("result", "").lower() == "yes" else 0,
            }
            
            current_results.append(result)
        
        # Save incremental results
        save_results_incrementally(current_results, output_file)
        logger.info(f"Saved progress: {len(current_results)} results to {output_file}")
        
        # Small delay between batches
        await asyncio.sleep(1)
    
    # Convert to final result format
    all_results = []
    for question_idx, data in question_results.items():
        row = data["row"]
        
        result = {
            "model": model_name,
            "split": "eval",
            "data_type": "futurebench",
            "idx": question_idx,
            "response": data["responses"],
            "extracted_answer": data["final_answers"],
            "prompt_tokens": data["prompt_tokens"],
            "completion_tokens": data["completion_tokens"],
            "reasoning": data["reasoning"],
            "use_answer_format": use_answer_format,
            # FutureBench specific fields
            "event_id": row.get("event_id", ""),
            "question": row.get("question", ""),
            "event_type": row.get("event_type", ""),
            "open_to_bet_until": row.get("open_to_bet_until", ""),
            "result": row.get("result", ""),
            "source": row.get("source", ""),
            # Convert result to binary label (1 for Yes, 0 for No/None of the above)
            "resolution": 1 if row.get("result", "").lower() == "yes" else 0,
        }
        
        all_results.append(result)
    
    return all_results


async def main():
    parser = argparse.ArgumentParser(description="FutureBench evaluation using OpenRouter API")
    # parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/futurebench", 
    #                    help="Base directory to save outputs")
    parser.add_argument('--base_save_dir', default="/home/nchandak/forecasting/evals/futurebench", 
                       help="Base directory to save outputs")
    parser.add_argument('--data_split', type=str, default="train", 
                       help="Data split to use (train/test/validation)")
    parser.add_argument('--num_generations', type=int, default=1, 
                       help="Number of generations to use per prompt")
    parser.add_argument('--use_answer_format', action='store_true', 
                       help="Whether to use YES/NO answer format instead of probability-only")
    parser.add_argument('--max_tokens', type=int, default=32768, 
                       help="Maximum number of tokens for generation")
    parser.add_argument('--models', nargs='+', default=[None],
                       help="List of models to evaluate")
    parser.add_argument('--batch_size', type=int, default=400,
                       help="Batch size for API requests")
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_base_dir = args.base_save_dir
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")
    
    # Load FutureBench dataset from HuggingFace
    logger.info(f"Loading FutureBench dataset")
    dataset = load_dataset('futurebench/data', split=args.data_split)
    
    logger.info(f"Original dataset size: {len(dataset)}")
    
    # Keep only unique event_ids (take first occurrence of each event_id)
    seen_event_ids = set()
    unique_events = []
    
    for item in dataset:
        event_id = item['event_id']
        if event_id not in seen_event_ids:
            seen_event_ids.add(event_id)
            unique_events.append(item)
    
    logger.info(f"Unique events found: {len(unique_events)}")
    
    # Convert to Dataset format and add index
    dataset = Dataset.from_list(unique_events)
    dataset = add_idx_column(dataset)
    
    # Convert to list format for processing
    questions_data = []
    for item in dataset:
        questions_data.append(dict(item))
    
    logger.info(f"Data split: {args.data_split}")
    logger.info(f"Dataset size: {len(questions_data)}")
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Use answer format: {args.use_answer_format}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Available models on OpenRouter for FutureBench evaluation
    models = [
        # "openai/gpt-4o",
        "openai/o4-mini-high", 
        # "google/gemini-2.5-pro-preview",
        # "meta-llama/llama-3.3-70b-instruct",
        # "google/gemini-2.5-flash-preview",
        # "meta-llama/llama-4-maverick",
        # "openai/gpt-oss-120b",
        # "x-ai/grok-3-mini",
        # "deepseek/deepseek-chat-v3-0824",
    ]
    
    # Handle models list
    if args.models == [None] or not args.models or args.models[0] is None:
        args.models = models
    
    logger.info(f"Models to evaluate: {args.models}")
    
    # Process each model
    for model_name in args.models:
        logger.info(f"Evaluating model: {model_name}")
        
        # Create output filename
        model_clean = model_name.split("/")[-1]
        answer_suffix = "_answer" if args.use_answer_format else ""
        output_file = os.path.join(
            output_base_dir, 
            f"{model_clean}_{args.data_split}_size_{len(questions_data)}_generations_{args.num_generations}{answer_suffix}.jsonl"
        )
        
        # Run evaluation
        all_results = await evaluate_model(
            model_name=model_name,
            dataset=questions_data,
            output_file=output_file,
            num_generations=args.num_generations,
            max_tokens=args.max_tokens,
            use_answer_format=args.use_answer_format,
            batch_size=args.batch_size,
        )
        
        # Final save (in case there were no new batches to process)
        save_results_incrementally(all_results, output_file)
        logger.info(f"Final save: {len(all_results)} question results to {output_file}")
        
        # Log some statistics
        total_generations = len(all_results) * args.num_generations
        valid_count = 0
        
        # Count valid answers
        for result in all_results:
            for final_answer in result['extracted_answer']:
                if final_answer is not None and len(final_answer) > 0:
                    valid_count += 1
        
        logger.info(f"Valid answers extracted: {valid_count}/{total_generations} ({valid_count/total_generations*100:.1f}%)")
        
        # Calculate accuracy for binary questions
        correct_count = 0
        for result in all_results:
            expected_label = result['resolution']  # 1 for Yes, 0 for No
            for final_answer in result['extracted_answer']:
                if final_answer is not None:
                    # Extract the answer key (YES/NO) and probability
                    answer_key = list(final_answer.keys())[0] if final_answer else None
                    if answer_key:
                        predicted_label = 1 if answer_key.strip().upper() == "YES" else 0
                        if predicted_label == expected_label:
                            correct_count += 1

        accuracy = correct_count / total_generations if total_generations > 0 else 0
        logger.info(f"Accuracy: {correct_count}/{total_generations} ({accuracy*100:.1f}%)")
        
        # Calculate statistics for numeric answers (probabilities)
        numeric_answers = []
        for answer in [ans for result in all_results for ans in result['extracted_answer']]:
            if answer is not None:
                try:
                    # Extract probability from the answer dict
                    probability = list(answer.values())[0]
                    if probability is not None:
                        numeric_val = float(probability)
                        numeric_answers.append(numeric_val)
                except (ValueError, TypeError, IndexError):
                    pass

        if numeric_answers:
            logger.info(f"Numeric answers: {len(numeric_answers)}/{valid_count}")
            logger.info(f"Mean prediction: {np.mean(numeric_answers):.3f} ± {np.std(numeric_answers):.3f}")
            logger.info(f"Prediction range: [{np.min(numeric_answers):.3f}, {np.max(numeric_answers):.3f}]")
            
            # Calculate correlation with true labels if available
            true_labels = []
            predicted_values = []
            for result in all_results:
                if result.get('resolution') is not None:
                    true_labels.append(result['resolution'])
                    if result['extracted_answer'] and result['extracted_answer'][0] is not None:
                        try:
                            probability = list(result['extracted_answer'][0].values())[0]
                            if probability is not None:
                                predicted_values.append(float(probability))
                        except (ValueError, TypeError, IndexError):
                            pass

            if len(true_labels) == len(predicted_values) and len(true_labels) > 1:
                correlation = np.corrcoef(true_labels, predicted_values)[0, 1]
                logger.info(f"Correlation with true labels: {correlation:.3f}")
        
        # Small delay between models
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main()) 