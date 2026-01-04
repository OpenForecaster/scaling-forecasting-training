#!/usr/bin/env python3

"""
Evaluation script for FutureX benchmark (past questions variant).
Evaluates models on historical forecasting questions from FutureX dataset.
Tests retrospective forecasting ability on questions with known outcomes.
Useful for evaluating model calibration and reasoning on past events.
Uses vLLM for efficient inference.
"""

import json
import os
import sys
import time
import ast
import numpy as np
from typing import List, Dict
from datetime import datetime
from vllm import SamplingParams

# Import common utilities
from utils import (
    setup_seeds, setup_logging, setup_environment,
    add_idx_column, extract_answer, extract_probability,
    extract_boxed_answer, load_model_and_tokenizer, apply_chat_template
)

# Setup
setup_seeds()
setup_environment()
logger = setup_logging()

MODEL_DIR = ""
DATA_SPLIT = "train"
OUTPUT_DIR = "/fast/nchandak/forecasting/evals/futurex-past/"
DATA = "futurex_past"


def format_futurex_past_prompt(
    question: str,
    options: List[str],
    end_time: str = "",
) -> str:
    """Format the prompt for FutureX-Past dataset (multiple choice format)."""
    
    # Format options
    options_text = ""
    if options:
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            options_text += f"{letter}. {option}\n"
    
    prompt = f"""You are an agent that can predict future events. The event to be predicted: "{question}"

{f"End time: {end_time}" if end_time else ""}

{options_text if options_text else ""}

Think step by step about the information provided and reason about the most likely outcome. Put your final answer in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

IMPORTANT: Your final answer MUST end with this exact format: listing all plausible options you have identified, separated by commas, within the box. For example: \\boxed{{A}} for a single option or \\boxed{{B, C, D}} for multiple options. Do not use any other format. Do not refuse to make a prediction. Do not say "I cannot predict the future." You must make a clear prediction based on the best data currently available, using the box format specified above.

Your response SHOULD STRICTLY END with <answer> </answer> tags, <probability> </probability> tags, and the \\boxed{{}} format.
"""

    return prompt


def add_binary_suffix(prompt: str) -> str:
    """Add the binary suffix to the prompt."""
    return prompt + """
Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""


def add_freeform_suffix(prompt: str) -> str:
    """Add the freeform suffix to the prompt."""
    return prompt + """
Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = -1.25 whereas if the answer was correct, then your score would be - (1 - 0.5)^2 = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be one of the options provided (A, B, C, etc.) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""


def add_numeric_suffix(prompt: str) -> str:
    """Add the numeric suffix to the prompt."""
    return prompt + """
Think step by step about the information provided, reason about uncertainty and give your best guess for the final answer (value of the event asked to upto two decimal places) in <answer> </answer> tags. 

You will be rewarded based on how close your prediction is to the actual value (1 - relative error).

Your response SHOULD STRICTLY END with <answer> </answer> tags.
"""


def format_futurex_past_binary_prompt(
    question: str,
    end_time: str = "",
) -> str:
    """Format the prompt for FutureX-Past binary questions."""
    
    prompt = f"""You are an agent that can predict future events. The event to be predicted: "{question}"

{f"End time: {end_time}" if end_time else ""}

Think step by step about the information provided and reason about the most likely outcome. Put your final answer (Yes or No) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

IMPORTANT: Your final answer MUST end with this exact format: \\boxed{{Yes}} or \\boxed{{No}}. Do not use any other format. Do not refuse to make a prediction. Do not say "I cannot predict the future." You must make a clear prediction based on the best data currently available, using the box format specified above.

Your response SHOULD STRICTLY END with <answer> </answer> tags, <probability> </probability> tags, and the \\boxed{{}} format.
"""

    return prompt


def fix_prompt_with_retrieval(prompt: str, retrieved_news_articles_summaries: str) -> str:
    """Add retrieval information to the prompt."""
    
    extra_info2 = ""
    extra_info1 = ""
    if len(retrieved_news_articles_summaries) > 10:
        extra_info1 = " You will also be provided with a list of retrieved news articles summaries which you may refer to when coming up with your answer."
        extra_info2 = f"\nRelevant passages from retrieved news articles:\n{retrieved_news_articles_summaries}\n"
        
        
    prefix1 = "come up with the best guess for the final answer."
    
    prefix1_idx = prompt.find(prefix1)
    if prefix1_idx != -1:
        before_idx = prefix1_idx + len(prefix1)
        suffix = prompt[before_idx:]
        prompt = prompt[:before_idx] + extra_info1 + suffix
    
    prefix2 = "Think step by step about the information provided, reason about uncertainty"
    
    prefix2_idx = prompt.find(prefix2)
    if prefix2_idx != -1:
        before_idx = prefix2_idx - 1
        suffix = prompt[before_idx:]
        prompt = prompt[:before_idx] + extra_info2 + suffix
    
    return prompt


def load_futurex_retrieval_data(file_path: str) -> List[dict]:
    """
    Load FutureX-Past retrieval data from JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing FutureX data with retrieval
        
    Returns:
        List of dictionaries with FutureX question data
    """
    questions_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    # Add idx if not present
                    if 'idx' not in item:
                        item['idx'] = line_idx
                    questions_data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx}: {e}")
                    continue

    logger.info(f"Loaded {len(questions_data)} questions from {file_path}")
    return questions_data



def evaluate_model(
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset: List[dict],
    max_new_tokens: int = 8192,
    batch_size: int = 32,
    max_prompt_length: int = 4096,
    num_generations: int = 8,
    num_articles: int = 10,
):
    """
    Run batched inference with multiple generations per prompt using vLLM
    """
    # Create prompts from each row
    all_prompts = []
    all_idxs = []
    all_row_data = []
    articles_to_use = max(1, num_articles)
    logger.info(f"Using {articles_to_use} articles per prompt")
    
    for i, row in enumerate(dataset):
        # Format the prompt - handle different field names for retrieval vs regular dataset
        options = row.get("answer", row.get("futurex_original_answer", []))
        level = row.get("level", row.get("futurex_level", 0))
        is_binary = row.get("is_binary", 0)
        
        # Handle options format - might be a list or string representation
        if isinstance(options, str) and options.startswith('['):
            try:
                options = ast.literal_eval(options)
            except (ValueError, SyntaxError):
                options = []
        elif not isinstance(options, list):
            options = []
        
        # Handle retrieval if available
        relevant_docs = row.get("relevant_articles_sorted_by_docs", [])
        retrieved_news_articles_summaries = ""
        
        if relevant_docs:
            j = 1
            for doc in relevant_docs[:articles_to_use]:
                
                article_title = None
                article_summary = None
                article_passage = None
                article_date = None
                article_source = None 
                source_text = ""
                date_text = ""
                # if not isinstance(doc[2], dict) and isinstance(doc[2], str):
                    
                for item in doc:
                    if isinstance(item, dict):
                        if "title" in item:
                            article_title = item["title"]
                            
                        if "relevant_passage" in item:
                            article_passage = item["relevant_passage"]
                            
                        elif "summary" in item and item["prompt_name"] == "create_forecast_summarization_prompt":
                            article_summary = item["summary"]
                            
                        if "max_date" in item:
                            article_date = item["max_date"]
                            # this is in ISO format in int, convert to human readable format
                            article_date = datetime.fromtimestamp(article_date).strftime("%B %d, %Y") 
                            date_text = f"Article Publish Date: {article_date}\n"
                            

                        if "source_domain" in item:
                            article_source = item["source_domain"]
                            source_text = f"Source: {article_source}\n"
                            
                            
                    if article_title is not None :
                        if article_passage is not None:
                            retrieved_news_articles_summaries += f"Article {j}:\nTitle: {article_title}\n{source_text}{date_text}Relevant Passage: {article_passage}\n\n"
                        elif article_summary is not None:
                            retrieved_news_articles_summaries += f"Article {j}:\nTitle: {article_title}\n{source_text}{date_text}Summary: {article_summary}\n\n"
                            
                
                
                j += 1
        
        # Get the original prompt from the data
        og_prompt = row.get("prompt", "")
        
        # Remove "IMPORTANT" section if present (will be re-added by suffixes)
        if "IMPORTANT" in og_prompt:
            og_prompt = og_prompt.split("IMPORTANT")[0].strip()
        
        # Generate prompt with or without retrieval
        if retrieved_news_articles_summaries:
            # Use retrieval versions
            local_prompt = fix_prompt_with_retrieval(og_prompt, retrieved_news_articles_summaries)
        else:
            # Use original versions without retrieval
            if is_binary or not options:
                local_prompt = add_binary_suffix(og_prompt)
            elif level <= 1:
                local_prompt = add_freeform_suffix(og_prompt)
            elif level == 4:
                local_prompt = add_numeric_suffix(og_prompt)
            else:
                # Fallback to freeform for other levels
                local_prompt = add_freeform_suffix(og_prompt)
                
        if i == 0:
            logger.info(f"Sample prompt: {local_prompt[:1000]}...")
            
        prompt = apply_chat_template(tokenizer, local_prompt, model_name)
            
        all_prompts.append(prompt)
        all_idxs.append(row["idx"])
        all_row_data.append(row)
    
    # Configure sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=num_generations,  # Number of generations per prompt
    )
    
    # Process all prompts with vLLM
    logger.info(f"Starting generation with vLLM for {len(all_prompts)} prompts, {num_generations} generations each")
    start_time = time.time()
    
    # Generate completions using vLLM's batched API
    all_outputs = model.generate(all_prompts, sampling_params)
    
    end_time = time.time()
    logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Process results - group by prompt instead of individual generations
    all_results = []
    
    for i, outputs in enumerate(all_outputs):
        prompt = all_prompts[i]
        idx = all_idxs[i]
        row = all_row_data[i]
        
        # Collect all generations for this prompt
        responses = []
        completion_tokens_list = []
        final_answers = []
        
        for output in outputs.outputs:
            generated_text = output.text
            
            # Find where the prompt ends and the completion begins
            prompt_end_idx = generated_text.find("Let me solve this step by step.\n<think>")
            if prompt_end_idx == -1:
                # Fallback if the expected text isn't found
                answer = generated_text
            else:
                answer = generated_text[prompt_end_idx:]
            
            # Calculate token counts (approximate for vLLM)
            completion_tokens = len(tokenizer.encode(answer))
            
            if "</think>" in answer:
                answer = answer.split("</think>")[1]
                
            # Extract answer and probability
            boxed_ans = extract_boxed_answer(answer)
            regular_ans = extract_answer(answer)
            final_prob = extract_probability(answer)
            
            # Use boxed answer if available, otherwise use regular answer
            final_answer_text = boxed_ans if boxed_ans else regular_ans
            
            # For binary questions, convert probability to YES/NO if no explicit answer
            if not final_answer_text and final_prob is not None:
                if final_prob > 0.5:
                    final_answer_text = "YES"
                else:
                    final_answer_text = "NO"
                    final_prob = 1 - final_prob
                    
            final_ans = {final_answer_text: final_prob} if final_answer_text else {}
                
            responses.append(answer)
            completion_tokens_list.append(completion_tokens)
            final_answers.append(final_ans)

        # Calculate prompt tokens once per prompt
        prompt_tokens = len(tokenizer.encode(prompt))
        
        # Store result with lists for generations
        result = {
            "model": model_name,
            "split": DATA_SPLIT,
            "data_type": DATA,
            "idx": idx,
            "response": responses,  # List of responses
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens_list,  # List of completion token counts
            "extracted_answer": final_answers,  # List of final answers
            # FutureX-Past specific fields
            "question_id": row.get("question_id", ""),
            "question": row.get("question", row.get("question_title", "")),
            "answer": row.get("answer", row.get("futurex_original_answer", "")),
            "options": row.get("futurex_options", row.get("options", [])),
            "end_time": row.get("question_close_date", row.get("end-time", "")),
            "level": row.get("futurex_level", row.get("level", 0)),
            "original_prompt": row.get("futurex_prompt", row.get("prompt", "")),
            "with_retrieval": len(retrieved_news_articles_summaries) > 10,
        }
        
        all_results.append(result)
    
    # Log mean output token length with standard deviation
    all_completion_tokens = []
    for result in all_results:
        all_completion_tokens.extend(result["completion_tokens"])
    mean_output_length = np.mean(all_completion_tokens)
    std_output_length = np.std(all_completion_tokens)
    logger.info(f"Mean output token length: {mean_output_length:.2f} ± {std_output_length:.2f}")
    
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FutureX-Past evaluation using local models with vLLM")
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/futurex-past86-retrieval", 
                       help="Base directory to save outputs")
    
    parser.add_argument('--model_dir', type=str, default="/fast/nchandak/models/Qwen3-8B", 
                       help="Model directory")
    parser.add_argument('--model', type=str, default="None", 
                       help="Model name")
    
    # Add max_new_tokens arg
    parser.add_argument('--max_new_tokens', type=int, default=32768, 
                       help="Maximum number of new tokens for generation")
    
    parser.add_argument('--data_split', type=str, default="train", 
                       help="Data split to use")
    
    parser.add_argument('--retrieval_dataset', type=str, 
                       default="/fast/nchandak/forecasting/datasets/futurex/with_retrieval/futurex-withretrieval_past_train_level_1_size_86_30.jsonl",
                       help="Path to retrieval dataset JSONL file")
    
    parser.add_argument('--num_generations', type=int, default=5, 
                       help="Number of generations to use per prompt")
    
    parser.add_argument('--num_articles', type=int, default=5, 
                       help="Number of articles to use per prompt")
    
    # unused argument
    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/datasets/futurex/with_retrieval/futurex-withretrieval_past_train_level_1_size_86_30.jsonl",
                       help="Path to JSONL file containing articles with final_question field")
    
    parser.add_argument('--level_filter', type=int, nargs='+', default=[1],
                       help="Filter dataset to only include questions of these levels (default: [1])")
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_base_dir = args.base_save_dir
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")

    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {gpu_count}")
    
    MODEL_DIR = args.model_dir
    DATA_SPLIT = args.data_split
    
    # Load FutureX-Past retrieval data
    logger.info(f"Loading FutureX-Past retrieval dataset from: {args.retrieval_dataset}")
    questions_data = load_futurex_retrieval_data(args.retrieval_dataset)
    
    if not questions_data:
        logger.error("No valid questions found in the input file")
        sys.exit(1)
    
    # Filter by level if specified
    if args.level_filter:
        filtered_items = []
        for item in questions_data:
            # Handle different level field names
            level = item.get('futurex_level', item.get('level', 0))
            if int(level) in args.level_filter:
                filtered_items.append(item)
        questions_data = filtered_items
        logger.info(f"Filtered to level {args.level_filter}: {len(questions_data)} questions")
    
    logger.info(f"Data split: {DATA_SPLIT}")
    logger.info(f"Dataset size: {len(questions_data)}")
    logger.info(f"Level filter: {args.level_filter}")
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Number of articles per prompt: {args.num_articles}")
    
    model_name = args.model
    
    # Extract model name from model_dir 
    if args.model == "None":
        model_name = MODEL_DIR.rstrip("/").split("/")[-1]
        if "__" in model_name:
            model_name = model_name.split("__")[1]
        
    logger.info(f"Model name: {model_name}")
    
    # Create output filename
    retrieval_suffix = f"_retrieval_{args.num_articles}"
    output_file = os.path.join(
        output_base_dir, 
        f"{model_name}_{DATA_SPLIT}_level_{','.join(map(str, args.level_filter))}_size_{len(questions_data)}_generations_{args.num_generations}{retrieval_suffix}.jsonl"
    )
    logger.info(f"Output file: {output_file}")
    
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Exiting without running evaluation.")
        exit(0)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir, model_name)
    
    # Run evaluation
    all_results = evaluate_model(
        model_name, 
        model, 
        tokenizer, 
        questions_data, 
        max_new_tokens=args.max_new_tokens, 
        num_generations=args.num_generations, 
        num_articles=args.num_articles
    )
    
    # Save results as JSONL
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(all_results)} question results to {output_file}")
    
    # Log some statistics
    total_generations = len(all_results) * args.num_generations
    valid_count = 0
    
    # Count valid answers
    for result in all_results:
        for final_answer in result['extracted_answer']:
            if final_answer is not None and len(final_answer) > 0:
                valid_count += 1
    
    logger.info(f"Valid answers extracted: {valid_count}/{total_generations} ({valid_count/total_generations*100:.1f}%)")
    
    # Calculate accuracy for multiple choice questions
    correct_count = 0
    total_with_ground_truth = 0
    
    for result in all_results:
        ground_truth = result.get('answer', [])
        if ground_truth:  # Only evaluate if we have ground truth
            ground_truth_str = str(ground_truth).lower()
            if ground_truth_str.startswith('['):
                ground_truth_str = ground_truth_str[2:-2].lower()
                
            total_with_ground_truth += len(result['extracted_answer'])
            for final_answer in result['extracted_answer']:
                if final_answer is not None and len(final_answer) > 0:
                    # Extract the predicted answer
                    predicted = list(final_answer.keys())[0].lower() if final_answer else None
                    if predicted:
                        # Check if prediction matches ground truth answer
                        if predicted == ground_truth_str or predicted.strip() in ground_truth_str:
                            correct_count += 1

    if total_with_ground_truth > 0:
        accuracy = correct_count / total_with_ground_truth
        logger.info(f"Accuracy: {correct_count}/{total_with_ground_truth} ({accuracy*100:.1f}%)")
    
    # # Calculate statistics for probabilities
    # numeric_answers = []
    # for answer in [ans for result in all_results for ans in result['extracted_answer']]:
    #     if answer is not None:
    #         try:
    #             # Extract probability from the answer dict
    #             probability = list(answer.values())[0]
    #             if probability is not None:
    #                 numeric_val = float(probability)
    #                 numeric_answers.append(numeric_val)
    #         except (ValueError, TypeError, IndexError):
    #             pass

    # if numeric_answers:
    #     logger.info(f"Numeric answers: {len(numeric_answers)}/{valid_count}")
    #     logger.info(f"Mean confidence: {np.mean(numeric_answers):.3f} ± {np.std(numeric_answers):.3f}")
    #     logger.info(f"Confidence range: [{np.min(numeric_answers):.3f}, {np.max(numeric_answers):.3f}]")
