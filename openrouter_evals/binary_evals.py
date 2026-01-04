#!/usr/bin/env python3
import re
import json
import os
import sys
import logging
import asyncio
import argparse
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
    
    try :
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

def format_forecasting_prompt_answer(
    question_title: str,
    background: str,
    resolution_criteria: str,
) -> str:
    """Format the prompt for binary forecasting (YES/NO)."""
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best guess for the final answer (YES/NO). Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final answer (YES or NO) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. Thus, the range of the score is [-1, 0]. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be either YES or NO and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt

def format_forecasting_prompt_binary_abstain(
    question_title: str,
    background: str,
    resolution_criteria: str,
) -> str:
    """Format the prompt for binary forecasting with abstain option."""
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best guess for the final answer (YES/NO). Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final answer (YES or NO) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. Thus, the range of the score is [-1, 0]. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Try hard to come up with the best guess for the final answer. ONLY IF you cannot decide which resolution is more likely, then just say "UNKNOWN" in the <answer> </answer> tags and assign a probability of *0.5* to it (i.e., both outcomes being equally likely). REMEMBER THAT YOU SHOULD ALWAYS TRY TO MAXIMIZE YOUR SCORE.

Your final answer should be either YES or NO and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt




def format_forecasting_prompt_binary(
    question_title: str,
    background: str,
    resolution_criteria: str,
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a binary forecasting question.  You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""

    return prompt




def format_forecasting_prompt_binary_with_retrieval(
    question_title: str,
    background: str,
    resolution_criteria: str,
    retrieved_news_articles_summaries: str,
) -> str:
    """Format the prompt for single outcome forecasting."""
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. You will also be provided with a list of retrieved news articles summaries which you may refer to when coming up with your answer. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Relevant passaged retrieved from News Articles:
{retrieved_news_articles_summaries}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

    return prompt


def load_questions_from_jsonl(file_path: str) -> List[dict]:
    """Load articles with questions from JSONL file and extract question components."""
    questions_data = []
    import os

    # Determine file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    try:
                        article = json.loads(line.strip())
                        
                        # Skip if question is not relevant or the article is not relevant
                        if 'question_relevant' in article and int(article['question_relevant']) == 0:
                            continue
                        
                        if 'article_relevant' in article and int(article['article_relevant']) == 0:
                            continue
                        
                        if 'no_good_question' in article and int(article['no_good_question']) == 1:
                            continue
                        
                        question_dict = {}
                        
                        if 'question_title' in article:
                            question_dict['question_title'] = article['question_title']
                        elif 'question' in article:
                            question_dict['question_title'] = article['question']
                            
                        if 'background' in article:
                            question_dict['background'] = article['background']
                        if 'resolution_criteria' in article:
                            question_dict['resolution_criteria'] = article['resolution_criteria']
                        if 'answer' in article:
                            question_dict['answer'] = article['answer']
                        if 'answer_type' in article:
                            question_dict['answer_type'] = article['answer_type']
                        
                        if len(list(question_dict.keys())) >= 5:
                            question_dict['resolution_date'] = article.get('resolution_date', '')
                            question_dict['question_start_date'] = article.get('question_start_date', article.get('date_begin', ''))
                            question_dict['question_close_date'] = article.get('question_close_date', article.get('date_close', ''))
                            
                            # Create a question entry with all necessary fields
                            question_entry = {
                                'idx': line_idx,
                                'question_title': question_dict.get('question_title', ''),
                                'background': question_dict.get('background', ''),
                                'resolution_criteria': question_dict.get('resolution_criteria', ''),
                                'answer': question_dict.get('answer', ''),
                                'answer_type': question_dict.get('answer_type', ''),
                                'resolution_date': question_dict.get('resolution_date', ''),
                                'question_start_date': question_dict.get('question_start_date', ''),
                                'question_close_date': question_dict.get('question_close_date', ''),
                                'nr_forecasters': article.get('nr_forecasters', ''),
                                'resolution': article.get('resolution', ''),
                                "relevant_docs": article.get('relevant_docs', article.get('relevant_articles_sorted_by_docs', [])),
                                'url': article.get('url', ''),
                                'prompt': article.get('prompt', ''),
                            }
                            # Only add if we have a valid question title
                            if question_entry['question_title'].strip():
                                questions_data.append(question_entry)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_idx}: {e}")
                        continue

    elif ext == ".json":
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, dict):
                    # Check if it's a numbered key structure (like {"1": {...}, "2": {...}})
                    if all(key.isdigit() for key in data.keys()):
                        # Convert to list format
                        articles = []
                        for key in sorted(data.keys(), key=int):
                            article = data[key]
                            article['_original_key'] = key  # Keep track of original key
                            articles.append(article)
                        data = articles
                    else:
                        # Single object, wrap in list
                        data = [data]
                
                for line_idx, article in enumerate(data):
                    # Skip if question is not relevant or the article is not relevant
                    if 'question_relevant' in article and int(article['question_relevant']) == 0:
                        continue
                    
                    if 'article_relevant' in article and int(article['article_relevant']) == 0:
                        continue
                    
                    if 'no_good_question' in article and int(article['no_good_question']) == 1:
                        continue
                    
                    question_dict = {}
                    
                    if 'question_title' in article:
                        question_dict['question_title'] = article['question_title']
                    elif 'question' in article:
                        question_dict['question_title'] = article['question']
                        
                    if 'background' in article:
                        question_dict['background'] = article['background']
                    if 'resolution_criteria' in article:
                        question_dict['resolution_criteria'] = article['resolution_criteria']
                    if 'answer' in article:
                        question_dict['answer'] = article['answer']
                    if 'answer_type' in article:
                        question_dict['answer_type'] = article['answer_type']
                    
                    if len(list(question_dict.keys())) >= 5:
                        question_dict['resolution_date'] = article.get('resolution_date', '')
                        question_dict['question_start_date'] = article.get('question_start_date', article.get('date_begin', ''))
                        question_dict['question_close_date'] = article.get('question_close_date', article.get('date_close', ''))
                        
                        # Create a question entry with all necessary fields
                        question_entry = {
                            'idx': line_idx,
                            'question_title': question_dict.get('question_title', ''),
                            'background': question_dict.get('background', ''),
                            'resolution_criteria': question_dict.get('resolution_criteria', ''),
                            'answer': question_dict.get('answer', ''),
                            'answer_type': question_dict.get('answer_type', ''),
                            'resolution_date': question_dict.get('resolution_date', ''),
                            'question_start_date': question_dict.get('question_start_date', ''),
                            'question_close_date': question_dict.get('question_close_date', ''),
                            'nr_forecasters': article.get('nr_forecasters', ''),
                            'resolution': article.get('resolution', ''),
                            'url': article.get('url', ''),
                            'prompt': article.get('prompt', ''),
                        }
                        # Only add if we have a valid question title
                        if question_entry['question_title'].strip():
                            questions_data.append(question_entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON file {file_path}: {e}")
    else:
        logger.error(f"Unsupported file extension for {file_path}. Only .jsonl and .json are supported.")
        return []

    logger.info(f"Loaded {len(questions_data)} valid questions from {file_path}")
    return questions_data

def parse_filename_for_dataset_info(file_path: str) -> tuple:
    """Extract dataset info from filename."""
    filename = os.path.basename(file_path)
    
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '').replace('.json', '')
    
    # Split by underscore and look for patterns
    parts = name_without_ext.split('_')
    
    # Try to find dataset name and num_lines pattern
    dataset_name = ""
    num_lines = ""
    
    # Look for common dataset names
    common_datasets = ['metaculus', 'manifold', 'halawi']
    for part in parts:
        if part.lower() in common_datasets:
            dataset_name = part.lower()
            break
    
    # Look for numbers
    for part in parts:
        if part.isdigit():
            num_lines = part
            break
    
    # Fallback
    if not dataset_name:
        dataset_name = "unknown"
    if not num_lines:
        num_lines = "unknown"
    
    return dataset_name, num_lines

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
    abstain: bool = False,
    batch_size: int = 5,
    num_articles: int = 5,
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
            
            # Format the prompt for each example
            relevant_docs = row.get("relevant_articles_sorted_by_docs", row.get("relevant_docs", []))
            retrieved_news_articles_summaries = ""
            
            j = 1 
            for doc in relevant_docs[:num_articles]:
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
                
            
            if len(retrieved_news_articles_summaries) > 0:
                prompt = format_forecasting_prompt_binary_with_retrieval(
                    question_title=row["question_title"],
                    background=row["background"],
                    resolution_criteria=row["resolution_criteria"],
                    retrieved_news_articles_summaries=retrieved_news_articles_summaries,
                )
            else :
                if abstain:
                    prompt = format_forecasting_prompt_binary_abstain(
                        question_title=row["question_title"],
                        background=row["background"],
                        resolution_criteria=row["resolution_criteria"],
                    )
                else:
                    prompt = format_forecasting_prompt_binary(
                        question_title=row["question_title"],
                        background=row["background"],
                        resolution_criteria=row["resolution_criteria"],
                    )
                
            
            if i == 101:
                logger.info(f"Prompt: {prompt}")
                resolution_date = row.get("resolution_date", "")
                # logger.info(f"Question resolution date: {datetime.fromtimestamp(resolution_date).strftime('%B %d, %Y')}")
            
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
                # Extract single answer and probability
                if abstain:
                    last_ans = extract_answer(response)
                else:
                    last_ans = "YES"
                    
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
                "data_type": "binary",
                "idx": question_idx,
                "response": data["responses"],
                "extracted_answer": data["final_answers"],
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "reasoning": data["reasoning"],
                "abstain": abstain,
                # Question metadata
                "article_url": row.get("url", ""),
                "question_title": row.get("question_title", ""),
                "resolution_date": row.get("resolution_date", ""),
                "question_start_date": row.get("question_start_date", ""), 
                "question_close_date": row.get("question_close_date", ""),
                "nr_forecasters": row.get("nr_forecasters", ""),
                "resolution": row.get("resolution", ""),
                "background": row.get("background", ""),
                "resolution_criteria": row.get("resolution_criteria", ""),
                "answer": row.get("answer", ""),
                "answer_type": row.get("answer_type", ""),
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
            "data_type": "binary",
            "idx": question_idx,
            "response": data["responses"],
            "extracted_answer": data["final_answers"],
            "abstain": abstain,
            "prompt_tokens": data["prompt_tokens"],
            "completion_tokens": data["completion_tokens"],
            "reasoning": data["reasoning"],
            # Question metadata
            "article_url": row.get("url", ""),
            "question_title": row.get("question_title", ""),
            "resolution_date": row.get("resolution_date", ""),
            "question_start_date": row.get("question_start_date", ""), 
            "question_close_date": row.get("question_close_date", ""),
            "nr_forecasters": row.get("nr_forecasters", ""),
            "resolution": row.get("resolution", ""),
            "background": row.get("background", ""),
            "resolution_criteria": row.get("resolution_criteria", ""),
            "answer": row.get("answer", ""),
            "answer_type": row.get("answer_type", ""),
        }
        
        all_results.append(result)
    
    return all_results

async def main():
    parser = argparse.ArgumentParser(description="Binary forecasting evaluation using OpenRouter API")
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/binary/metaculusOct_30/", 
                       help="Base directory to save outputs")
    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/with_retreival/metaculus-05-2025_30.jsonl",
    #                    help="Path to JSONL file containing articles with question fields")
    # parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/with_retreival/metaculus-gptfiltered_30.jsonl",
                       help="Path to JSONL file containing articles with question fields")
    parser.add_argument('--num_generations', type=int, default=3, 
                       help="Number of generations to use per prompt")
    parser.add_argument('--abstain', action='store_true', 
                       help="Whether to use abstain option in the prompt")
    parser.add_argument('--max_tokens', type=int, default=32768, 
                       help="Maximum number of tokens for generation")
    parser.add_argument('--models', nargs='+', default=[None],
                       help="List of models to evaluate")
    parser.add_argument('--batch_size', type=int, default=500,
                       help="Batch size for API requests")
    parser.add_argument('--num_articles', type=int, default=5, help="Number of articles to use per prompt") 
    
    args = parser.parse_args()
    
    # Extract dataset info from filename
    dataset_name, num_lines = parse_filename_for_dataset_info(args.questions_file)
    # full_dataset_name = f"{dataset_name}_{num_lines}"
    
    # Create output directory structure
    output_base_dir = args.base_save_dir
    # join with dataset_name
    # output_base_dir = os.path.join(output_base_dir, dataset_name)
    
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")
    
    # Load questions from JSONL file
    logger.info(f"Loading questions from: {args.questions_file}")
    questions_data = load_questions_from_jsonl(args.questions_file)
    
    if not questions_data:
        logger.error("No valid questions found in the input file")
        sys.exit(1)
    
    logger.info(f"Dataset size: {len(questions_data)}")
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Abstain: {args.abstain}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Available models on OpenRouter for binary evaluation
    models = [
        # "openai/o4-mini-high",
        # "meta-llama/llama-3.3-70b-instruct",
        # "meta-llama/llama-4-maverick",
        # "openai/gpt-oss-120b",
        # "openai/gpt-oss-20b",
        # "x-ai/grok-4",
        # "deepseek/deepseek-chat-v3-0824",
        # "deepseek/deepseek-r1-distill-llama-70b",
        # "google/gemma-3-27b-it",
        # "qwen/qwen-2.5-7b-instruct",
        # "deepseek/deepseek-r1-distill-llama-8b",
        # "meta-llama/llama-3.1-8b-instruct",
    ]
    
    
    models = [
        # "openai/gpt-4o",
        # "deepseek/deepseek-chat-v3-0324",
        
        # "openai/o4-mini-high",
        # "google/gemini-2.5-pro-preview",
        # "meta-llama/llama-3.3-70b-instruct",
        
        # "google/gemini-2.5-flash-preview",
        # "meta-llama/llama-4-maverick",
        # "meta-llama/llama-4-scout",
        
        
        # "openai/gpt-oss-120b",
        # "x-ai/grok-4-fast",
        # "openai/gpt-oss-20b",
        
        
        # "deepseek/deepseek-r1",
        # "openai/o4-mini-high",
        "deepseek/deepseek-r1-0528",
        # "qwen/qwen3-32b",
        # "qwen/qwen3-235b-a22b",
        # "inception/mercury",
        # "x-ai/grok-4.1-fast",
        # "moonshotai/kimi-k2",
        
        # "qwen/qwen3-235b-a22b-07-25",
        # "qwen/qwen3-235b-a22b-thinking-2507",
        
        # "x-ai/grok-3-mini-beta",
        # "x-ai/grok-3-mini",
        # "deepseek/deepseek-r1-0528",
        # "mistralai/mistral-medium-3",
        # "microsoft/phi-4",
        # "deepseek/deepseek-r1-distill-llama-70b",
        # "meta-llama/llama-4-scout",
        # "qwen/qwen-2.5-72b-instruct",
        # "google/gemma-3-27b-it",
        # "openai/gpt-4.1-nano",
        
        # "openai/o4-mini",
        # "openai/o3",
        # "qwen/qwen3-14b"
        # "qwen/qwen3-32b"
        
        # "qwen/qwen-2.5-7b-instruct",
        
        # "deepseek/deepseek-r1-distill-qwen-7b",
        # "deepseek/deepseek-r1-distill-llama-8b",
        # "meta-llama/llama-3.1-8b-instruct",
        # "deepseek/deepseek-r1-distill-qwen-32b",
        # "qwen/qwen-2.5-72b-instruct",
        
        
        
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
        abstain_suffix = "_abstain" if args.abstain else ""
        output_file = os.path.join(
            output_base_dir, 
            f"{model_clean}_eval_size_{len(questions_data)}_generations_{args.num_generations}{abstain_suffix}_num_articles_{args.num_articles}.jsonl"
        )
        
        # Run evaluation
        all_results = await evaluate_model(
            model_name=model_name,
            dataset=questions_data,
            output_file=output_file,
            num_generations=args.num_generations,
            max_tokens=args.max_tokens,
            abstain=args.abstain,
            batch_size=args.batch_size,
            num_articles=args.num_articles,
        )
        
        # Final save (in case there were no new batches to process)
        save_results_incrementally(all_results, output_file)
        logger.info(f"Final save: {len(all_results)} question results to {output_file}")
        
        # Log some statistics
        total_generations = len(all_results) * args.num_generations
        valid_count = 0
        
        # For binary outcomes
        for result in all_results:
            for final_answer in result['extracted_answer']:
                if final_answer is not None and len(final_answer) > 0:
                    valid_count += 1
        
        logger.info(f"Valid answers extracted: {valid_count}/{total_generations} ({valid_count/total_generations*100:.1f}%)")
        
        # Small delay between models
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main()) 