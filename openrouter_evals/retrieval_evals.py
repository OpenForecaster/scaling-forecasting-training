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


def extract_multiple_answers_and_probabilities(completion: str) -> dict:
    """
    Extract multiple answers and their corresponding probabilities from the LLM's output.
    Expected format: <answer1> outcome1 </answer1> <probability1> prob1 </probability1>
    
    Returns:
        Dictionary with answers as keys and probabilities as values
        If no valid answers found, returns {}
    """
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
        
    answer_prob_dict = {}
    
    # Find all answer-probability pairs
    answer_pattern = r"<answer(\d+)>(.*?)<\/answer\1>\s*<probability\1>(.*?)<\/probability\1>"
    matches = re.finditer(answer_pattern, completion, re.DOTALL)
    
    for match in matches:
        answer_num = match.group(1)
        answer_text = match.group(2).strip()
        prob_text = match.group(3).strip()
        
        # Try to parse probability as float
        try:
            probability = float(prob_text)
            # Validate probability is between 0 and 1
            if 0 <= probability <= 1:
                answer_prob_dict[answer_text] = probability
            else:
                logger.warning(f"Invalid probability value {probability} for answer {answer_num}")
        except (ValueError, TypeError):
            logger.warning(f"Could not parse probability '{prob_text}' for answer {answer_num}")
            continue
    
    # Log extraction results
    if answer_prob_dict:
        total_prob = sum(answer_prob_dict.values())
        logger.debug(f"Extracted {len(answer_prob_dict)} answers with total probability {total_prob:.3f}")
        if (total_prob - 1.0) > 0.1:  # Warning if probabilities don't sum to <= 1
            logger.warning(f"Probabilities sum to {total_prob:.3f}, not 1.0")
    else:
        logger.debug("No valid answer-probability pairs found")
    
    return answer_prob_dict

def format_forecasting_prompt(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
) -> str:
    """Format the prompt for single outcome forecasting."""
    
    prompt = f"""You will be asked a forecasting question (which might be from the past). You have to come up with the best guess for the final answer. Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Expected Answer Type: {answer_type}

Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. TRY HARD TO COME UP WITH THE BEST GUESS FOR THE FINAL ANSWER. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt

def format_forecasting_prompt_multiple_outcomes(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
) -> str:
    """Format the prompt for multiple outcomes forecasting."""
    
    prompt = f"""You will be asked a forecasting question. Please provide your reasoning before stating your final answer.

Think step by step about the information provided. You are expected to reason about the possible outcomes and list your best estimate of how likely each of them are. Thus, you have to provide a list of mostly likely outcomes and their forecasted probability for each of them. YOUR PROBABILITIES MUST SUM LESS THAN OR EQUAL TO 1 (BUT NOT NECESSARILY 1).

Your will be rewarded based on your probability listed for the different outcomes in reference to the actual (true) outcome of the event. The rule to evaluate your answer will be the multi-class brier scoring rule which is basically: (- \\sum_k (p_k - y_k)^2) where p_k is the probability you assigned to the k^th outcome and y_k is 1 if the k^th outcome is the true outcome and 0 otherwise. If you don't assign list the actual true outcome (i.e., don't assign any probability to it), then it assumed that your probability for that outcome is 0. Thus, the best score you can achieve is 0 by assigning all the probability to the correct outcome (getting (1 - 1)^2 = 0 for the correct outcome and 0 for the incorrect outcomes, so total is 0). Whereas, the worst score you can get is -2 by assigning all the probability to a single incorrect outcome (getting -(1 - 0)^2 = -1 for the incorrect outcomes and -((0 - 1)^2) = -1 for the correct outcome, so total is -2). Hence, the range of your score is [-2, 0]. THINK HARD ABOUT THE POSSIBLE OUTCOMES AND THEIR CHANCE OF HAPPENING. YOU HAVE TO MAXIMIZE YOUR SCORE. BUT ALSO ENSURE THAT YOUR PROBABILITIES DO NOT SUM MORE THAN 1.

**Example**

**Example Question**:
Question Title: Who will win the Nobel Prize in Literature in 2016?
Background: Question Start Date: 10th January 2016. The Nobel Prize in Literature is awarded annually by the Swedish Academy to authors for their outstanding contributions to literature.
Resolution Criteria: The question will resolve when the Swedish Academy publicly announces the official 2016 Nobel Prize in Literature laureate(s) typically via a press release on NobelPrize.org (expected on or about October 13, 2016). The full name of the laureate exactly as given in the announcement should be provided.

**Example Reasoning and Output**:
...Based on recent literary achievements and past Nobel Prize patterns, I'll consider the most likely candidates. The Nobel Prize in Literature often recognizes established authors with significant global impact, though it can sometimes surprise with unconventional choices.

<answer1> Haruki Murakami </answer1> <probability1> 0.35 </probability1>
<answer2> Philip Roth </answer2> <probability2> 0.25 </probability2>
<answer3> Joyce Carol Oates </answer3> <probability3> 0.18 </probability3>
<answer4> Bob Dylan </answer4> <probability4> 0.11 </probability4>
<answer5> Ngugi wa Thiong'o </answer5> <probability5> 0.09 </probability5>

**Example Score**:
The sum of the probabilities is 0.35 + 0.25 + 0.18 + 0.11 + 0.09 = 0.6 + 0.29 + 0.09 = 0.98. The correct answer to the question is Bob Dylan. The probability assigned to Bob Dylan is 0.11. Hence, the score is - ( (1 - 0.11)^2 + (0 - 0.35)^2 + (0 - 0.25)^2 + (0 - 0.18)^2 + (0 - 0.09)^2 ) = - ( (0.89)^2 + (-0.35)^2 + (-0.25)^2 + (-0.18)^2 + (-0.09)^2 ) = - ( 0.7921 + 0.1225 + 0.0625 + 0.0324 + 0.0081 ) = - 1.0176.

**ACTUAL QUESTION**:

Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Expected Answer Type: {answer_type}

**Output Format**
... {{reasoning}} ...
<answer1> outcome1 </answer1> <probability1> probability1 </probability1>
<answer2> outcome2 </answer2> <probability2> probability2 </probability2>
<answer3> outcome3 </answer3> <probability3> probability3 </probability3>
...
<answerN> outcomeN </answerN> <probabilityN> probabilityN </probabilityN>

{{ IMPORTANT: probability1 + probability2 + ... + probabilityN <= 1 }}

EACH ANSWER YOU LIST SHOULD BE AN ACTUAL POSSIBLE ANSWER/OUTCOME OF THE QUESTION/EVENT ASKED AND IT SHOULD NOT BE "other", "rest", "etc.". EACH (DISTINCT) ANSWER YOU OUTPUT WILL BE COUNTED AS A SEPARATE OUTCOME CLASS.
"""

    return prompt



def format_forecasting_prompt_with_retrieval(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
    retrieved_news_articles_summaries: str,
) -> str:
    """Format the prompt for single outcome forecasting."""
    
    prompt = f"""You will be asked a forecasting question (which might be from the past). You have to come up with the best guess for the final answer. You will also be provided with a list of retrieved news articles summaries which you may refer to when coming up with your answer. Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Expected Answer Type: {answer_type}

Relevant passaged retrieved from News Articles:
{retrieved_news_articles_summaries}

Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = -1.25 whereas if the answer was correct, then your score would be - (1 - 0.5)^2 = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt


def load_questions_from_jsonl(file_path: str) -> List[dict]:
    """Load articles with questions from JSONL file and extract question components."""
    questions_data = []
    
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
                    
                    # Create a question entry with all necessary fields
                    question_entry = {
                        'idx': line_idx,
                        'question_title': article.get('question_title', ''),
                        'background': article.get('background', ''),
                        'resolution_criteria': article.get('resolution_criteria', ''),
                        'answer': article.get('answer', ''),
                        'answer_type': article.get('answer_type', ''),
                        'resolution_date': article.get('resolution_date', ''),
                        'question_start_date': article.get('question_start_date', ''),
                        'url': article.get('url', ''),
                        'date_publish': article.get('date_publish', ''),
                        'relevant_docs': article.get('relevant_docs', article.get('relevant_articles_sorted_by_docs', [])),
                    }
                    
                    # Only add if we have a valid question title
                    if question_entry['question_title'].strip():
                        questions_data.append(question_entry)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx}: {e}")
                    continue
    
    logger.info(f"Loaded {len(questions_data)} valid questions from {file_path}")
    return questions_data

def parse_filename_for_dataset_info(file_path: str) -> tuple:
    """Extract news_source and num_lines from filename."""
    filename = os.path.basename(file_path)
    
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Split by underscore and look for patterns
    parts = name_without_ext.split('_')
    
    # Try to find news_source_numlines pattern
    news_source = ""
    num_lines = ""
    
    for i in range(len(parts) - 1):
        # Check if current part could be news source and next could be number
        if parts[i] and parts[i+1].isdigit():
            news_source = parts[i]
            num_lines = parts[i+1]
            break
    
    # If we couldn't find the pattern, try alternative approaches
    if not news_source or not num_lines:
        # Look for common news sources
        common_sources = ['dw', 'cnn', 'cbsnews', 'foxnews', 'reuters', 'theguardian', 'bbc', 'ap', 'npr', 'wsj', 'nyt']
        for part in parts:
            if part.lower() in common_sources:
                news_source = part.lower()
                break
        
        # Look for numbers
        for part in parts:
            if part.isdigit():
                num_lines = part
                break
    
    # Fallback
    if not news_source:
        news_source = "unknown"
    if not num_lines:
        num_lines = "unknown"
    
    return news_source, num_lines

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
    multiple_outcomes: bool = False,
    batch_size: int = 5,
    num_articles: int = 10,
):
    """Run inference using the existing OpenRouterInference engine with incremental saving."""
    
    # Load existing results
    existing_results = load_existing_results(output_file)
    
    # Initialize the inference engine
    inference_engine = OpenRouterInference(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.7  # Will be adjusted automatically based on model
    )
    
    # Determine what needs to be processed
    missing_prompts = []
    missing_metadata = []
    # articles_to_use = max(1, num_articles)
    articles_to_use = num_articles
    
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
                # print(existing_responses)
                valid_responses = sum(1 for resp in existing_responses if resp and resp.strip() and "<answer" in resp and "<probability" in resp)
                extracted_answers = existing_result.get("extracted_answer", [])
                num_extracted_answers = sum(1 for ans in extracted_answers if ans and len(ans) > 0)
                if valid_responses >= num_generations and num_extracted_answers >= num_generations:
                    continue  # Skip this question, it's already complete
        
        # This question needs processing - add all generations for it
        for gen_idx in range(num_generations):
            
            
            # Format the prompt for each example
            relevant_docs = row.get("relevant_articles_sorted_by_docs", row.get("relevant_docs", []))
            retrieved_news_articles_summaries = ""
            
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


            if multiple_outcomes:
                prompt = format_forecasting_prompt_multiple_outcomes(
                    question_title=row["question_title"],
                    background=row["background"],
                    resolution_criteria=row["resolution_criteria"],
                    answer_type=row["answer_type"],
                )
            else:
                if len(retrieved_news_articles_summaries) > 10:
                    prompt = format_forecasting_prompt_with_retrieval(
                        question_title=row["question_title"],
                        background=row["background"],
                        resolution_criteria=row["resolution_criteria"],
                        answer_type=row["answer_type"],
                        retrieved_news_articles_summaries=retrieved_news_articles_summaries,
                    )
                else :
                    prompt = format_forecasting_prompt(
                        question_title=row["question_title"],
                        background=row["background"],
                        resolution_criteria=row["resolution_criteria"],
                        answer_type=row["answer_type"],
                    )
            
            if i == 101:
                logger.info(f"Prompt: {prompt}")
                
            # # Format the prompt
            # if multiple_outcomes:
            #     prompt = format_forecasting_prompt_multiple_outcomes(
            #         question_title=row["question_title"],
            #         background=row["background"],
            #         resolution_criteria=row["resolution_criteria"],
            #     )
            # else:
            #     prompt = format_forecasting_prompt(
            #         question_title=row["question_title"],
            #         background=row["background"],
            #         resolution_criteria=row["resolution_criteria"],
            #     )
            
            
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
                "prompt_tokens": existing_result.get("prompt_tokens", []),
                "completion_tokens": existing_result.get("completion_tokens", []),
                "reasoning": existing_result.get("reasoning", []),
                "final_answers": existing_result.get("extracted_answer", []),
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
                response = ""
                final_ans = {}
                prompt_tokens = 0
                completion_tokens = 0
                reasoning = ""
            else:
                response = completion['response']
                prompt_tokens = completion['prompt_tokens']
                completion_tokens = completion['completion_tokens']
                reasoning = completion['reasoning']
                # Extract answer based on format
                if multiple_outcomes:
                    # Extract multiple answers and probabilities as dictionary
                    answer_prob_dict = extract_multiple_answers_and_probabilities(response)
                    final_ans = answer_prob_dict  # Store dictionary of answers and probabilities
                else:
                    # Extract single answer
                    last_ans = extract_answer(response)
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
                "data_type": "freeform_retrieval",
                "idx": question_idx,
                "response": data["responses"],
                "extracted_answer": data["final_answers"],
                "multiple_outcomes": multiple_outcomes,
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "reasoning": data["reasoning"],
                # Question metadata
                "article_url": row.get("url", ""),
                "question_title": row.get("question_title", ""),
                "resolution_date": row.get("resolution_date", ""),
                "question_start_date": row.get("question_start_date", ""), 
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
            "data_type": "freeform_retrieval",
            "idx": question_idx,
            "response": data["responses"],
            "extracted_answer": data["final_answers"],
            "multiple_outcomes": multiple_outcomes,
            "completion_tokens": data["completion_tokens"],
            "prompt_tokens": data["prompt_tokens"],
            "reasoning": data["reasoning"],
            # Question metadata
            "article_url": row.get("url", ""),
            "question_title": row.get("question_title", ""),
            "resolution_date": row.get("resolution_date", ""),
            "question_start_date": row.get("question_start_date", ""), 
            "background": row.get("background", ""),
            "resolution_criteria": row.get("resolution_criteria", ""),
            "answer": row.get("answer", ""),
            "answer_type": row.get("answer_type", ""),
        }
        
        all_results.append(result)
    
    return all_results

async def main():
    parser = argparse.ArgumentParser(description="Freeform forecasting evaluation using OpenRouter API")
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/freeform/manual", 
                       help="Base directory to save outputs")
    # parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian_207_free_3_cleaned.jsonl",
    #                     help="Path to JSONL file containing articles with question fields")
    # parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian-retrieval-new-30_207_free_3_cleaned.jsonl",
    #                     help="Path to JSONL file containing articles with question fields")
    
    # parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_1000_30.jsonl",
    #                     help="Path to JSONL file containing articles with question fields")
    
    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_validation-retrieval_207_30.jsonl",
                        help="Path to JSONL file containing articles with question fields")
    
    parser.add_argument('--num_generations', type=int, default=3, 
                       help="Number of generations to use per prompt")
    parser.add_argument('--multiple_outcomes', action='store_true', 
                       help="Whether to use multiple outcomes in the prompt")
    parser.add_argument('--max_tokens', type=int, default=32768, # 16384, #32768, 
                       help="Maximum number of tokens for generation")
    parser.add_argument('--models', nargs='+', default=[None],
                       help="List of models to evaluate")
    parser.add_argument('--batch_size', type=int, default=500,
                       help="Batch size for API requests")
    parser.add_argument('--num_articles', type=int, default=5, help="Number of articles to use per prompt")
    
    args = parser.parse_args()
    
    # Extract dataset info from filename
    news_source, num_lines = parse_filename_for_dataset_info(args.questions_file)
    dataset_name = f"{news_source}_{num_lines}"
    
    # Create output directory structure
    output_base_dir = os.path.join(args.base_save_dir, dataset_name)
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
    logger.info(f"Multiple outcomes: {args.multiple_outcomes}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Batch size: {args.batch_size}")
    
    
    
    # Available models on OpenRouter
    
    models = [
        # "openai/gpt-4o",
        # "deepseek/deepseek-chat-v3-0324",
        
        # "x-ai/grok-4",
        # "openai/o4-mini-high",
        # "google/gemini-2.5-pro-preview",
        # "google/gemini-2.5-flash",
        # "meta-llama/llama-3.3-70b-instruct",
        
        # "google/gemini-2.5-flash-preview",
        # "meta-llama/llama-4-maverick",
        # "meta-llama/llama-4-scout",
        
        # "x-ai/grok-code-fast-1",
        # "openai/gpt-oss-120b",
        # "x-ai/grok-4.1-fast:online",
        # "openai/gpt-oss-20b",
        # "qwen/qwen3-30b-a3b",
        # "qwen/qwen3-30b-a3b-thinking-2507",
        
        # "google/gemini-3-flash-preview",
        # "qwen/qwen3-235b-a22b",
        # "deepseek/deepseek-r1-0528",
        # "deepseek/deepseek-r1",
        # "qwen/qwen3-32b",
        "qwen/qwen3-235b-a22b"
        # "inception/mercury",
        # "moonshotai/kimi-k2",
        # "moonshotai/kimi-k2-thinking",
        
        # "deepseek/deepseek-chat-v3-0324",
        # "openai/gpt-5",
        # "openai/gpt-5.1",
        # "deepseek/deepseek-v3.2",
        # "deepseek/deepseek-chat-v3.1",
        
        # "qwen/qwen3-235b-a22b-07-25",
        # "qwen/qwen3-235b-a22b-thinking-2507",
        
        # "x-ai/grok-3-mini-beta",
        # "x-ai/grok-3-mini",
        # "x-ai/grok-4.1-fast",
        # "x-ai/grok-4-fast:free",
        # "mistralai/mistral-medium-3",
        # "microsoft/phi-4",
        # "meta-llama/llama-4-scout",
        # "qwen/qwen-2.5-72b-instruct",
        # "google/gemma-3-27b-it",
        # "openai/gpt-4.1-nano",
        
        # "openai/o4-mini",
        # "openai/o3",
        # "qwen/qwen3-14b"
        # "qwen/qwen3-32b"
        
        # "qwen/qwen-2.5-7b-instruct",
        # "deepseek/deepseek-r1-distill-llama-70b",
        # "deepseek/deepseek-r1-distill-llama-8b",
        # "meta-llama/llama-3.1-8b-instruct",
        
        # "deepseek/deepseek-r1-distill-qwen-7b",
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
        # model_clean = model_name.replace("/", "_").replace("-", "_")
        model_clean = model_name.split("/")[-1]
        multiple_outcomes_suffix = "_multiple_outcomes" if args.multiple_outcomes else ""
        output_file = os.path.join(
            output_base_dir, 
            f"{model_clean}_eval_size_{len(questions_data)}_generations_{args.num_generations}_num_articles_{args.num_articles}{multiple_outcomes_suffix}.jsonl"
        )
        
        # Run evaluation
        all_results = await evaluate_model(
            model_name=model_name,
            dataset=questions_data,
            output_file=output_file,
            num_generations=args.num_generations,
            max_tokens=args.max_tokens,
            multiple_outcomes=args.multiple_outcomes,
            batch_size=args.batch_size,
            num_articles=args.num_articles,
        )
        
        # Final save (in case there were no new batches to process)
        save_results_incrementally(all_results, output_file)
        logger.info(f"Final save: {len(all_results)} question results to {output_file}")
        
        # Log some statistics
        total_generations = len(all_results) * args.num_generations
        valid_count = 0
        
        if args.multiple_outcomes:
            # For multiple outcomes, count valid answer sets (dictionaries)
            all_prob_sums = []
            for result in all_results:
                for final_answer in result['extracted_answer']:
                    if final_answer is not None and isinstance(final_answer, dict) and len(final_answer) > 0:
                        valid_count += 1
                        # Calculate probability sum for this generation
                        prob_sum = sum(final_answer.values()) if final_answer.values() else 0
                        all_prob_sums.append(prob_sum)
            
            # Log probability statistics
            if all_prob_sums:
                import numpy as np
                mean_prob_sum = np.mean(all_prob_sums)
                std_prob_sum = np.std(all_prob_sums)
                logger.info(f"Probability sums: {mean_prob_sum:.3f} ± {std_prob_sum:.3f}")
                prob_sums_near_one = sum(1 for p in all_prob_sums if abs(p - 1.0) <= 0.1)
                logger.info(f"Probability sums near 1.0 (±0.1): {prob_sums_near_one}/{len(all_prob_sums)} ({prob_sums_near_one/len(all_prob_sums)*100:.1f}%)")
        else:
            # For single outcomes
            for result in all_results:
                for final_answer in result['extracted_answer']:
                    if final_answer is not None and len(final_answer) > 0:
                        valid_count += 1
        
        logger.info(f"Valid answers extracted: {valid_count}/{total_generations} ({valid_count/total_generations*100:.1f}%)")
        
        # Small delay between models
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
