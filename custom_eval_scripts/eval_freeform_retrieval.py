"""
Evaluation script for freeform forecasting with retrieval-augmented generation.
Evaluates models on freeform questions with retrieved articles provided as context.
Supports configurable number of articles (num_articles parameter) to include in prompts.
Uses vLLM for efficient inference. Variant of eval_freeform.py with RAG support.
"""

import json
import os
import sys
import time
import numpy as np
import torch
from typing import List, Tuple
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Import common utilities
from utils import (
    setup_seeds, setup_logging, setup_environment,
    extract_answer, extract_probability,
    extract_multiple_answers_and_probabilities, extract_question,
    load_model_and_tokenizer, apply_chat_template
)

# Setup
setup_seeds()
setup_environment()
logger = setup_logging()

MODEL_DIR = ""
DATA_SPLIT = "train"
OUTPUT_DIR = "/fast/nchandak/forecasting/evals/manual/"
DATA = "halawi"

def load_questions_from_jsonl(file_path: str) -> List[dict]:
    """
    Load articles with questions from JSONL file and extract question components.
    
    Args:
        file_path: Path to the JSONL file containing articles with final_question field
        
    Returns:
        List of dictionaries with extracted question components and article data
    """
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
                        final_question = article.get('final_question', '')
                        
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
                            question_dict['question_start_date'] = article.get('question_start_date', '')
                            
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
                                # Article fields for context if needed
                                'article_title': article.get('article_title', ''),
                                'article_description': article.get('article_description', ''),
                                'article_maintext': article.get('article_maintext', ''),
                                'date_publish': article.get('article_date_publish', ''),
                                'url': article.get('url', ''),
                                'relevant_docs': article.get('relevant_docs', article.get('relevant_articles_sorted_by_docs', [])),
                            }
                            # Only add if we have a valid question title
                            if question_entry['question_title'].strip():
                                questions_data.append(question_entry)
                        
                        elif final_question and len(final_question.strip()) >= 10:
                            # Extract question components from final_question field
                            question_dict = extract_question(final_question)
                            
                            # Create a question entry with all necessary fields
                            question_entry = {
                                'idx': line_idx,
                                'question_title': question_dict.get('question_title', ''),
                                'background': question_dict.get('background', ''),
                                'resolution_criteria': question_dict.get('resolution_criteria', ''),
                                'answer': question_dict.get('answer', ''),
                                'answer_type': question_dict.get('answer_type', ''),
                                'final_question': final_question,
                                # Article fields for context if needed
                                'article_title': article.get('title', ''),
                                'article_description': article.get('description', ''),
                                'article_maintext': article.get('maintext', ''),
                                'date_publish': article.get('date_publish', ''),
                                'url': article.get('url', ''),
                                'relevant_docs': article.get('relevant_docs', article.get('relevant_articles_sorted_by_docs', [])),
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
                    final_question = article.get('final_question', '')
                    
                    # Skip if question is not relevant or the article is not relevant
                    if 'question_relevant' in article and int(article['question_relevant']) == 0:
                        continue
                    
                    if 'article_relevant' in article and int(article['article_relevant']) == 0:
                        continue
                    
                    if 'no_good_question' in article and int(article['no_good_question']) == 1:
                        continue
                    
                    question_dict = {}
                    # print("Loading....")
                    
                    if 'question_title' in article:
                        question_dict['question_title'] = article['question_title']
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
                        question_dict['question_start_date'] = article.get('question_start_date', '')
                        
                        # print("Creating question entry....")
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
                            # Article fields for context if needed
                            'article_title': article.get('article_title', ''),
                            'article_description': article.get('article_description', ''),
                            'article_maintext': article.get('article_maintext', ''),
                            'date_publish': article.get('article_date_publish', ''),
                            'url': article.get('url', ''),
                            'data_source': article.get('data_source', ''),
                            'relevant_docs': json.dumps(article.get('relevant_docs', article.get('relevant_articles_sorted_by_docs', [])), ensure_ascii=False),
                        }
                        # Only add if we have a valid question title
                        if question_entry['question_title'].strip():
                            questions_data.append(question_entry)
                    
                    elif final_question and len(final_question.strip()) >= 10:
                        # Extract question components from final_question field
                        question_dict = extract_question(final_question)
                        
                        # Create a question entry with all necessary fields
                        question_entry = {
                            'idx': line_idx,
                            'question_title': question_dict.get('question_title', ''),
                            'background': question_dict.get('background', ''),
                            'resolution_criteria': question_dict.get('resolution_criteria', ''),
                            'answer': question_dict.get('answer', ''),
                            'answer_type': question_dict.get('answer_type', ''),
                            'final_question': final_question,
                            # Article fields for context if needed
                            'article_title': article.get('title', ''),
                            'article_description': article.get('description', ''),
                            'article_maintext': article.get('maintext', ''),
                            'date_publish': article.get('date_publish', ''),
                            'url': article.get('url', ''),
                            'data_source': article.get('data_source', ''),
                            'relevant_docs': json.dumps(article.get('relevant_docs', article.get('relevant_articles_sorted_by_docs', [])), ensure_ascii=False),
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

def parse_filename_for_dataset_info(file_path: str) -> Tuple[str, str]:
    """
    Extract news_source and num_lines from filename.
    
    Expected format: something_like: deepseek-chat-v3-0324_dw_30_free_1.jsonl
    Returns: (news_source, num_lines) like ("dw", "30")
    """
    import os
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
            
        for common_source in common_sources:
            for part in parts:
                if common_source in part.lower():
                    news_source = common_source
                    break
            if news_source:
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


def format_forecasting_prompt_with_retrieval_old(
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




def format_forecasting_prompt_with_retrieval(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
    retrieved_news_articles_summaries: str,
) -> str:
    """Format the prompt for single outcome forecasting."""
    
    extra_info2 = ""
    extra_info1 = ""
    if len(retrieved_news_articles_summaries) > 10:
        extra_info1 = " You will also be provided with a list of retrieved news articles summaries which you may refer to when coming up with your answer."
        extra_info2 = f"\nRelevant passages from retrieved news articles:\n{retrieved_news_articles_summaries}\n"
    
    prompt = f"""You will be asked a forecasting question (which might be from the past). You have to come up with the best guess for the final answer.{extra_info1} Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Expected Answer Type: {answer_type}
{extra_info2}
Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt




def format_forecasting_prompt_with_retrieval_passages(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
    retrieved_news_articles_passages: str,
) -> str:
    """Format the prompt for single outcome forecasting."""
    
    prompt = f"""You will be asked a forecasting question (which might be from the past). You have to come up with the best guess for the final answer. You will also be provided with a list of retrieved news articles summaries which you may refer to when coming up with your answer. Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Expected Answer Type: {answer_type}

Retrieved News Articles:
{retrieved_news_articles_passages}

Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

TRY HARD TO COME UP WITH THE BEST GUESS FOR THE FINAL ANSWER. ONLY IF you cannot think of any answer, then just say "UNKNOWN" in the <answer> </answer> tags and assign a probability of 0 to it. REMEMBER THAT YOU SHOULD ALWAYS TRY TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt


def format_forecasting_prompt_multiple_probabilities_no_article(
    question_title: str,
    background: str,
    resolution_criteria: str,
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a forecasting question. Please provide your reasoning before stating your final answer.

Think step by step about the information provided. You are expected to reason about the possible outcomes and list your best estimate of how likely each of them are. Thus, you have to provide a list of mostly likely outcomes and their forecasted probability for each of them. YOUR PROBABILITIES MUST SUM LESS THAN OR EQUAL TO 1.

Your will be rewarded based on your probability listed for the different outcomes in reference to the actual (true) outcome of the event. The rule to evaluate your answer will be the multi-class brier scoring rule which is basically - \sum_k (p_k - y_k)^2 where p_k is the probability you assigned to the k^th outcome and y_k is 1 if the k^th outcome is the true outcome and 0 otherwise. YOU HAVE TO MAXIMIZE YOUR SCORE. BUT ALSO ENSURE THAT YOUR PROBABILITIES DO NOT SUM MORE THAN 1.

**Example**

**Example Question**:
Question Title: Who will win the Nobel Prize in Literature in 2016?
Background: Question Start Date: 10th January 2016. The Nobel Prize in Literature is awarded annually by the Swedish Academy to authors for their outstanding contributions to literature.
Resolution Criteria: The question will resolve when the Swedish Academy publicly announces the official 2016 Nobel Prize in Literature laureate(s) typically via a press release on NobelPrize.org (expected on or about October 13, 2016). The full name of the laureate exactly as given in the announcement should be provided.

**Example Reasoning and Output**:
Based on recent literary achievements and past Nobel Prize patterns, I'll consider the most likely candidates. The Nobel Prize in Literature often recognizes established authors with significant global impact, though it can sometimes surprise with unconventional choices.

<answer1> Haruki Murakami </answer1> <probability1> 0.35 </probability1>
<answer2> Philip Roth </answer2> <probability2> 0.25 </probability2>
<answer3> Joyce Carol Oates </answer3> <probability3> 0.18 </probability3>
<answer4> Bob Dylan </answer4> <probability4> 0.12 </probability4>
<answer5> Ngugi wa Thiong'o </answer5> <probability5> 0.09 </probability5>

**Example Score**:
The sum of the probabilities is 0.35 + 0.25 + 0.18 + 0.12 + 0.09 = 0.6 + 0.3 + 0.09 = 0.99. The correct answer to the question is Bob Dylan. The probability assigned to Bob Dylan is 0.12. Hence, the score is - (1 - 0.12)^2 = -0.7744.

**ACTUAL QUESTION**:

Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

**Output Format**
... {{reasoning}} ...
<answer1> outcome1 </answer1> <probability1> probability1 </probability1>
<answer2> outcome2 </answer2> <probability2> probability2 </probability2>
<answer3> outcome3 </answer3> <probability3> probability3 </probability3>
...
<answerN> outcomeN </answerN> <probabilityN> probabilityN </probabilityN>

{{ IMPORTANT: probability1 + probability2 + ... + probabilityN <= 1 }}
"""

    return prompt



def evaluate_model(
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset,
    max_new_tokens: int = 8192,
    batch_size: int = 32,
    max_prompt_length: int = 4096,
    num_generations: int = 8,  # Added parameter for number of generations
    prompts_available: bool = False,
    multiple_outcomes: bool = False,  # Added parameter for using multiple outcomes
    num_articles: int = 10,
):
    """
    Run batched inference with multiple generations per prompt using vLLM
    """
    # Create prompts from each row
    all_prompts = []
    all_idxs = []
    all_row_data = []
    # using at least 1 article
    articles_to_use = num_articles # max(1, num_articles)
    logger.info(f"Using {articles_to_use} articles per prompt")
    
    for i, row in enumerate(dataset):
        # Check if row has a pre-formatted prompt
        if "prompt" in row and row["prompt"]:
            local_prompt = row["prompt"]
        else:
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
                local_prompt = format_forecasting_prompt_multiple_probabilities_no_article(
                    question_title=row["question_title"],
                    background=row["background"],
                    resolution_criteria=row["resolution_criteria"],
                )
            else:
                local_prompt = format_forecasting_prompt_with_retrieval(
                    question_title=row["question_title"],
                    background=row["background"],
                    resolution_criteria=row["resolution_criteria"],
                    answer_type=row["answer_type"],
                    retrieved_news_articles_summaries=retrieved_news_articles_summaries,
                )
                
        if i == 101:
            logger.info(f"Processing prompt {i} of {len(dataset)}")
            logger.info(f"Prompt: {local_prompt}")
            
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
    
    # sample_prompt = "Who is the president of the United States?"
    # generated_text = generate_text(model, sample_prompt, max_tokens=100, temperature=0.7, use_chat_template=True)
    # logger.info(f"Generated SAMPLE text: {generated_text}")
    
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
            
            # if "</think>" in answer:
            #     answer = answer.split("</think>")[1]
                
            # Extract the final answer based on format
            if multiple_outcomes:
                # Extract multiple answers and probabilities as dictionary
                answer_prob_dict = extract_multiple_answers_and_probabilities(answer)
                final_ans = answer_prob_dict  # Store dictionary of answers and probabilities
            else:
                # Extract single answer (keep original type, don't cast)
                last_ans = extract_answer(answer)
                final_prob = extract_probability(answer)
                
                data_source = row.get("data_source", "na")
                # if data_source.lower() == "metaculus":
                #     print(f"Metaculus question: {row.get('question_title', 'NA')}")
                #     print(f"Metaculus answer: {last_ans}")
                #     print(f"Model answer: {last_ans}")
                #     last_ans = "YES"
                    
                if last_ans is None and final_prob:
                    last_ans = "YES"
                
                final_ans = {last_ans: final_prob}
                
            responses.append(answer)
            completion_tokens_list.append(completion_tokens)
            final_answers.append(final_ans)

        # Calculate prompt tokens once per prompt
        prompt_tokens = len(tokenizer.encode(prompt))
        
        # Store result with lists for generations
        result = {
            "model": model_name,
            # "prompt": prompt,
            "split": DATA_SPLIT,
            "data_type": DATA,
            "idx": idx,
            "response": responses,  # List of responses
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens_list,  # List of completion token counts
            "extracted_answer": final_answers,  # List of final answers
            "multiple_outcomes": multiple_outcomes,
            # Additional fields requested
            "article_url": row.get("url", ""),
            "article_title": row.get("article_title", ""),
            "full_question": row.get("final_question", ""),
            "question_title": row.get("question_title", ""),
            "resolution_date": row.get("resolution_date", ""),
            "question_start_date": row.get("question_start_date", ""), 
            "background": row.get("background", ""),
            "resolution_criteria": row.get("resolution_criteria", ""),
            "answer": row.get("answer", ""),
            "answer_type": row.get("answer_type", ""),
            "with_retrieval": True,
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
    from datasets import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/freeform/manual", help="Base directory to save outputs")
    
    parser.add_argument('--model_dir', type=str, default="/fast/nchandak/models/Qwen3-8B", help="Model directory")
    parser.add_argument('--model', type=str, default="None", help="Model name")
    
    # Add max_new_tokens arg
    parser.add_argument('--max_new_tokens', type=int, default=16384, help="Maximum number of new tokens for generation")
    
    parser.add_argument('--data_split', type=str, default="eval", help="Data split to use")
    
    # parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/newsdata/ameya_retrieval/downloaded_files/o4-mini-test-set/o4-mini-high_theguardian-retrieval_207_free_3_cleaned.json",
    #                   help="Path to JSONL file containing articles with final_question field")
    
    # parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_validation-retrieval_207_30.jsonl",
    #                   help="Path to JSONL file containing articles with final_question field")

    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/newsdata/testset/withretrieval/o4-mini-high_test5news_302_30.jsonl",
                      help="Path to JSONL file containing articles with final_question field")

    # parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_1000_30.jsonl",
    #                   help="Path to JSONL file containing articles with final_question field")
    
    parser.add_argument('--num_generations', type=int, default=3, help="Number of generations to use per prompt")
    
    parser.add_argument('--num_articles', type=int, default=5, help="Number of articles to use per prompt")
    
    parser.add_argument('--multiple_outcomes', action='store_true', help="Whether to use multiple outcomes in the prompt")
    
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Extract dataset info from filename
    news_source, num_lines = parse_filename_for_dataset_info(args.questions_file)
    dataset_name = f"{news_source}_{num_lines}"
    
    if "metaculus" in args.questions_file:
        dataset_name = "metaculus"
        last_folder_name = args.questions_file.rstrip("/").split("/")[-1]
        dataset_name = dataset_name + "_" + last_folder_name
        args.base_save_dir = "/fast/nchandak/forecasting/evals/binary/with_retrieval"
    
    # Create output directory structure
    output_base_dir = os.path.join(args.base_save_dir, dataset_name)
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")

    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {gpu_count}")
    
    MODEL_DIR = args.model_dir
    DATA_SPLIT = args.data_split
    DATA = dataset_name
    
    # Load questions from JSONL file
    logger.info(f"Loading questions from: {args.questions_file}")
    questions_data = load_questions_from_jsonl(args.questions_file)
    
    if not questions_data:
        logger.error("No valid questions found in the input file")
        sys.exit(1)
    
    # Convert to Dataset format
    # dataset = Dataset.from_list(questions_data)
    dataset = questions_data
    
    # Limit dataset size if max_samples is specified
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
        logger.info(f"Limited dataset to {args.max_samples} samples for testing")
    
    logger.info(f"Data split: {DATA_SPLIT}")
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Dataset size: {len(dataset)}") 

    # dataset = add_idx_column(dataset)
    new_tokens = args.max_new_tokens
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max new tokens: {new_tokens}")
    logger.info(f"Model directory: {MODEL_DIR}")
    
    model_name = args.model
    
    # Extract model name from model_dir 
    if args.model == "None":
        model_name = MODEL_DIR.rstrip("/").split("/")[-1]
        if "__" in model_name:
            model_name = model_name.split("__")[1]
        # Remove any checkpoint suffix after model name
        # if "checkpoint" in MODEL_DIR:
        #     model_name = MODEL_DIR.rstrip("/").split("/")[-2] + "__" + MODEL_DIR.rstrip("/").split("/")[-1]
        
    logger.info(f"Model name: {model_name}")
    
    # Create output filename with use_article info
    if args.multiple_outcomes:
        multiple_outcomes_suffix = "_multiple_outcomes"
    else:
        multiple_outcomes_suffix = ""
    output_file = os.path.join(
        output_base_dir, 
        f"{model_name}_{DATA_SPLIT}_size_{len(dataset)}_generations_{args.num_generations}_num_articles_{args.num_articles}{multiple_outcomes_suffix}.jsonl"
    )
    logger.info(f"Output file: {output_file}")
    
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Exiting without running evaluation.")
        exit(0)

    
    model, tokenizer = load_model_and_tokenizer(args.model_dir, model_name)
    
    # Run evaluation
    all_results = evaluate_model(
        model_name, 
        model, 
        tokenizer, 
        dataset, 
        max_new_tokens=new_tokens, 
        num_generations=args.num_generations, 
        prompts_available=False,
        multiple_outcomes=args.multiple_outcomes,
        num_articles=args.num_articles
    )
    
    # Save results as JSONL
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(all_results)} question results to {output_file}")
    
    # Log some statistics
    total_generations = len(all_results) * args.num_generations
    all_final_answers = []
    valid_count = 0
    
    if args.multiple_outcomes:
        # For multiple outcomes, count valid answer sets (dictionaries)
        all_prob_sums = []
        for result in all_results:
            for final_answer in result['extracted_answer']:
                all_final_answers.append(final_answer)
                if final_answer is not None and isinstance(final_answer, dict) and len(final_answer) > 0:
                    valid_count += 1
                    # Calculate probability sum for this generation
                    prob_sum = sum(final_answer.values())
                    all_prob_sums.append(prob_sum)
        
        # Log probability statistics
        if all_prob_sums:
            mean_prob_sum = np.mean(all_prob_sums)
            std_prob_sum = np.std(all_prob_sums)
            logger.info(f"Probability sums: {mean_prob_sum:.3f} ± {std_prob_sum:.3f}")
            prob_sums_near_one = sum(1 for p in all_prob_sums if abs(p - 1.0) <= 0.1)
            logger.info(f"Probability sums near 1.0 (±0.1): {prob_sums_near_one}/{len(all_prob_sums)} ({prob_sums_near_one/len(all_prob_sums)*100:.1f}%)")
            
            # Log average number of outcomes per generation
            num_outcomes = [len(final_answer) for result in all_results for final_answer in result['extracted_answer'] 
                          if isinstance(final_answer, dict)]
            if num_outcomes:
                mean_outcomes = np.mean(num_outcomes)
                logger.info(f"Average number of outcomes per generation: {mean_outcomes:.1f}")
    else:
        # For single outcomes
        for result in all_results:
            for final_answer in result['extracted_answer']:
                all_final_answers.append(final_answer)
                if final_answer is not None:
                    valid_count += 1
    
    logger.info(f"Valid answers extracted: {valid_count}/{total_generations} ({valid_count/total_generations*100:.1f}%)")
    
    # # Calculate statistics for numeric answers only
    # numeric_answers = []
    # for answer in all_final_answers:
    #     if answer is not None:
    #         try:
    #             numeric_val = float(answer)
    #             numeric_answers.append(numeric_val)
    #         except (ValueError, TypeError):
    #             pass
    
    # if numeric_answers:
    #     logger.info(f"Numeric answers: {len(numeric_answers)}/{valid_count}")
    #     logger.info(f"Mean prediction: {np.mean(numeric_answers):.3f} ± {np.std(numeric_answers):.3f}")
    #     logger.info(f"Prediction range: [{np.min(numeric_answers):.3f}, {np.max(numeric_answers):.3f}]")