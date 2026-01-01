#!/usr/bin/env python3
"""
Script to extract resolution dates from forecasting questions using vLLM.

This script:
1. Takes input as either a directory (processes all .jsonl files) or a single .jsonl file
2. For each entry, extracts question, background, and resolution_criteria
3. Uses vLLM to ask model to output resolution date in YYYY-MM-DD format
4. Saves the extracted date in 'resolution_date' field
5. Skips entries that already have 'resolution_date' field
"""

import os
import argparse
import logging
import json
import sys
import re
from typing import List, Dict, Any
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from pathlib import Path

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def extract_answer(response: str) -> str:
    """
    Extract answer from model response, looking for content in <answer></answer> tags.
    
    Args:
        response: Model response string
        
    Returns:
        Extracted answer as string, or empty string if not found
    """
    if not response:
        return ""
    
    # Try to extract from <answer> tags - use findall to get all matches and take the last one
    answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if answer_matches:
        return answer_matches[-1].strip()
    
    # Look for \boxed{} notation commonly used in math
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', response, re.DOTALL)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # Fallback: look for date patterns in the response
    date_patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD format
        r'\b(\d{4}/\d{2}/\d{2})\b',  # YYYY/MM/DD format
        r'\b(\d{4}\.\d{2}\.\d{2})\b'  # YYYY.MM.DD format
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, response)
        if matches:
            # Convert to YYYY-MM-DD format
            date_str = matches[-1].replace('/', '-').replace('.', '-')
            return date_str
    
    return ""

def extract_question_components(entry: Dict[str, Any]) -> tuple[str, str, str]:
    """
    Extract question components from entry.
    Returns (question_title, background, resolution_criteria)
    """
    # Try to get from individual fields first
    question_title = entry.get('question_title', '')
    background = entry.get('background', '')
    resolution_criteria = entry.get('resolution_criteria', '')
    
    # If we have these fields, return them
    if question_title and background and resolution_criteria:
        return question_title, background, resolution_criteria
    
    # Otherwise, try to extract from final_question field
    final_question = entry.get('final_question', '')
    if final_question:
        def extract_tag_content(text: str, tag: str) -> str:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            last_open = text.rfind(open_tag)
            if last_open == -1:
                return ""
            start = last_open + len(open_tag)
            end = text.find(close_tag, start)
            if end == -1:
                return ""
            return text[start:end].strip()
        
        if not question_title:
            question_title = extract_tag_content(final_question, 'question_title')
        if not background:
            background = extract_tag_content(final_question, 'background')
        if not resolution_criteria:
            resolution_criteria = extract_tag_content(final_question, 'resolution_criteria')
    
    return question_title, background, resolution_criteria

def create_date_extraction_prompt(question_title: str, background: str, resolution_criteria: str) -> str:
    """Create a prompt for the LLM to extract resolution date."""
    
    prompt = f"""You are an expert at analyzing forecasting questions and extracting key information. Your task is to determine the RESOLUTION DATE FOR THE GIVEN QUESTION (WHICH SHOULD BE STATED IN THE QUESTION).

The resolution date is the specific date when the question will be resolved or when the answer becomes known. This should be extracted from the question content, background information, or resolution criteria.

**Question Title:** {question_title}

**Background:** {background}

**Resolution Criteria:** {resolution_criteria}

**Instructions:**
1. Carefully read through all the provided information
2. Look for any explicit dates mentioned when the question will be resolved
3. Look for phrases like "by [date]", "before [date]", "on [date]", "resolution date", "close date", etc.
4. If multiple dates are mentioned, choose the final resolution date (when the answer becomes definitively known)
5. IF NO SPECIFIC DAY IS GIVEN AND ONLY MONTH AND YEAR ARE MENTIONED, USE THE LAST DAY OF THE MONTH.
6. Format the date as YYYY-MM-DD (e.g., 2024-12-31)
7. If no specific date can be determined, output "UNKNOWN"

**Examples of what to look for:**
- "This question will resolve on December 31, 2024"
- "Resolution occurs by April 30, 2025"
- "The question closes on March 15, 2024"
- "The resolution occurs when the date is publicly confirmed, no later than October 4, 2024." (you should infer it as 2024-10-04)
- "Results will be announced in January 2025" (you should infer it as 2025-01-31)

**Important:** Only extract dates that are explicitly mentioned or can be reasonably inferred from the context. DO NOT MAKE UP DATES.

Please provide your answer in the following format:
<answer>YYYY-MM-DD</answer>

If you cannot determine a specific resolution date, respond with:
<answer>UNKNOWN</answer>
"""

    return prompt

def load_articles_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load articles from a single JSONL file."""
    articles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        articles.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
    return articles

def get_files_to_process(input_path: str) -> List[str]:
    """Get list of .jsonl files to process."""
    path = Path(input_path)
    
    if path.is_file():
        if path.suffix == '.jsonl':
            return [str(path)]
        else:
            logger.error(f"Input file must have .jsonl extension: {input_path}")
            return []
    elif path.is_dir():
        jsonl_files = list(path.glob('*.jsonl'))
        return [str(f) for f in jsonl_files]
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return []

def extract_dates_with_vllm(articles: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
    """Extract resolution dates using vLLM and add resolution_date field."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("VLLM not installed. Please install it with: pip install vllm")
        return articles
    
    # Filter articles that need resolution date extraction
    articles_to_process = []
    process_indices = []
    
    for i, article in enumerate(articles):
        # Skip if resolution_date already exists
        if 'resolution_date' in article and "unknown" not in article['resolution_date'].lower():
            continue
            
        # Extract question components
        question_title, background, resolution_criteria = extract_question_components(article)
        
        # Skip if we don't have enough information
        if not question_title or not (background or resolution_criteria):
            continue
            
        articles_to_process.append(article)
        process_indices.append(i)
    
    if not articles_to_process:
        logger.info("No articles need resolution date extraction")
        return articles
    
    logger.info(f"Processing {len(articles_to_process)} articles for resolution date extraction")
    
    # Initialize the model
    logger.info(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        max_model_len=4096,
        dtype="auto",
        trust_remote_code=True
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,    # Lower temperature for more consistent extraction
        top_p=0.95,
        top_k=20,
        max_tokens=2048,     # We only need short responses
        stop=["<|im_end|>"]
    )
    
    # Create all prompts at once and apply chat template
    logger.info("Creating prompts for all articles and applying chat template...")
    raw_prompts = []
    for article in articles_to_process:
        question_title, background, resolution_criteria = extract_question_components(article)
        prompt = create_date_extraction_prompt(question_title, background, resolution_criteria)
        raw_prompts.append(prompt)
    
    # Use tokenizer's apply_chat_template with tokenize=False
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in raw_prompts
    ]
    
    if prompts:
        logger.info(f"Example prompt preview: {prompts[0][:500]}...")
    
    # Process all prompts together
    logger.info(f"Processing {len(prompts)} prompts with VLLM...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process outputs and update articles
    logger.info(f"Processing {len(articles_to_process)} resolution date extraction results...")
    
    extracted_count = 0
    for i, (article, output) in enumerate(tqdm(zip(articles_to_process, outputs), desc="Processing resolution dates")):
        response = output.outputs[0].text.strip()
        if i == 0:
            logger.info(f"Example response: {response}")
            
        # Extract resolution date from response
        resolution_date = extract_answer(response)
        
        # Validate date format (basic check)
        if resolution_date and resolution_date != "UNKNOWN":
            # Basic validation - should be YYYY-MM-DD format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', resolution_date):
                article['resolution_date'] = resolution_date
                extracted_count += 1
            else:
                # logger.warning(f"Invalid date format extracted: {resolution_date}")
                article['resolution_date'] = "UNKNOWN"
        else:
            article['resolution_date'] = "UNKNOWN"
        
        # Store the response for debugging if needed
        article['resolution_date_response'] = response
        
        # Log some examples for debugging
        if i < 5:  # Log first 5 articles
            question_preview = extract_question_components(article)[0][:50]
            logger.info(f"Question {i + 1}: '{question_preview}...' -> Date: {article['resolution_date']}")

    logger.info(f"Resolution date extraction complete. {extracted_count} out of {len(articles_to_process)} articles got valid dates.")
    
    return articles

def process_file(file_path: str, model_path: str) -> None:
    """Process a single file for resolution date extraction."""
    logger.info(f"Processing file: {file_path}")
    
    # Load articles from file
    articles = load_articles_from_file(file_path)
    if not articles:
        logger.warning(f"No articles loaded from {file_path}")
        return
    
    original_count = len(articles)
    existing_dates = sum(1 for article in articles if article.get('resolution_date'))
    
    logger.info(f"Loaded {original_count} articles, {existing_dates} already have resolution dates")
    
    # Extract resolution dates
    start_time = time.time()
    articles = extract_dates_with_vllm(articles, model_path)
    extraction_time = time.time() - start_time
    
    logger.info(f"Date extraction completed in {extraction_time:.2f} seconds")
    
    # Save results back to the same file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        # Count final statistics
        final_dates = sum(1 for article in articles if article.get('resolution_date') and article.get('resolution_date') != 'UNKNOWN')
        logger.info(f"Updated {file_path} - {final_dates} articles now have resolution dates")
        
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract resolution dates from forecasting questions using vLLM")
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input directory (processes all .jsonl files) or single .jsonl file"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        # default="/fast/nchandak/models/Qwen3-4B",
        default="/fast/nchandak/models/Llama-4-Scout",
        help="Path to the model to use for date extraction"
    )
    
    args = parser.parse_args()
    
    # Get files to process
    files_to_process = get_files_to_process(args.input_path)
    
    if not files_to_process:
        logger.error("No .jsonl files found to process")
        return
    
    logger.info(f"Found {len(files_to_process)} .jsonl files to process")
    
    # Process each file
    for file_path in files_to_process:
        try:
            process_file(file_path, args.model_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info("All files processed!")

if __name__ == "__main__":
    main()
