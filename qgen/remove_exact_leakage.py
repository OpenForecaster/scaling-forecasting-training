#!/usr/bin/env python3
"""
Script to extract and remove exact leakage from question components.

This script:
1. Takes input as either a directory (processes all .jsonl files) or a single .jsonl file
2. For each entry, examines question_title, background, and resolution_criteria fields
3. Identifies and removes leakage patterns like "e.g.", "example:", etc.
4. Saves the modified entries back to the same files
"""

import os
import argparse
import logging
import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def remove_leakage_patterns(text: str, answer: str = "") -> Tuple[str, List[str]]:
    """
    Remove leakage patterns from text and return cleaned text with list of removed patterns.
    
    Args:
        text: The text to process
        answer: The answer string to check for in examples
        
    Returns:
        Tuple of (cleaned_text, list_of_removed_patterns)
    """
    if not text:
        return text, []
    
    removed_patterns = []
    cleaned_text = text
    
    # If we have an answer, first remove complete example phrases containing the answer
    if answer and answer.strip():
        # Patterns that might introduce examples containing the answer
        example_intro_patterns = [
            r'\b(?:e\.g\.|eg\.)\s*',
            r'\b(?:example|Example):\s*',
            r'\b(?:for example|For example)\s*',
            r'\b(?:such as|Such as)\s*',
            r'\b(?:like|Like)\s*',
            r'\b(?:including|Including)\s*',
            r'\b(?:specifically|Specifically)\s*',
            r'\b(?:namely|Namely)\s*',
            r'\b(?:i\.e\.|ie\.)\s*',
            r'\b(?:that is|That is)\s*',
            r'\b(?:in other words|In other words)\s*',
            r'\b(?:to illustrate|To illustrate)\s*',
            r'\b(?:as an example|As an example)\s*',
            r'\b(?:for instance|For instance)\s*',
            r'\b(?:take for example|Take for example)\s*',
            r'\b(?:consider|Consider)\s*',
            r'\b(?:suppose|Suppose)\s*',
            r'\b(?:imagine|Imagine)\s*',
            r'\b(?:let\'s say|Let\'s say)\s*',
            r'\b(?:say|Say)\s*',
            r'\b(?:assume|Assume)\s*',
            r'\b(?:if|If)\s*',
            r'\b(?:when|When)\s*',
        ]
        
        # Escape special regex characters in the answer
        escaped_answer = re.escape(answer.strip())
        
        # For each intro pattern, look for the pattern followed by content ending with the answer
        for intro_pattern in example_intro_patterns:
            # Pattern to match: intro_pattern + any content + answer + punctuation or end
            full_pattern = f"({intro_pattern})([^.!?]*?{escaped_answer}[^.!?]*?[.!?]?)"
            
            matches = re.findall(full_pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                intro, content = match
                full_example = intro + content
                removed_patterns.append(full_example)
                cleaned_text = cleaned_text.replace(full_example, '')
    
    # # Define leakage patterns to remove (for cases not containing the answer)
    # leakage_patterns = [
    #     # "e.g." or "eg." patterns (standalone)
    #     (r'\b(?:e\.g\.|eg\.)\s*', ''),
    #     # "example:" or "Example:" patterns (standalone)
    #     (r'\b(?:example|Example):\s*', ''),
    #     # "for example" patterns (standalone)
    #     (r'\b(?:for example|For example)\s*', ''),
    #     # "namely" patterns
    #     (r'\b(?:namely|Namely)\s*', ''),
    #     # "i.e." or "ie." patterns
    #     (r'\b(?:i\.e\.|ie\.)\s*', ''),
    #     # "that is" patterns
    #     (r'\b(?:that is|That is)\s*', ''),
    #     # "in other words" patterns
    #     (r'\b(?:in other words|In other words)\s*', ''),
    #     # "to illustrate" patterns
    #     (r'\b(?:to illustrate|To illustrate)\s*', ''),
    #     # "as an example" patterns
    #     (r'\b(?:as an example|As an example)\s*', ''),
    #     # "for instance" patterns
    #     (r'\b(?:for instance|For instance)\s*', ''),
    #     # "take for example" patterns
    #     (r'\b(?:take for example|Take for example)\s*', ''),
    #     # "consider" patterns (when used for examples)
    #     (r'\b(?:consider|Consider)\s*', ''),
    #     # "suppose" patterns (when used for examples)
    #     (r'\b(?:suppose|Suppose)\s*', ''),
    #     # "imagine" patterns (when used for examples)
    #     (r'\b(?:imagine|Imagine)\s*', ''),
    #     # "let's say" patterns
    #     (r'\b(?:let\'s say|Let\'s say)\s*', ''),
    #     # "say" patterns (when used for examples)
    #     (r'\b(?:say|Say)\s*', ''),
    #     # "suppose that" patterns
    #     (r'\b(?:suppose that|Suppose that)\s*', ''),
    #     # "assume that" patterns
    #     (r'\b(?:assume that|Assume that)\s*', ''),
    #     # "let's assume" patterns
    #     (r'\b(?:let\'s assume|Let\'s assume)\s*', ''),
    #     # "let's suppose" patterns
    #     (r'\b(?:let\'s suppose|Let\'s suppose)\s*', ''),
    #     # "let's say that" patterns
    #     (r'\b(?:let\'s say that|Let\'s say that)\s*', ''),
    #     # "for example, if" patterns
    #     (r'\b(?:for example, if|For example, if)\s*', ''),
    #     # "for instance, if" patterns
    #     (r'\b(?:for instance, if|For instance, if)\s*', ''),
    #     # "e.g., if" patterns
    #     (r'\b(?:e\.g\., if|eg\., if)\s*', ''),
    #     # "example: if" patterns
    #     (r'\b(?:example: if|Example: if)\s*', ''),
    # ]
    
    # # Apply each pattern
    # for pattern, replacement in leakage_patterns:
    #     # Check if pattern exists in text
    #     if re.search(pattern, cleaned_text, re.IGNORECASE):
    #         # Store the pattern that was found
    #         matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
    #         removed_patterns.extend(matches)
            
    #         # Remove the pattern
    #         cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
    
    # # Clean up extra whitespace and punctuation
    # cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single space
    # cleaned_text = re.sub(r'\s*,\s*', ', ', cleaned_text)  # Clean up commas
    # cleaned_text = re.sub(r'\s*\.\s*', '. ', cleaned_text)  # Clean up periods
    # cleaned_text = re.sub(r'\s*:\s*', ': ', cleaned_text)  # Clean up colons
    # cleaned_text = re.sub(r'\s*;\s*', '; ', cleaned_text)  # Clean up semicolons
    # cleaned_text = cleaned_text.strip()
    
    return cleaned_text, removed_patterns

def extract_question_components(entry: Dict[str, Any]) -> Tuple[str, str, str]:
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

def process_file(file_path: str) -> None:
    """Process a single file for leakage removal."""
    logger.info(f"Processing file: {file_path}")
    
    # Load articles from file
    articles = load_articles_from_file(file_path)
    if not articles:
        logger.warning(f"No articles loaded from {file_path}")
        return
    
    original_count = len(articles)
    processed_count = 0
    total_removed_patterns = 0
    
    logger.info(f"Loaded {original_count} articles")
    
    # Process each article
    for article in articles:
        # Extract question components
        question_title, background, resolution_criteria = extract_question_components(article)
        
        modified = False
        removed_patterns = []
        
        # Get the answer for comparison
        answer = article.get('answer', '')
        
        # Process question_title
        if question_title:
            cleaned_title, title_patterns = remove_leakage_patterns(question_title, answer)
            if cleaned_title != question_title:
                # Print before and after for question_title
                print(f"\n=== QUESTION TITLE LEAKAGE REMOVED ===")
                print(f"Answer: {answer}")
                print(f"BEFORE: {question_title}")
                print(f"AFTER:  {cleaned_title}")
                print(f"Removed patterns: {title_patterns}")
                print("=" * 50)
                
                article['question_title'] = cleaned_title
                modified = True
                removed_patterns.extend(title_patterns)
        
        # Process background
        if background:
            cleaned_background, background_patterns = remove_leakage_patterns(background, answer)
            if cleaned_background != background:
                # Print before and after for background
                print(f"\n=== BACKGROUND LEAKAGE REMOVED ===")
                print(f"Answer: {answer}")
                print(f"BEFORE: {background}")
                print(f"AFTER:  {cleaned_background}")
                print(f"Removed patterns: {background_patterns}")
                print("=" * 50)
                
                article['background'] = cleaned_background
                modified = True
                removed_patterns.extend(background_patterns)
        
        # Process resolution_criteria
        if resolution_criteria:
            cleaned_criteria, criteria_patterns = remove_leakage_patterns(resolution_criteria, answer)
            if cleaned_criteria != resolution_criteria:
                # Print before and after for resolution_criteria
                # print(f"\n=== RESOLUTION CRITERIA LEAKAGE REMOVED ===")
                # print(f"Answer: {answer}")
                # print(f"BEFORE: {resolution_criteria}")
                # print(f"AFTER:  {cleaned_criteria}")
                # print(f"Removed patterns: {criteria_patterns}")
                # print("=" * 50)
                
                article['resolution_criteria'] = cleaned_criteria
                modified = True
                removed_patterns.extend(criteria_patterns)
        
        # Update the entry if we found and removed leakage
        if modified:
            processed_count += 1
            total_removed_patterns += len(removed_patterns)
            
            # Log what was removed for debugging
            if removed_patterns:
                logger.debug(f"Removed patterns: {removed_patterns}")
    
    logger.info(f"Processed {processed_count} articles, removed {total_removed_patterns} leakage patterns")
    
    # Save results back to the same file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"Updated {file_path} - {processed_count} articles had leakage removed")
        
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Remove exact leakage from question components")
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input directory (processes all .jsonl files) or single .jsonl file"
    )
    
    args = parser.parse_args()
    
    # Get files to process
    files_to_process = get_files_to_process(args.input_path)
    
    if not files_to_process:
        logger.error("No .jsonl files found to process")
        return
    
    logger.info(f"Found {len(files_to_process)} .jsonl files to process")
    
    # Process each file
    total_processed = 0
    total_removed_patterns = 0
    
    for file_path in files_to_process:
        try:
            process_file(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info("All files processed!")

if __name__ == "__main__":
    main()
