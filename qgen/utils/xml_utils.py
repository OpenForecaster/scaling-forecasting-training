"""
XML parsing and extraction utilities.

This module provides functions for:
- Extracting question components from XML-formatted strings
- Extracting content from XML tags
- Extracting final questions from complex XML structures
"""

import re
from typing import Dict, Any, Tuple


def extract_tag_content(text: str, tag: str) -> str:
    """
    Extract content from the last occurrence of an XML tag.
    
    Args:
        text: Text containing XML tags
        tag: Tag name (without < >)
        
    Returns:
        Content between opening and closing tags, or empty string if not found
        
    Example:
        >>> text = "<answer>42</answer>"
        >>> extract_tag_content(text, "answer")
        '42'
    """
    if not text:
        return ""
    
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


def extract_question_components(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract question components from an entry dictionary.
    
    Tries to extract from individual fields first, then falls back to
    parsing the 'final_question' field if it exists.
    
    Args:
        entry: Dictionary containing question data
        
    Returns:
        Tuple of (question_title, background, resolution_criteria)
        
    Example:
        >>> entry = {"question_title": "Who will win?", "background": "..."}
        >>> title, bg, criteria = extract_question_components(entry)
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
        if not question_title:
            question_title = extract_tag_content(final_question, 'question_title')
        if not background:
            background = extract_tag_content(final_question, 'background')
        if not resolution_criteria:
            resolution_criteria = extract_tag_content(final_question, 'resolution_criteria')
    
    return question_title, background, resolution_criteria


def extract_final_question(response_text: str) -> str:
    """
    Extract the final question content from a response string.
    
    This function extracts a complete question structure by finding the last
    occurrence of each required tag and assembling them into a complete question.
    
    Args:
        response_text: Text containing question XML structure
        
    Returns:
        Complete question wrapped in <q1> tags, or empty string if not found
        
    Example:
        >>> response = "Analysis... <question_title>Test</question_title>..."
        >>> question = extract_final_question(response)
    """
    if not response_text:
        return ""
    
    # Check for "NO GOOD QUESTION" case
    if "NO GOOD QUESTION" in response_text.upper():
        return ""
    
    # Extract each required tag by finding its last occurrence
    tags = [
        "question_id",
        "question_title",
        "background",
        "resolution_criteria",
        "answer",
        "answer_type"
    ]
    
    blocks = []
    for tag in tags:
        block = extract_tag_content(response_text, tag)
        if not block:
            # If any required tag is missing, return empty
            return ""
        # Reconstruct the tag with its content
        blocks.append(f"<{tag}>\n{block}\n</{tag}>")
    
    # Assemble complete question
    content = "\n".join(blocks)
    return f"<q1>\n{content}\n</q1>"


def extract_answer_from_response(response: str) -> str:
    """
    Extract answer from model response, looking for <answer> tags or common patterns.
    
    Args:
        response: Model response string
        
    Returns:
        Extracted answer as string, or empty string if not found
        
    Example:
        >>> response = "The answer is <answer>42</answer>"
        >>> extract_answer_from_response(response)
        '42'
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
    
    # Fallback: look for phrases like "the answer is X"
    answer_phrases = [
        r'the answer is\s+(.+?)(?:\.|$)',
        r'answer:\s+(.+?)(?:\.|$)',
        r'final answer:?\s+(.+?)(?:\.|$)'
    ]
    
    for phrase in answer_phrases:
        matches = re.findall(phrase, response.lower(), re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    return ""

