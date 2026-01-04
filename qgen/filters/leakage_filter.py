"""
Leakage detection and removal for forecasting questions.

This module consolidates three approaches to handling answer leakage:
1. LLM-based leakage removal (LeakageRemover class)
2. Pattern-based exact leakage removal (remove_exact_leakage_patterns)
3. Filtering questions with detected leakage (filter_by_leakage)

Leakage occurs when the answer to a question appears in the question text,
background, or resolution criteria, making the question trivially answerable.
"""

import os
import json
import logging
import asyncio
import re
from typing import List, Dict, Tuple, Any
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgen.utils.xml_utils import extract_question_components
from qgen.utils.io_utils import load_articles_from_file, save_jsonl

logger = logging.getLogger(__name__)


class LeakageRemover:
    """
    LLM-based leakage removal for forecasting questions.
    
    This class uses a language model to detect and remove answer leakage
    from question components. It can process entire datasets incrementally.
    
    Attributes:
        inference_engine: Engine for text generation
        use_freeq: Whether processing free-form or MCQ questions
        question_generator: Reuses question generator methods for leakage checking
    
    Example:
        >>> from qgen.inference.openrouter_inference import OpenRouterInference
        >>> engine = OpenRouterInference(model="meta-llama/llama-4-maverick")
        >>> remover = LeakageRemover(inference_engine=engine, use_freeq=True)
        >>> entries = remover.load_jsonl("questions.jsonl")
        >>> await remover.remove_leakage_from_entries(entries, "questions.jsonl")
    """
    
    def __init__(self, inference_engine, use_freeq: bool = True):
        """
        Initialize the leakage remover.
        
        Args:
            inference_engine: Engine for text generation (must implement BaseInference)
            use_freeq: If True, process free-form questions, else MCQ questions
        """
        self.inference_engine = inference_engine
        self.use_freeq = use_freeq
        
        # Import here to avoid circular dependency
        from qgen.qgen_core.question_generator import ForecastingQuestionGenerator
        
        # Create a question generator instance to reuse its methods
        self.question_generator = ForecastingQuestionGenerator(
            inference_engine=inference_engine,
            use_freeq=use_freeq,
            check_leakage=True,
            leakage_engine=inference_engine
        )
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file."""
        return load_articles_from_file(file_path)
    
    def save_jsonl(self, data: List[Dict], file_path: str) -> None:
        """Save data to JSONL file."""
        save_jsonl(data, file_path)
    
    def extract_question_from_entry(self, entry: Dict) -> str:
        """
        Extract the question text from a data entry.
        
        Tries multiple possible fields where the question might be stored.
        """
        question_fields = [
            'final_question',
            'choose_best_response', 
            'generated_questions',
            'question'
        ]
        
        for field in question_fields:
            if field in entry and entry[field]:
                question_text = entry[field]
                if isinstance(question_text, str) and len(question_text.strip()) > 10:
                    # If it's from choose_best_response, extract the final question
                    if field == 'choose_best_response':
                        return self.question_generator.extract_final_question(question_text)
                    return question_text
        
        return ""
    
    def needs_leakage_removal(self, entry: Dict) -> bool:
        """
        Check if entry needs leakage removal processing.
        
        Args:
            entry: Dictionary containing question data
            
        Returns:
            True if needs processing, False otherwise
        """
        # Skip if already processed successfully
        if len(entry.get('leakage_removed_question', '')) >= 10:
            return False
        
        if entry.get('leakage_removed', 0) == 1:
            return False
        
        # Skip if no valid question found
        question_text = self.extract_question_from_entry(entry)
        if not question_text or len(question_text.strip()) <= 10:
            return False
            
        return True

    async def remove_leakage_from_entries(
        self, 
        entries: List[Dict], 
        input_file: str, 
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Remove leakage from all entries in the list with incremental saving.
        
        Args:
            entries: List of data entries
            input_file: Path to save results incrementally (overwrites input file)
            batch_size: Number of entries to process in parallel
            
        Returns:
            List of entries with leakage removed
        """
        # Filter entries that need processing
        pending_entries = []
        pending_indices = []
        
        for i, entry in enumerate(entries):
            if self.needs_leakage_removal(entry):
                pending_entries.append(entry)
                pending_indices.append(i)
        
        logger.info(f"Found {len(pending_entries)} entries that need leakage removal out of {len(entries)} total")
        
        if not pending_entries:
            logger.info("No entries need leakage removal processing")
            return entries
        
        # Process entries in batches
        for i in range(0, len(pending_entries), batch_size):
            batch_entries = pending_entries[i:i+batch_size]
            batch_indices = pending_indices[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(pending_entries) + batch_size - 1)//batch_size}...")
            
            # Extract questions and prepare prompts
            questions = []
            valid_indices = []
            
            for j, entry in enumerate(batch_entries):
                question_text = self.extract_question_from_entry(entry)
                if question_text and len(question_text.strip()) > 10:
                    questions.append(question_text)
                    valid_indices.append(j)
                else:
                    logger.warning(f"Skipping entry: No valid question found")
            
            if not questions:
                logger.info("No valid questions in this batch, skipping...")
                continue
            
            # Prepare leakage check prompts using the question generator's method
            leakage_prompts = self.question_generator._prepare_leakage_check_prompts(
                questions, batch_entries[:len(questions)]
            )
            
            # Filter out SKIP prompts
            valid_prompts = []
            prompt_to_entry_map = []
            for idx, prompt in enumerate(leakage_prompts):
                if prompt != "SKIP":
                    valid_prompts.append(prompt)
                    prompt_to_entry_map.append(valid_indices[idx])
            
            if not valid_prompts:
                logger.info("No valid prompts in this batch, skipping...")
                continue
            
            try:
                # Generate corrected questions
                corrected_texts = await self.inference_engine.generate(
                    valid_prompts, 
                    batch_size=len(valid_prompts)
                )
                
                # Update entries with corrected questions
                for prompt_idx, corrected_text in enumerate(corrected_texts):
                    if prompt_idx < len(prompt_to_entry_map):
                        entry_idx = prompt_to_entry_map[prompt_idx]
                        entry = batch_entries[entry_idx]
                        original_idx = batch_indices[entry_idx]
                        
                        # Check if corrected_text is valid (not None or empty)
                        if corrected_text is not None and len(str(corrected_text).strip()) > 10:
                            # Store the corrected question
                            entry['leakage_removed_question'] = self.question_generator.extract_final_question(corrected_text)
                            entry['leakage_removed'] = 1
                            
                            # Update the original entry in the full list
                            entries[original_idx] = entry
                        else:
                            # Handle None or empty responses
                            entry['leakage_removed'] = 0
                            entry['leakage_removal_error'] = "Received None or empty response from inference engine"
                            entries[original_idx] = entry
                            logger.warning(f"Failed to process entry {original_idx + 1}/{len(entries)}: Received None or empty response")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Mark failed entries
                for j in valid_indices:
                    entry = batch_entries[j]
                    original_idx = batch_indices[j]
                    entry['leakage_removed'] = 0
                    entry['leakage_removal_error'] = str(e)
                    entries[original_idx] = entry
            
            # Save results incrementally after each batch
            logger.info(f"Saving results after batch {i//batch_size + 1}...")
            self.save_jsonl(entries, input_file)
        
        # Calculate and log final statistics
        total_processed = len(pending_entries)
        successful = sum(1 for entry in entries if entry.get('leakage_removed', 0) == 1)
        failed = sum(1 for entry in entries if 'leakage_removal_error' in entry)
        skipped = len(entries) - total_processed
        
        logger.info("\n" + "="*60)
        logger.info("LEAKAGE REMOVAL STATISTICS (LLM-based)")
        logger.info("="*60)
        logger.info(f"Total entries in dataset: {len(entries)}")
        logger.info(f"Entries processed for leakage removal: {total_processed}")
        logger.info(f"Successfully corrected: {successful}")
        logger.info(f"Failed to correct: {failed}")
        logger.info(f"Already processed (skipped): {skipped}")
        if total_processed > 0:
            logger.info(f"Success rate: {successful/total_processed*100:.1f}%")
        logger.info("="*60)
        
        logger.info("Finished leakage removal processing")
        return entries


def remove_exact_leakage_patterns(text: str, answer: str = "") -> Tuple[str, List[str]]:
    """
    Remove exact leakage patterns from text using regex.
    
    This function identifies and removes example phrases that contain the answer,
    particularly those introduced by phrases like "e.g.", "for example", etc.
    
    Args:
        text: The text to process
        answer: The answer string to check for in examples
        
    Returns:
        Tuple of (cleaned_text, list_of_removed_patterns)
        
    Example:
        >>> text = "The winner will be announced. For example, Bob Dylan won in 2016."
        >>> answer = "Bob Dylan"
        >>> cleaned, patterns = remove_exact_leakage_patterns(text, answer)
        >>> print(cleaned)  # "The winner will be announced."
    """
    if not text:
        return text, []
    
    removed_patterns = []
    cleaned_text = text
    
    # If we have an answer, remove complete example phrases containing the answer
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
    
    return cleaned_text, removed_patterns


def remove_exact_leakage_from_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact leakage patterns from all entries using regex-based pattern matching.
    
    This function processes all entries and removes example phrases that contain
    the answer from question components (background, resolution_criteria).
    
    Args:
        entries: List of question dictionaries
        
    Returns:
        List of entries with patterns removed
        
    Example:
        >>> entries = load_jsonl("questions.jsonl")
        >>> cleaned_entries = remove_exact_leakage_from_entries(entries)
    """
    total_entries = len(entries)
    entries_with_patterns = 0
    total_patterns_removed = 0
    component_breakdown = {
        'background': 0,
        'resolution_criteria': 0
    }
    
    for entry in entries:
        answer = entry.get('answer', '')
        if not answer or not answer.strip():
            continue
        
        entry_had_patterns = False
        
        # Process background
        background = entry.get('background', '')
        if background:
            cleaned_bg, removed = remove_exact_leakage_patterns(background, answer)
            if removed:
                entry['background'] = cleaned_bg
                total_patterns_removed += len(removed)
                component_breakdown['background'] += 1
                entry_had_patterns = True
        
        # Process resolution_criteria
        resolution_criteria = entry.get('resolution_criteria', '')
        if resolution_criteria:
            cleaned_rc, removed = remove_exact_leakage_patterns(resolution_criteria, answer)
            if removed:
                entry['resolution_criteria'] = cleaned_rc
                total_patterns_removed += len(removed)
                component_breakdown['resolution_criteria'] += 1
                entry_had_patterns = True
        
        if entry_had_patterns:
            entries_with_patterns += 1
    
    # Log statistics
    logger.info("\n" + "="*60)
    logger.info("PATTERN-BASED LEAKAGE REMOVAL STATISTICS")
    logger.info("="*60)
    logger.info(f"Total entries processed: {total_entries}")
    logger.info(f"Entries with patterns found: {entries_with_patterns}")
    logger.info(f"Total patterns removed: {total_patterns_removed}")
    if entries_with_patterns > 0:
        logger.info(f"Average patterns per affected entry: {total_patterns_removed/entries_with_patterns:.1f}")
    logger.info(f"\nPatterns removed by component:")
    for component, count in component_breakdown.items():
        if count > 0:
            logger.info(f"  - {component}: {count} entries modified")
    logger.info("="*60)
    
    return entries


def filter_by_leakage(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filter out entries where the answer appears in question components.
    
    This function checks if the answer is present in the question_title,
    background, resolution_criteria, or answer_type fields and removes
    entries with detected leakage.
    
    Args:
        entries: List of question dictionaries
        
    Returns:
        Tuple of (filtered_entries, count_of_filtered_out_entries)
        
    Example:
        >>> entries = [{"question_title": "Who won?", "answer": "Bob", ...}]
        >>> filtered, removed_count = filter_by_leakage(entries)
        >>> print(f"Removed {removed_count} entries with leakage")
    """
    filtered_entries = []
    leakage_count = 0
    leakage_breakdown = {
        'question_title': 0,
        'background': 0,
        'resolution_criteria': 0,
        'answer_type': 0
    }
    
    for entry in entries:
        answer = entry.get('answer', '')
        if not answer or not answer.strip():
            filtered_entries.append(entry)
            continue
        
        # Extract question components
        question_title, background, resolution_criteria = extract_question_components(entry)
        answer_type = entry.get('answer_type', '')
        
        # Check if answer appears in any of the components
        components_to_check = {
            'question_title': question_title,
            'background': background,
            'resolution_criteria': resolution_criteria,
            'answer_type': answer_type
        }
        
        # Clean the answer for comparison
        clean_answer = answer.strip().lower()
        
        has_leakage = False
        leaked_in = []
        for component_name, component_text in components_to_check.items():
            if component_text and clean_answer in component_text.lower():
                has_leakage = True
                leaked_in.append(component_name)
                leakage_breakdown[component_name] += 1
        
        if has_leakage:
            leakage_count += 1
        else:
            filtered_entries.append(entry)
    
    # Log statistics
    total_entries = len(entries)
    kept_entries = len(filtered_entries)
    
    logger.info("\n" + "="*60)
    logger.info("LEAKAGE FILTERING STATISTICS (Answer in Question)")
    logger.info("="*60)
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Entries with leakage detected: {leakage_count}")
    logger.info(f"Entries kept (no leakage): {kept_entries}")
    if total_entries > 0:
        logger.info(f"Filtering rate: {leakage_count/total_entries*100:.1f}% removed")
    logger.info(f"\nLeakage breakdown by component:")
    for component, count in leakage_breakdown.items():
        if count > 0:
            logger.info(f"  - {component}: {count} entries")
    logger.info("="*60)
    
    return filtered_entries, leakage_count


