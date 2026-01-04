"""
Date processing for forecasting questions.

This module consolidates all date-related operations:
1. Extracting resolution dates using VLLM
2. Extracting and removing start dates from backgrounds
3. Updating resolution dates to minimum of available dates
4. Updating start dates to minimum of available dates

All date operations are centralized in the DateProcessor class.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgen.utils.date_utils import parse_date, normalize_date_format
from qgen.utils.xml_utils import extract_question_components
from qgen.utils.io_utils import load_articles_from_file, save_jsonl, get_files_to_process

logger = logging.getLogger(__name__)


class DateProcessor:
    """
    Unified date processing for forecasting questions.
    
    This class provides methods for:
    - Extracting resolution dates using VLLM
    - Extracting start dates from question backgrounds
    - Updating resolution/start dates to minimum of available dates
    - Removing start dates from backgrounds
    
    Example:
        >>> from qgen.inference.openrouter_inference import OpenRouterInference
        >>> engine = OpenRouterInference(model="meta-llama/llama-4-scout")
        >>> processor = DateProcessor(model_path="/path/to/model", inference_engine=engine)
        >>> articles = processor.load_from_file("questions.jsonl")
        >>> processor.extract_resolution_dates(articles)
        >>> processor.save_to_file(articles, "questions.jsonl")
    """
    
    def __init__(self, model_path: str = None, inference_engine=None):
        """
        Initialize the date processor.
        
        Args:
            model_path: Path to VLLM model (for resolution date extraction)
            inference_engine: Pre-initialized inference engine (alternative to model_path)
        """
        self.model_path = model_path
        self.inference_engine = inference_engine
    
    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load articles from a JSONL file."""
        return load_articles_from_file(file_path)
    
    def save_to_file(self, articles: List[Dict[str, Any]], file_path: str) -> None:
        """Save articles to a JSONL file."""
        save_jsonl(articles, file_path)
    
    def extract_resolution_dates(
        self, 
        articles: List[Dict[str, Any]], 
        skip_existing: bool = True,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract resolution dates using inference engine (OpenRouter or VLLM).
        
        Uses a language model to analyze question components and extract
        the resolution date mentioned in the question. Retries failed extractions
        up to max_retries times.
        
        Args:
            articles: List of article dictionaries
            skip_existing: If True, skip articles that already have resolution_date
            max_retries: Number of times to attempt extraction (default: 3)
            
        Returns:
            List of articles with resolution_date field added
        """
        # If inference_engine is provided (OpenRouter), use it
        if self.inference_engine is not None:
            return self._extract_resolution_dates_openrouter(articles, skip_existing, max_retries)
        
        # Otherwise, use VLLM
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            logger.error("VLLM not installed. Please install it with: pip install vllm")
            return articles
        
        # Initialize the model once
        logger.info(f"Loading model from {self.model_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        llm = LLM(
            model=self.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=4096,
            dtype="auto",
            trust_remote_code=True
        )
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.3,    # Lower temperature for more consistent extraction
            top_p=0.95,
            top_k=20,
            max_tokens=2048,
            stop=["<|im_end|>"]
        )
        
        # Perform extraction with retries
        for attempt in range(max_retries):
            logger.info(f"\n{'='*60}")
            logger.info(f"Extraction attempt {attempt + 1}/{max_retries}")
            logger.info(f"{'='*60}")
            
            # Filter articles that need resolution date extraction
            articles_to_process = []
            process_indices = []
            
            for i, article in enumerate(articles):
                # Skip if already has a valid resolution_date
                if skip_existing and self._is_valid_resolution_date(article):
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
                break
            
            logger.info(f"Processing {len(articles_to_process)} articles for resolution date extraction")
            
            # Create prompts for all articles
            logger.info("Creating prompts for all articles...")
            raw_prompts = []
            for article in articles_to_process:
                question_title, background, resolution_criteria = extract_question_components(article)
                prompt = self._create_date_extraction_prompt(question_title, background, resolution_criteria)
                raw_prompts.append(prompt)
            
            # Apply chat template
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for prompt in raw_prompts
            ]
            
            # Process all prompts together
            logger.info(f"Processing {len(prompts)} prompts with VLLM...")
            outputs = llm.generate(prompts, sampling_params)
            
            # Process outputs and update articles
            extracted_this_round = 0
            for i, (article, output) in enumerate(zip(articles_to_process, outputs)):
                response = output.outputs[0].text.strip()
                
                # Extract resolution date from response
                from qgen.utils.xml_utils import extract_tag_content
                resolution_date = extract_tag_content(response, 'answer')
                
                # Validate date format (basic check)
                if resolution_date and resolution_date != "UNKNOWN":
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', resolution_date):
                        article['resolution_date'] = resolution_date
                        extracted_this_round += 1
                    else:
                        article['resolution_date'] = "UNKNOWN"
                else:
                    article['resolution_date'] = "UNKNOWN"
                
                article['resolution_date_response'] = response
            
            logger.info(f"Attempt {attempt + 1}: Extracted {extracted_this_round} valid dates from {len(articles_to_process)} articles")
        
        # Count total successful extractions
        total_extracted = sum(1 for article in articles if self._is_valid_resolution_date(article))
        logger.info(f"\n{'='*60}")
        logger.info(f"Final Statistics:")
        logger.info(f"Total articles with valid resolution dates: {total_extracted}/{len(articles)}")
        logger.info(f"{'='*60}\n")
        
        return articles
    
    def _create_date_extraction_prompt(self, question_title: str, background: str, resolution_criteria: str) -> str:
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

**Important:** Only extract dates that are explicitly mentioned or can be reasonably inferred from the context. DO NOT MAKE UP DATES.

Please provide your answer in the following format:
<answer>YYYY-MM-DD</answer>

If you cannot determine a specific resolution date, respond with:
<answer>UNKNOWN</answer>
"""
        return prompt
    
    def _is_valid_resolution_date(self, article: Dict[str, Any]) -> bool:
        """
        Check if an article has a valid resolution date.
        
        Args:
            article: Article dictionary
            
        Returns:
            True if the article has a valid resolution_date, False otherwise
        """
        if 'resolution_date' not in article:
            return False
        
        resolution_date = article['resolution_date']
        
        # Check if it's not unknown and matches YYYY-MM-DD format
        if resolution_date and resolution_date != "UNKNOWN" and "unknown" not in resolution_date.lower():
            return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', resolution_date))
        
        return False
    
    def _extract_resolution_dates_openrouter(
        self, 
        articles: List[Dict[str, Any]], 
        skip_existing: bool = True,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract resolution dates using OpenRouter inference engine with retry logic.
        
        Args:
            articles: List of article dictionaries
            skip_existing: If True, skip articles that already have resolution_date
            max_retries: Number of times to attempt extraction
            
        Returns:
            List of articles with resolution_date field added
        """
        import asyncio
        from qgen.utils.xml_utils import extract_tag_content
        
        # Perform extraction with retries
        for attempt in range(max_retries):
            logger.info(f"\n{'='*60}")
            logger.info(f"Extraction attempt {attempt + 1}/{max_retries}")
            logger.info(f"{'='*60}")
            
            # Filter articles that need resolution date extraction
            articles_to_process = []
            process_indices = []
            
            for i, article in enumerate(articles):
                # Skip if already has a valid resolution_date
                if skip_existing and self._is_valid_resolution_date(article):
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
                break
            
            logger.info(f"Processing {len(articles_to_process)} articles for resolution date extraction using OpenRouter")
            
            # Create prompts for all articles
            prompts = []
            for article in articles_to_process:
                question_title, background, resolution_criteria = extract_question_components(article)
                prompt = self._create_date_extraction_prompt(question_title, background, resolution_criteria)
                prompts.append(prompt)
            
            # Process all prompts with OpenRouter
            async def process_all():
                results = await self.inference_engine.generate(prompts, batch_size=500)
                return results
            
            # Run async batch generation
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(process_all())
            
            # Process outputs and update articles
            extracted_this_round = 0
            for article, result in zip(articles_to_process, results):
                # Extract response text from result dictionary
                if result is None:
                    response = ""
                elif isinstance(result, dict):
                    response = result.get('response', '')
                else:
                    response = str(result)
                
                # Extract resolution date from response
                resolution_date = extract_tag_content(response, 'answer')
                
                # Validate date format (basic check)
                if resolution_date and resolution_date != "UNKNOWN":
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', resolution_date):
                        article['resolution_date'] = resolution_date
                        extracted_this_round += 1
                    else:
                        article['resolution_date'] = "UNKNOWN"
                else:
                    article['resolution_date'] = "UNKNOWN"
                
                article['resolution_date_response'] = response
            
            logger.info(f"Attempt {attempt + 1}: Extracted {extracted_this_round} valid dates from {len(articles_to_process)} articles")
        
        # Count total successful extractions
        total_extracted = sum(1 for article in articles if self._is_valid_resolution_date(article))
        logger.info(f"\n{'='*60}")
        logger.info(f"Final Statistics:")
        logger.info(f"Total articles with valid resolution dates: {total_extracted}/{len(articles)}")
        logger.info(f"{'='*60}\n")
        
        return articles
    
    def extract_and_remove_start_dates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract start dates from backgrounds and remove them.
        
        Identifies patterns like "Question Start Date: ..." in backgrounds,
        extracts the date, stores it in question_start_date field, and
        removes it from the background text.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of articles with start dates extracted and removed
        """
        processed_count = 0
        extracted_count = 0
        
        for article in articles:
            question_title, background, resolution_criteria = extract_question_components(article)
            
            if not background:
                continue
            
            # Clean leading punctuation
            if background.lower()[:2] == ". ":
                background = background[2:]
                article['background'] = background
            
            start_date_index = self._get_date_from_background(background)
            
            if start_date_index <= 0:
                continue
            
            start_date_ending_index = background.find(".", start_date_index)
            if start_date_ending_index == -1:
                continue
                
            start_date_str = background[:start_date_ending_index]
            cleaned_background = background[start_date_ending_index:]
            
            # Extract and normalize start date
            start_date = self._extract_start_date_from_text(start_date_str)
            
            # Update the entry if we found a start date or if background was cleaned
            if start_date or cleaned_background.lower() != background.lower():
                processed_count += 1
                
                # Update background
                if 'background' in article:
                    article['background'] = cleaned_background
                
                # Add start date field if we extracted one
                if start_date:
                    article['question_start_date'] = start_date
                    extracted_count += 1
        
        logger.info(f"Processed {processed_count} articles, extracted start dates from {extracted_count}")
        return articles
    
    def _get_date_from_background(self, background: str) -> int:
        """Get the position of the date in the background."""
        if not background:
            return -1
        
        years = re.findall(r'\b\d{4}\b', background)
        if "start date" not in background.lower():
            return -1
        
        # Return the position at the end of the first occurrence of year (if exists)
        for year in years:
            if year in background:
                return background.index(year) + len(year)
                
        return -1
    
    def _extract_start_date_from_text(self, date_text: str) -> str:
        """
        Extract start date from text and normalize it.
        
        Looks for patterns like "Question Start Date: 10th March 2024"
        and normalizes to YYYY-MM-DD format.
        """
        if not date_text:
            return ""
        
        # Define patterns for start date extraction
        start_date_patterns = [
            r'Question Start Date:\s*([^.]+?)(?:\.|$)',
            r'Start Date:\s*([^.]+?)(?:\.|$)',
            r'(?i)question start date:\s*([^.]+?)(?:\.|$)',
            r'(?i)start date:\s*([^.]+?)(?:\.|$)',
        ]
        
        extracted_date = ""
        
        # Try each pattern to find and extract start date
        for pattern in start_date_patterns:
            match = re.search(pattern, date_text)
            if match:
                # Extract the date portion
                date_str = match.group(1).strip()
                
                # Clean up common suffixes and prefixes
                date_str = re.sub(r'\s*[\.,;]\s*$', '', date_str)
                date_str = re.sub(r'^\s*[\.,;]\s*', '', date_str)
                
                # Try to normalize the date format
                normalized_date = normalize_date_format(date_str)
                if normalized_date:
                    extracted_date = normalized_date
                else:
                    extracted_date = date_str
                
                break
        
        return extracted_date
    
    def update_resolution_dates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update resolution_date to be the minimum of available date fields.
        
        Compares resolution_date, article_publish_date, article_modify_date,
        and article_download_date, and sets resolution_date to the minimum.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of articles with updated resolution_date
        """
        entries_updated = 0
        
        for article in articles:
            date_fields = [
                'resolution_date',
                'article_publish_date',
                'article_modify_date',
                'article_download_date'
            ]
            
            # If the resolution_date can't be parsed, skip this article
            if 'resolution_date' not in article or parse_date(article['resolution_date']) is None:
                continue

            # Parse all available dates
            parsed_dates = []
            for field in date_fields:
                if field in article and article[field] is not None:
                    parsed_date = parse_date(article[field])
                    if parsed_date is not None:
                        parsed_dates.append(parsed_date)
            
            if not parsed_dates:
                continue
            
            # Find the minimum date
            min_date = min(parsed_dates)
            original_date = article.get('resolution_date')
            
            # Update the resolution_date field with just the date part (YYYY-MM-DD format)
            article['resolution_date'] = min_date.strftime('%Y-%m-%d')
            
            if article['resolution_date'] != original_date:
                entries_updated += 1
        
        logger.info(f"Updated resolution_date for {entries_updated} articles")
        return articles
    
    def update_start_dates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update question_start_date to be the minimum of available date fields.
        
        Compares question_start_date, resolution_date, article_publish_date,
        article_modify_date, and article_download_date, and sets question_start_date
        to the minimum.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of articles with updated question_start_date
        """
        entries_updated = 0
        
        for article in articles:
            date_fields = [
                'question_start_date',
                'resolution_date',
                'article_publish_date',
                'article_modify_date',
                'article_download_date'
            ]
            
            # If the question_start_date can't be parsed, skip this article
            if 'question_start_date' not in article or parse_date(article['question_start_date']) is None:
                continue

            # Parse all available dates
            parsed_dates = []
            for field in date_fields:
                if field in article and article[field] is not None:
                    parsed_date = parse_date(article[field])
                    if parsed_date is not None:
                        parsed_dates.append(parsed_date)
            
            if not parsed_dates:
                continue
            
            # Find the minimum date
            min_date = min(parsed_dates)
            original_date = article.get('question_start_date')
            
            # Update the question_start_date field with just the date part (YYYY-MM-DD format)
            article['question_start_date'] = min_date.strftime('%Y-%m-%d')
            
            if article['question_start_date'] != original_date:
                entries_updated += 1
        
        logger.info(f"Updated question_start_date for {entries_updated} articles")
        return articles
    
    def process_directory(
        self, 
        input_path: str,
        extract_resolution: bool = False,
        extract_start: bool = False,
        update_resolution: bool = False,
        update_start: bool = False
    ) -> None:
        """
        Process all JSONL files in a directory with specified date operations.
        
        When extract_resolution is True:
        - Performs extraction with retries
        - Saves to new file with "_date_extracted.jsonl" suffix
        - Only keeps entries with valid resolution dates
        
        Args:
            input_path: Path to directory or single file
            extract_resolution: Whether to extract resolution dates using VLLM/OpenRouter
            extract_start: Whether to extract and remove start dates
            update_resolution: Whether to update resolution dates to minimum
            update_start: Whether to update start dates to minimum
        """
        files_to_process = get_files_to_process(input_path)
        
        if not files_to_process:
            logger.error("No .jsonl files found to process")
            return
        
        logger.info(f"Found {len(files_to_process)} .jsonl files to process")
        
        for file_path in files_to_process:
            try:
                logger.info(f"Processing file: {file_path}")
                articles = self.load_from_file(file_path)
                original_count = len(articles)
                save_path = file_path
                
                if extract_resolution:
                    articles = self.extract_resolution_dates(articles)
                    filtered_count = len(articles)
                    
                    # # Filter to only keep entries with valid resolution dates
                    # articles = [a for a in articles if self._is_valid_resolution_date(a)]
                    # filtered_count = len(articles)
                    
                    # logger.info(f"Filtered from {original_count} to {filtered_count} entries with valid resolution dates")
                    
                    # Save to new file with "_date_extracted.jsonl" suffix
                    path_obj = Path(file_path)
                    output_path = path_obj.parent / f"{path_obj.stem}_date_extracted.jsonl"
                    save_path = str(output_path)
                    self.save_to_file(articles, str(output_path))
                    logger.info(f"Saved {filtered_count} entries to {output_path}")
                
                if extract_start:
                    articles = self.extract_and_remove_start_dates(articles)
                
                if update_resolution:
                    articles = self.update_resolution_dates(articles)
                
                if update_start:
                    articles = self.update_start_dates(articles)
                
                # For non-extraction operations, save back to original file
                self.save_to_file(articles, save_path)
                logger.info(f"Updated {save_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info("All files processed!")

