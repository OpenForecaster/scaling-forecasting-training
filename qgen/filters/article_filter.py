"""
Article filtering and selection for forecasting question generation.

This module combines two major functionalities:
1. Filtering articles by relevance using VLLM (ArticleFilter class)
2. Selecting and sampling articles based on criteria (ArticleSelector class)

The ArticleFilter evaluates whether articles are suitable for generating
forecasting questions, while ArticleSelector handles the mechanics of
selecting subsets based on date ranges, word counts, and sampling strategies.
"""

import os
import argparse
import logging
import json
import sys
import re
from typing import List, Dict, Any
from tqdm import tqdm
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgen.qgen_core.article_processor import ArticleProcessor

logger = logging.getLogger(__name__)


class ArticleFilter:
    """
    VLLM-based article relevance filtering for forecasting.
    
    Uses a language model to evaluate whether articles are relevant
    for forecasting question generation based on:
    - Broad interest (>= 1000 people would care)
    - Not minute details (material drivers/outcomes)
    
    Example:
        >>> filter = ArticleFilter(model_path="/path/to/model")
        >>> articles = filter.load_articles("articles.jsonl")
        >>> filtered = filter.evaluate_articles(articles)
        >>> filter.save_articles(filtered, "articles_filtered.jsonl")
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the article filter.
        
        Args:
            model_path: Path to VLLM-compatible model
        """
        self.model_path = model_path
    
    def load_articles(self, file_path: str) -> List[Dict[str, Any]]:
        """Load articles from a JSONL file."""
        articles = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line.strip()))
        return articles
    
    def save_articles(
        self, 
        articles: List[Dict[str, Any]], 
        file_path: str,
        filter_articles: bool = False,
        filter_questions: bool = False
    ) -> None:
        """Save articles with relevance fields back to a JSONL file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        total_count = len(articles)
        logger.info(f"Updated {total_count} articles in {file_path}")
        
        # Log article relevance statistics if articles were filtered
        if filter_articles:
            article_relevant_count = sum(1 for article in articles if article.get('article_relevant', 0) == 1)
            logger.info(f"Article relevance: {article_relevant_count} out of {total_count} articles marked as relevant ({article_relevant_count/total_count*100:.1f}%)")
        
        # Log question relevance statistics if questions were filtered
        if filter_questions:
            valid_questions_count = sum(1 for article in articles if article.get('final_question', '') and len(article.get('final_question', '').strip()) >= 10)
            question_relevant_count = sum(1 for article in articles if article.get('question_relevant', 0) == 1)
            if valid_questions_count > 0:
                logger.info(f"Question relevance: {question_relevant_count} out of {valid_questions_count} valid questions marked as relevant ({question_relevant_count/valid_questions_count*100:.1f}% of valid questions)")
    
    def evaluate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate articles using VLLM and add article_relevant field.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of articles with article_relevant field added
        """
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            logger.error("VLLM not installed. Please install it with: pip install vllm")
            return articles
        
        # Initialize the model with tensor parallel size 8
        logger.info(f"Loading model from {self.model_path} with tensor parallel size 8...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        llm = LLM(
            model=self.model_path,
            tensor_parallel_size=8,
            trust_remote_code=True
        )
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=4096,
            stop=["<|im_end|>"]
        )
        
        # Create all prompts at once and apply chat template
        logger.info("Creating prompts for all articles and applying chat template...")
        raw_prompts = [self._create_article_prompt(article) for article in articles]
        
        # Use tokenizer's apply_chat_template with tokenize=False
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
        logger.info(f"Loaded {len(articles)} articles for filtering.")
        
        results = []
        for i, (article, output) in enumerate(tqdm(zip(articles, outputs), desc="Processing results")):
            response = output.outputs[0].text.strip()
            
            # Extract relevance from response
            relevance = int(self._extract_answer(response))
            
            # Add relevance field to article
            article['article_relevant'] = relevance
            results.append(article)
            
            # Log some examples for debugging
            if i < 5:  # Log first 5 articles
                logger.info(f"Article {i + 1}: Title='{article.get('article_title', '')[:50]}...' -> Relevant: {relevance}")

        num_relevant = sum(1 for article in results if article.get('article_relevant', 0) == 1)
        logger.info(f"Filtering complete. {num_relevant} out of {len(results)} articles marked as relevant.")
        
        return results
    
    def evaluate_questions(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate questions using VLLM and add question_relevant field.
        
        Args:
            articles: List of article dictionaries with questions
            
        Returns:
            List of articles with question_relevant field added
        """
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            logger.error("VLLM not installed. Please install it with: pip install vllm")
            return articles
        
        # Filter articles that have valid final_question field
        valid_articles = []
        question_indices = []
        for i, article in enumerate(articles):
            final_question = article.get('final_question', '')
            
            final_question_valid = article.get('final_question_valid', 1)
            if not final_question_valid:
                continue
            
            no_good_question = article.get('no_good_question', 0)
            if no_good_question:
                continue
            
            if final_question and len(final_question.strip()) >= 10:
                valid_articles.append(article)
                question_indices.append(i)
            else:
                # Set question_relevant to 0 for articles without valid questions
                article['question_relevant'] = 0
        
        if not valid_articles:
            logger.info("No valid questions found (final_question field missing or < 10 characters)")
            return articles
        
        logger.info(f"Found {len(valid_articles)} articles with valid questions out of {len(articles)} total")
        
        # Initialize the model
        logger.info(f"Loading model from {self.model_path} with tensor parallel size 8...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        llm = LLM(
            model=self.model_path,
            tensor_parallel_size=8,
            trust_remote_code=True
        )
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=4096,
            stop=["<|im_end|>"]
        )
        
        # Create all prompts at once and apply chat template
        logger.info("Creating prompts for all questions and applying chat template...")
        raw_prompts = [self._create_question_prompt(article['final_question']) for article in valid_articles]
        
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in raw_prompts
        ]
        
        # Process all prompts together
        logger.info(f"Processing {len(prompts)} question prompts with VLLM...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Process outputs and update articles
        for i, (article, output) in enumerate(tqdm(zip(valid_articles, outputs), desc="Processing question results")):
            response = output.outputs[0].text.strip()
            
            # Extract relevance from response
            relevance = int(self._extract_answer(response))
            
            # Add question relevance field to article
            article['question_relevant'] = relevance
            article['question_relevance_response'] = response

        num_relevant = sum(1 for article in valid_articles if article.get('question_relevant', 0) == 1)
        logger.info(f"Question filtering complete. {num_relevant} out of {len(valid_articles)} questions marked as relevant.")
        
        return articles
    
    def _create_article_prompt(self, article: Dict[str, Any]) -> str:
        """Create a prompt for evaluating article relevance."""
        title = article.get('title', '')
        description = article.get('description', '')
        maintext = article.get('maintext', '')
        publication_date = article.get('date_download', '')
        
        prompt = f"""You are an expert news-event analyst. Your job is to decide whether the news article provided is **forecast-relevant**. A forecast-relevant article IF IT SATISFIES ALL THE FOLLOWING CRITERIA:

**DECISION CRITERIA**  
1. **Broad Interest** - Would >= 1000 people plausibly care about the event?  
   - "Yes" examples: national policy changes, large-cap earnings, major sports finals, widely-used product launches, natural-disaster alerts.  
   - "No" examples: a small remote town's zoning notice, schedule of a random local event.

2. **Not Minute Details** - The article is not relevant if its main point is merely operational minutiae, daily price ticks, gossip, or incremental line-item edits. It should discuss a material driver or outcome.  

If BOTH THE criteria above are met, ONLY THEN is the article **forecast-relevant**. Otherwise, it is **not forecast-relevant**.

Article Title: {title}
Article Publication Date: {publication_date}
Article Description: {description}
Article Text: {maintext}

Consider whether this article contains information that would be valuable for forecasting purposes and whether it covers events that people would want to know about in advance.

RESPOND ONLY WITH "<answer>1</answer>" if the article is relevant for forecasting or "<answer>0</answer>" if it is not relevant.

**Output Format**
<answer>1/0</answer>
"""
        return prompt
    
    def _create_question_prompt(self, question: str) -> str:
        """Create a prompt for evaluating question relevance."""
        prompt = f"""You are an expert forecasting analyst. Your job is to decide whether the given question is **forecasting-relevant**. A forecasting-relevant question should satisfy ALL THE FOLLOWING CRITERIA:

**DECISION CRITERIA**  

1. **Broad Interest** - Would >= 1000 people plausibly care about the (outcome of the) event covered in the question?  
   - "Yes" examples: national policy changes, large-cap earnings, major sports finals, widely-used product launches, natural-disaster alerts. 
   - "No" examples: trivial personal events, extremely narrow technical details, schedule of a random local event.
   - These examples given above are only for reference. They are not exhaustive. Please use your own judgement to determine relevance.  
   
2. **Validity**: Check if the QUESTION is STATED in a forward-looking manner. Even if the question is in the past, it should be stated in a forward-looking manner. Also check whether the difference between the start date and resolution date is at least a single day.

3. **Definite Answer**: EXTRACT THE ACTUAL ANSWER TO THE QUESTION PROVIDED IN ITS <answer> </answer> TAG. The extracted answer should be short, definite, well-defined and not uncertain or vague. It SHOULD NOT BE A PHRASE OR A RANGE like "between XYZ and ABC" or "above XYZ" or "below PQR".

4. **Single Correct Answer**: ANALYZE WHETHER THE QUESTION CAN HAVE MULTIPLE OUTCOMES OR RIGHT ANSWERS. IF SO, THE QUESTION FAILS THIS CRITERIA. OTHERWISE, ENSURE THAT THE PROVIDED ANSWER IS THE SOLE CORRECT ANSWER TO THE QUESTION. IT SHOULD NOT BE THE CASE THAT THE QUESTION CAN HAVE MULTIPLE (DISTINCT) CORRECT ANSWERS.

Question: {question}

Consider whether this question satisfies ALL THE ABOVE CRITERIA or not.

RESPOND ONLY WITH "<answer>1</answer>" if the question is forecasting-relevant (satisfies ALL THE ABOVE CRITERIA) or "<answer>0</answer>" if it is not relevant.

**Output Format**
<answer>1/0</answer>
"""
        return prompt
    
    def _extract_answer(self, response: str) -> str:
        """Extract a digit answer ('0' or '1') from model response."""
        if not response:
            return ""
        
        # Try to extract from <answer> tags
        answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
        if answer_matches:
            last = answer_matches[-1].strip()
            digit_match = re.search(r'\b([01])\b', last)
            if digit_match:
                return digit_match.group(1)
        
        # Fallback: look for standalone 0 or 1
        digit_matches = re.findall(r'\b([01])\b', response)
        if digit_matches:
            return digit_matches[-1]
        
        return "1"


class ArticleSelector:
    """
    Article selection and sampling utilities.
    
    Provides methods for:
    - Loading and filtering articles by date ranges
    - Filtering by language and word count
    - Random sampling with yearly balancing
    - Saving selected subsets
    
    Example:
        >>> selector = ArticleSelector("/path/to/articles.jsonl")
        >>> selector.load_articles(language_code="en")
        >>> selector.filter_by_years([2021, 2022])
        >>> selector.filter_by_date_cutoff("2023-01")
        >>> selector.filter_by_min_word_count(100)
        >>> selected = selector.random_sample(1000, balance_yearly=True)
        >>> selector.save_selected(selected, "output/selected.jsonl")
    """
    
    def __init__(self, article_path: str):
        """
        Initialize the article selector.
        
        Args:
            article_path: Path to article file or directory
        """
        self.processor = ArticleProcessor(article_path)
    
    def load_articles(self, limit: int = None, language_code: str = None) -> List[Dict]:
        """Load articles with optional language filtering."""
        return self.processor.load_articles(limit=limit, language_code=language_code)
    
    def filter_by_years(self, years: List[int]) -> List[Dict]:
        """Filter articles by max date year."""
        return self.processor.filter_by_years(years)
    
    def filter_by_date_cutoff(self, cutoff_date: str, reverse: bool = False) -> List[Dict]:
        """Filter articles by date cutoff."""
        return self.processor.filter_by_date_cutoff(cutoff_date, reverse=reverse)
    
    def filter_by_min_word_count(self, min_word_count: int = 100, log_stats: bool = True) -> List[Dict]:
        """Filter articles by minimum word count."""
        return self.processor.filter_by_min_word_count(min_word_count, log_stats=log_stats)
    
    def random_sample(self, sample_size: int, balance_yearly: bool = False) -> List[Dict]:
        """Randomly sample articles."""
        return self.processor.random_sample(sample_size, balance_yearly=balance_yearly)
    
    def save_selected(self, articles: List[Dict], output_path: str) -> None:
        """Save selected articles to a file."""
        with open(output_path, 'w') as f:
            for article in articles:
                f.write(json.dumps(article) + '\n')
        logger.info(f"Saved {len(articles)} articles to {output_path}")
    
    def get_articles(self) -> List[Dict]:
        """Get the current list of articles."""
        return self.processor.articles


