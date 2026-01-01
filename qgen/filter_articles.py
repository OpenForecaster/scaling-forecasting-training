#!/usr/bin/env python3
import os
import argparse
import logging
import json
import sys
from typing import List, Dict, Any
from tqdm import tqdm
import time
from transformers import AutoTokenizer
import re

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


        
def extract_answer(response: str) -> str:
    """
    Extract a digit answer ('0' or '1') from model response.
    
    Args:
        response: Model response string
        
        Returns:
            Extracted answer as string ('0' or '1'), or '1' if not found
    """
    if not response:
        return ""
    
    # Try to extract from <answer> tags - use findall to get all matches and take the last one
    answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if answer_matches:
        last = answer_matches[-1].strip()
        digit_match = re.search(r'\b([01])\b', last)
        if digit_match:
            return digit_match.group(1)
    
    # Look for \boxed{} notation commonly used in math
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', response, re.DOTALL)
    if boxed_matches:
        last = boxed_matches[-1].strip()
        digit_match = re.search(r'\b([01])\b', last)
        if digit_match:
            return digit_match.group(1)
    
    # Fallback: look for phrases like "the answer is 0" or "the answer is 1"
    answer_phrases = [
        r'the answer is\s+([01])\b',
        r'I choose\s+([01])\b',
        r'answer:\s+([01])\b',
        r'final answer:?\s+([01])\b'
    ]
    
    for phrase in answer_phrases:
        matches = re.findall(phrase, response.lower(), re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    # As a last resort, search for all standalone 0 or 1 in the response and return the last one
    digit_matches = re.findall(r'\b([01])\b', response)
    if digit_matches:
        return digit_matches[-1]
    
    return "1"

def load_articles(file_path: str) -> List[Dict[str, Any]]:
    """Load articles from a JSONL file."""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line.strip()))
    return articles

def create_prompt(article: Dict[str, Any]) -> str:
    """Create a prompt for the LLM to evaluate article relevance."""
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

def create_question_prompt(question: str) -> str:
    """Create a prompt for the LLM to evaluate question relevance for forecasting."""
    
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

def evaluate_articles_with_vllm(articles: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
    """Evaluate articles using VLLM and add relevance field."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("VLLM not installed. Please install it with: pip install vllm")
        return articles
    
    # Initialize the model with tensor parallel size 8
    logger.info(f"Loading model from {model_path} with tensor parallel size 8...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        trust_remote_code=True
    )
    
    # Set up sampling parameters with updated values
    sampling_params = SamplingParams(
        temperature=0.6,    # Updated temperature
        top_p=0.95,        # Added top_p parameter
        top_k=20,          # Added top_k parameter
        max_tokens=4096,      # We only need short responses
        stop=["<|im_end|>"]  # Stop at the end of our expected format
    )
    
    # Create all prompts at once and apply chat template
    logger.info("Creating prompts for all articles and applying chat template...")
    raw_prompts = [create_prompt(article) for article in articles]
    # Use tokenizer's apply_chat_template with tokenize=False
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in raw_prompts
    ]
    logger.info(f"Example prompt: {prompts[0]}")
    
    # Process all prompts together
    logger.info(f"Processing {len(prompts)} prompts with VLLM...")
    outputs = llm.generate(prompts, sampling_params)
    # Process outputs and update articles
    logger.info(f"Loaded {len(articles)} articles for filtering.")
    
    results = []
    for i, (article, output) in enumerate(tqdm(zip(articles, outputs), desc="Processing results")):
        response = output.outputs[0].text.strip()
        if i == 0:
            logger.info(f"Example response: {response}")
            
        # Extract relevance from response
        relevance = int(extract_answer(response))
        
        # Add relevance field to article
        article['article_relevant'] = relevance
        # article['response'] = response
        results.append(article)
        
        # Log some examples for debugging
        if i < 5:  # Log first 5 articles
            logger.info(f"Article {i + 1}: Title='{article.get('article_title', '')[:50]}...' -> Relevant: {relevance}")

    num_relevant = sum(1 for article in results if article.get('article_relevant', 0) == 1)
    logger.info(f"Filtering complete. {num_relevant} out of {len(results)} articles marked as relevant.")
    
    
    return results

def evaluate_questions_with_vllm(articles: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
    """Evaluate questions using VLLM and add question_relevant field."""
    try:
        from vllm import LLM, SamplingParams
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
    
    # Initialize the model with tensor parallel size 8
    logger.info(f"Loading model from {model_path} with tensor parallel size 8...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        trust_remote_code=True
    )
    
    # Set up sampling parameters with updated values
    sampling_params = SamplingParams(
        temperature=0.6,    # Updated temperature
        top_p=0.95,        # Added top_p parameter
        top_k=20,          # Added top_k parameter
        max_tokens=4096,      # We only need short responses
        stop=["<|im_end|>"]  # Stop at the end of our expected format
    )
    
    # Create all prompts at once and apply chat template
    logger.info("Creating prompts for all questions and applying chat template...")
    raw_prompts = [create_question_prompt(article['final_question']) for article in valid_articles]
    # Use tokenizer's apply_chat_template with tokenize=False
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in raw_prompts
    ]
    logger.info(f"Example question prompt: {prompts[0]}")
    
    # Process all prompts together
    logger.info(f"Processing {len(prompts)} question prompts with VLLM...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process outputs and update articles
    logger.info(f"Processing {len(valid_articles)} question evaluation results...")
    
    for i, (article, output) in enumerate(tqdm(zip(valid_articles, outputs), desc="Processing question results")):
        response = output.outputs[0].text.strip()
        if i == 0:
            logger.info(f"Example question response: {response}")
            
        # Extract relevance from response
        relevance = int(extract_answer(response))
        
        # Add question relevance field to article
        article['question_relevant'] = relevance
        article['question_relevance_response'] = response
        
        # Log some examples for debugging
        if i < 5:  # Log first 5 questions
            question_preview = article.get('final_question', '')[:50]
            logger.info(f"Question {i + 1}: '{question_preview}...' -> Relevant: {relevance}")

    num_relevant = sum(1 for article in valid_articles if article.get('question_relevant', 0) == 1)
    logger.info(f"Question filtering complete. {num_relevant} out of {len(valid_articles)} questions marked as relevant.")
    
    return articles

def save_articles(articles: List[Dict[str, Any]], file_path: str, filter_articles: bool, filter_questions: bool):
    """Save articles with relevance fields back to the same JSONL file."""
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
        # Count articles with valid questions
        valid_questions_count = sum(1 for article in articles if article.get('final_question', '') and len(article.get('final_question', '').strip()) >= 10)
        question_relevant_count = sum(1 for article in articles if article.get('question_relevant', 0) == 1)
        logger.info(f"Question relevance: {question_relevant_count} out of {valid_questions_count} valid questions marked as relevant ({question_relevant_count/valid_questions_count*100:.1f}% of valid questions)" if valid_questions_count > 0 else "Question relevance: No valid questions found")

def main():
    parser = argparse.ArgumentParser(description="Filter articles and/or questions using LLM relevance evaluation")
    
    parser.add_argument(
        "--articles_path",
        type=str,
        default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.dw.com_2025-05_selected12.jsonl",
        help="Path to the articles JSONL file"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/fast/nchandak/models/Qwen3-32B",
        help="Path to the model to use for evaluation"
    )
    
    parser.add_argument(
        "--filter_articles",
        action="store_true",
        help="Filter articles for forecasting relevance (adds article_relevant field)"
    )
    
    parser.add_argument(
        "--filter_questions",
        action="store_true", 
        help="Filter questions for forecasting relevance (adds question_relevant field)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.articles_path):
        logger.error(f"Articles file not found: {args.articles_path}")
        return
    
    # Check that at least one filtering option is selected
    if not args.filter_articles and not args.filter_questions:
        logger.error("Please specify at least one filtering option: --filter_articles or --filter_questions")
        return
    
    # Load articles
    logger.info(f"Loading articles from {args.articles_path}")
    articles = load_articles(args.articles_path)
    logger.info(f"Loaded {len(articles)} articles")
    
    # Start timing
    start_time = time.time()
    
    # Filter articles if requested
    if args.filter_articles:
        logger.info("Starting article evaluation with VLLM...")
        articles = evaluate_articles_with_vllm(articles, args.model_path)
    
    # Filter questions if requested
    if args.filter_questions:
        logger.info("Starting question evaluation with VLLM...")
        articles = evaluate_questions_with_vllm(articles, args.model_path)
    
    evaluation_time = time.time() - start_time
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Save results back to the same file
    logger.info(f"Saving results back to {args.articles_path}")
    save_articles(articles, args.articles_path, args.filter_articles, args.filter_questions)
    
    logger.info("Filtering completed!")

if __name__ == "__main__":
    main()
