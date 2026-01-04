"""
Common utilities for forecasting evaluation scripts.
Provides shared functions for:
- Model loading and tokenization
- Answer extraction and parsing
- Data loading and preprocessing
- Environment setup
"""

import re
import os
import torch
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

def setup_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(level: int = logging.INFO):
    """Configure logging with consistent format."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )
    return logging.getLogger(__name__)


def setup_environment():
    """Set environment variables for optimal performance."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# DATASET UTILITIES
# ============================================================================

def add_idx_column(dataset: Dataset) -> Dataset:
    """Add an 'idx' column to the dataset storing the original row index."""
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)


def parse_filename_for_dataset_info(file_path: str) -> Tuple[str, str]:
    """
    Extract news_source and num_lines from filename.
    
    Expected format: something_like: deepseek-chat-v3-0324_dw_30_free_1.jsonl
    Returns: (news_source, num_lines) like ("dw", "30")
    """
    import os
    filename = os.path.basename(file_path)
    name_without_ext = filename.replace('.jsonl', '').replace('.json', '')
    parts = name_without_ext.split('_')
    
    news_source = ""
    num_lines = ""
    
    # Try to find news_source_numlines pattern
    for i in range(len(parts) - 1):
        if parts[i] and parts[i+1].isdigit():
            news_source = parts[i]
            num_lines = parts[i+1]
            break
    
    # Look for common news sources
    if not news_source or not num_lines:
        common_sources = ['dw', 'cnn', 'cbsnews', 'foxnews', 'forbes', 'reuters', 'theguardian', 'bbc', 'ap', 'npr', 'wsj', 'nyt']
        for part in parts:
            for common_source in common_sources:
                if common_source in part.lower():
                    news_source = part.lower()
                    break
        
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


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================

def extract_answer(completion: str) -> Optional[str]:
    """Extract the final answer from LLM output using <answer></answer> tags."""
    if completion is None or not isinstance(completion, str):
        return None
    
    if not completion.strip():
        return None
    
    # Remove thinking section if present
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    
    try:
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
    """Extract probability from LLM output using <probability></probability> tags."""
    if completion is None or not isinstance(completion, str):
        return None
    
    if not completion.strip():
        return None
    
    # Remove thinking section if present
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


def extract_multiple_answers_and_probabilities(completion: str) -> Dict[str, float]:
    """
    Extract multiple answers and probabilities from LLM output.
    Expected format: <answer1>text</answer1> <probability1>0.5</probability1>
    
    Returns:
        Dictionary with answers as keys and probabilities as values
    """
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
        except (ValueError, TypeError):
            continue
    
    return answer_prob_dict


def extract_boxed_answer(completion: str) -> Optional[str]:
    """Extract answers from \\boxed{...} format (used in FutureX)."""
    if completion is None or not isinstance(completion, str):
        return None
    
    if not completion.strip():
        return None
    
    # Remove thinking section if present
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    
    try:
        # Look for \\boxed{...} pattern
        matches = re.finditer(r"\\boxed\{([^}]+)\}", completion, re.DOTALL)
        matches_list = list(matches)
    except:
        return None
    
    if not matches_list:
        return None
    
    # Get the last match
    last_match = matches_list[-1]
    answer_text = last_match.group(1).strip()
    
    return answer_text


def extract_question(choose_best_output: str) -> Dict[str, str]:
    """
    Extract question components from structured output.
    
    Args:
        choose_best_output: The output containing question components
        
    Returns:
        Dictionary with question components as keys
    """
    if not choose_best_output:
        return {}
    
    # Check for "NO GOOD QUESTION" case
    if "NO GOOD QUESTION" in choose_best_output.upper():
        return {}
    
    def extract_last_tag_block(text, tag):
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        last_open = text.rfind(open_tag)
        if last_open == -1:
            return ""
        start = last_open + len(open_tag)
        
        end = text.find(close_tag, start)
        if end == -1:
            return ""
        return text[start:end]

    tags = [
        "question_title",
        "background",
        "resolution_criteria",
        "answer",
        "answer_type"
    ]
    return_dict = {tag: extract_last_tag_block(choose_best_output, tag) for tag in tags}
    
    return return_dict


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(
    model_path: str, 
    model_name: str = None,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "auto"
) -> Tuple[LLM, AutoTokenizer]:
    """
    Load model with vLLM and tokenizer.
    
    Args:
        model_path: Path to model directory
        model_name: Optional model name (defaults to last directory in path)
        gpu_memory_utilization: GPU memory to use (0-1)
        dtype: Data type for model weights
        
    Returns:
        Tuple of (vLLM model, tokenizer)
    """
    logger = logging.getLogger(__name__)
    
    if model_name is None:
        model_name = model_path.rstrip("/").split("/")[-1]
    logger.info(f"Using model_name: {model_name}")
    logger.info(f"Loading model with vLLM from: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Check if model is multimodal
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            is_multimodal = hasattr(config, 'vision_config') or 'vision' in str(config).lower()
            logger.info(f"Detected multimodal model: {is_multimodal}")
        except:
            is_multimodal = False
        
        # Initialize vLLM model with tensor parallelism
        vllm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": torch.cuda.device_count(),
        }
        
        # Adjust settings for multimodal models
        if is_multimodal:
            logger.warning("Detected multimodal model. This may not be fully supported by vLLM.")
            vllm_kwargs["gpu_memory_utilization"] = 0.75
            vllm_kwargs["limit_mm_per_prompt"] = {"image": 0}
        
        model = LLM(**vllm_kwargs)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Trying alternative loading approach...")
        
        # Try different dtypes as fallback
        for fallback_dtype in ["bfloat16", "float16", "auto"]:
            try:
                logger.info(f"Attempting to load with dtype: {fallback_dtype}")
                model = LLM(
                    model=model_path,
                    trust_remote_code=True,
                    dtype=fallback_dtype,
                    gpu_memory_utilization=0.75,
                    tensor_parallel_size=1,
                    enforce_eager=True,
                )
                logger.info(f"Successfully loaded model with dtype: {fallback_dtype}")
                break
            except Exception as inner_e:
                logger.warning(f"Failed with dtype {fallback_dtype}: {inner_e}")
                if fallback_dtype == "auto":
                    raise RuntimeError(f"Could not load model with any dtype. Last error: {inner_e}")
        
    return model, tokenizer


# ============================================================================
# DATA LOADING
# ============================================================================

def load_questions_from_jsonl(file_path: str, logger=None) -> List[dict]:
    """
    Load questions from JSONL file and extract question components.
    
    Args:
        file_path: Path to JSONL file
        logger: Optional logger instance
        
    Returns:
        List of question dictionaries
    """
    import json
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    questions_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    article = json.loads(line.strip())
                    question_dict = {}
                    
                    # Extract question fields
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
                    
                    if len(question_dict) >= 5:
                        # Create full question entry
                        question_entry = {
                            'idx': line_idx,
                            'question_title': question_dict.get('question_title', ''),
                            'background': question_dict.get('background', ''),
                            'resolution_criteria': question_dict.get('resolution_criteria', ''),
                            'answer': question_dict.get('answer', ''),
                            'answer_type': question_dict.get('answer_type', ''),
                            'resolution_date': article.get('resolution_date', ''),
                            'question_start_date': article.get('date_begin', article.get('question_start_date', '')),
                            'question_close_date': article.get('date_close', article.get('question_close_date', '')),
                            'nr_forecasters': article.get('nr_forecasters', ''),
                            'resolution': article.get('resolution', ''),
                            'url': article.get('url', ''),
                            'prompt': article.get('prompt', ''),
                        }
                        
                        if question_entry['question_title'].strip():
                            questions_data.append(question_entry)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx}: {e}")
                    continue
    
    logger.info(f"Loaded {len(questions_data)} valid questions from {file_path}")
    return questions_data


# ============================================================================
# PROMPT UTILITIES
# ============================================================================

def apply_chat_template(
    tokenizer: AutoTokenizer,
    prompt: str,
    model_name: str = "",
    continue_final: bool = True
) -> str:
    """
    Apply chat template to prompt with model-specific handling.
    
    Args:
        tokenizer: Model tokenizer
        prompt: Raw prompt text
        model_name: Model name for model-specific handling
        continue_final: Whether to continue final message
        
    Returns:
        Formatted prompt with chat template applied
    """
    try:
        chat = [{"role": "user", "content": prompt}]
        
        if 'qwen3' in model_name.lower():
            return tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=True
            )
        else:
            return tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                continue_final_message=continue_final
            )
    except Exception as e:
        logging.getLogger(__name__).info(f"Error applying chat template: {e}")
        return prompt

