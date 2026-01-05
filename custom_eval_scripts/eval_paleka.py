"""
Evaluation script for Paleka's CCFLMF (Consistency Checks for Language Model Forecasting) benchmark.
Evaluates models on forecasting consistency checks from dpaleka/ccflmf dataset.
Tests whether models maintain consistent predictions across related questions.
Uses vLLM for efficient inference.

Usage:
    # Load from HuggingFace (default)
    python eval_paleka.py --model_dir=/path/to/model
    
    # Load from local files
    python eval_paleka.py --model_dir=/path/to/model --use_local --data_dir=/path/to/tuples_2028
    
    # Specify different config
    python eval_paleka.py --model_dir=/path/to/model --config_name=tuples_2026
"""

import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import hf_hub_download
from typing import Optional, List, Tuple, Dict, Any
from accelerate import Accelerator
from transformers import AutoTokenizer
from tqdm import tqdm 
import json
import os 
import logging
import time 
import sys
from typing import Callable
import glob

# Import vLLM for faster generation
from vllm import LLM, SamplingParams

# Set SEED
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Set cuDNN for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables to control threading for various libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def extract_probability(completion: str) -> Optional[float]:
    """
    Extracts the probability from the LLM's output.
    Returns the probability as a float.
    """
    matches = re.finditer(r"<probability>(.*?)<\/probability>", completion, re.DOTALL)
    matches_list = list(matches)

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


def format_forecasting_prompt_binary(
    question_title: str,
    resolution_criteria: str,
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a binary forecasting question.  You have to come up with the best probability estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {question_title}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""

    return prompt



def load_paleka_dataset_from_hf(config_name: str = "tuples_2028") -> Dict[str, List[dict]]:
    """
    Load the Paleka CCFLMF dataset from HuggingFace by downloading JSONL files directly.
    
    Args:
        config_name: Configuration name (e.g., "tuples_2028", "tuples_scraped", "tuples_newsapi")
        
    Returns:
        Dictionary mapping checker names to list of question data
    """
    # Mapping of config names to HuggingFace file paths
    CONFIG_TO_PATH = {
        "tuples_2028": "src/data/tuples/2028",
        "tuples_scraped": "src/data/tuples/scraped",
        "tuples_newsapi": "src/data/tuples/newsapi",
    }
    
    # All checker types
    CHECKER_TYPES = [
        "Neg", "And", "Or", "AndOr", "But", 
        "Cond", "CondCond", "Consequence", 
        "Paraphrase", "ExpectedEvidence"
    ]
    
    if config_name not in CONFIG_TO_PATH:
        logger.error(f"Unknown config: {config_name}. Available: {list(CONFIG_TO_PATH.keys())}")
        return None
    
    base_path = CONFIG_TO_PATH[config_name]
    logger.info(f"Loading dataset from HuggingFace: dpaleka/ccflmf, config: {config_name}")
    
    checker_data = {}
    
    try:
        for checker_type in CHECKER_TYPES:
            filename = f"{base_path}/{checker_type}Checker.jsonl"
            
            try:
                # Download the file from HuggingFace
                local_path = hf_hub_download(
                    repo_id="dpaleka/ccflmf",
                    filename=filename,
                    repo_type="dataset"
                )
                
                # Parse the JSONL file
                questions_data = []
                with open(local_path, 'r', encoding='utf-8') as f:
                    for line_idx, line in enumerate(f):
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                question_entry = {
                                    'idx': line_idx,
                                    'original_data': data,
                                    'checker_type': checker_type
                                }
                                questions_data.append(question_entry)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse line {line_idx} in {filename}: {e}")
                                continue
                
                checker_data[checker_type] = questions_data
                logger.info(f"Loaded {len(questions_data)} questions for {checker_type}Checker")
                
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")
                continue
        
        if not checker_data:
            logger.error("No checker data loaded from HuggingFace")
            return None
            
        return checker_data
        
    except Exception as e:
        logger.error(f"Failed to load dataset from HuggingFace: {e}")
        return None


def load_paleka_questions_from_jsonl(file_path: str) -> List[dict]:
    """
    Load questions from a Paleka JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries with question data
    """
    questions_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    # Store the original line data and add an index
                    question_entry = {
                        'idx': line_idx,
                        'original_data': data,
                        'file_path': file_path
                    }
                    questions_data.append(question_entry)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx} in {file_path}: {e}")
                    continue
    
    logger.info(f"Loaded {len(questions_data)} questions from {file_path}")
    return questions_data

def load_model_and_tokenizer(model_path: str, model_name: str = None):
    if model_name is None:
        model_name = model_path.rstrip("/").split("/")[-1]
    logger.info(f"Using model_name: {model_name}")

    logger.info(f"Loading model with vLLM from local directory: {model_path}")
    
    # Initialize vLLM model
    try:
        # Load tokenizer separately for prompt processing
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Check if model is multimodal (like Llama-4-Scout)
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            is_multimodal = hasattr(config, 'vision_config') or 'vision' in str(config).lower()
            logger.info(f"Detected multimodal model: {is_multimodal}")
        except:
            is_multimodal = False
        
        # Use bfloat16 for better compatibility, especially with multimodal models
        dtype = "auto" #  "bfloat16"
        
        # Initialize vLLM model with tensor parallelism
        vllm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "dtype": dtype,
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": torch.cuda.device_count(),
        }
        
        # For multimodal models, we might need different settings
        if is_multimodal:
            logger.warning("Detected multimodal model. This may not be fully supported by vLLM.")
            # Reduce GPU memory utilization for multimodal models
            vllm_kwargs["gpu_memory_utilization"] = 0.75
            # Try to disable vision processing if possible
            vllm_kwargs["limit_mm_per_prompt"] = {"image": 0}
        
        model = LLM(**vllm_kwargs)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Trying alternative loading approach...")
        
        # Alternative approach: try different dtypes and settings
        for dtype in ["bfloat16", "float16", "auto"]:
            try:
                logger.info(f"Attempting to load with dtype: {dtype}")
                model = LLM(
                    model=model_path,
                    trust_remote_code=True,
                    dtype=dtype,
                    gpu_memory_utilization=0.75,
                    tensor_parallel_size=1,  # Use single GPU to avoid multi-GPU issues
                    enforce_eager=True,  # Use eager mode for better compatibility
                )
                logger.info(f"Successfully loaded model with dtype: {dtype}")
                break
            except Exception as inner_e:
                logger.warning(f"Failed with dtype {dtype}: {inner_e}")
                if dtype == "auto":  # Last attempt
                    raise RuntimeError(f"Could not load model with any dtype. Last error: {inner_e}")
        
    return model, tokenizer

def evaluate_paleka_questions(
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    questions_data: List[dict],
    max_new_tokens: int = 8192,
    num_generations: int = 1,
) -> List[dict]:
    """
    Run batched inference on Paleka questions and return results in the required format
    """
    # First, prepare all prompts and metadata
    all_prompts = []
    prompt_metadata = []  # Store info about each prompt for later mapping
    
    # Process each question's original data structure
    for question_entry in questions_data:
        original_data = question_entry['original_data']
        # Support both local file loading (file_path) and HF loading (checker_type)
        if 'file_path' in question_entry:
            source_name = question_entry['file_path']
        else:
            # For HF-loaded data, construct filename from checker_type
            checker_type = question_entry.get('checker_type', 'Unknown')
            source_name = f"{checker_type}Checker.jsonl"
        
        # Process each component in the original data
        for component_key, component_data in original_data.items():
            if isinstance(component_data, dict) and 'title' in component_data:
                question_title = component_data.get('title', '')
                resolution_criteria = component_data.get('body', 'N/A')
                
                if question_title:
                    # Create prompt
                    prompt = format_forecasting_prompt_binary(question_title=question_title, resolution_criteria=resolution_criteria)
                    
                    try:
                        chat = [{"role": "user", "content": prompt}]
                        if 'qwen3' in model_name.lower():
                            formatted_prompt = tokenizer.apply_chat_template(
                                chat, tokenize=False, add_generation_prompt=True, enable_thinking=True
                            )
                        else:
                            formatted_prompt = tokenizer.apply_chat_template(
                                chat, tokenize=False, continue_final_message=True
                            )
                    except Exception as e:
                        logger.warning(f"Error in tokenizer.apply_chat_template: {e}")
                        formatted_prompt = prompt
                    
                    all_prompts.append(formatted_prompt)
                    
                    # Store metadata for this prompt
                    prompt_metadata.append({
                        'question_entry_idx': question_entry['idx'],
                        'component_key': component_key,
                        'question_title': question_title,
                        'source_name': source_name,
                        'original_data_idx': questions_data.index(question_entry)
                    })
    
    if not all_prompts:
        logger.warning("No valid prompts found!")
        return []
    
    # Configure sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=num_generations,
    )
    
    # Process all prompts with vLLM in a single batch
    logger.info(f"Starting generation with vLLM for {len(all_prompts)} prompts, {num_generations} generations each")
    start_time = time.time()
    
    # Generate completions using vLLM's batched API
    all_outputs = model.generate(all_prompts, sampling_params)
    
    end_time = time.time()
    logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Group results by original question entry
    results_by_question = {}
    extraction_success = 0
    
    # Process all outputs
    for i, outputs in enumerate(all_outputs):
        metadata = prompt_metadata[i]
        question_entry_idx = metadata['question_entry_idx']
        component_key = metadata['component_key']
        question_title = metadata['question_title']
        source_name = metadata['source_name']
        original_data_idx = metadata['original_data_idx']
        
        # Extract probability from the first generation
        generated_text = outputs.outputs[0].text
        
        # Process the response to extract probability
        if "</think>" in generated_text:
            generated_text = generated_text.split("</think>")[1]
        
        prob = extract_probability(generated_text)
        if prob is None:
            prob = 0.5  # Default probability if extraction fails
        else:
            extraction_success += 1
            
        # Initialize result structure for this question if not exists
        if original_data_idx not in results_by_question:
            results_by_question[original_data_idx] = {
                "line": {},
                "original_file": os.path.basename(source_name),
                "idx": question_entry_idx
            }
        
        # Add forecast for this component
        results_by_question[original_data_idx]["line"][component_key] = {
            "question": {
                "title": question_title
            },
            "forecast": {
                "prob": prob
            }
        }
    
    # Convert to list and filter out empty results
    all_results = []
    for original_data_idx in sorted(results_by_question.keys()):
        result = results_by_question[original_data_idx]
        if result["line"]:  # Only add if we have at least one component
            all_results.append(result)
    
    logger.info(f"Extraction success rate: {extraction_success / len(all_outputs) * 100:.2f}%")
    return all_results

def process_paleka_dataset(
    data_source: str,
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    max_new_tokens: int = 8192,
    num_generations: int = 1,
    use_hf: bool = True,
    config_name: str = "tuples_2028",
    fallback_data_dir: str = None,
):
    """
    Process Paleka dataset either from HuggingFace or local directory
    
    Args:
        data_source: Either HF config name or local directory path
        use_hf: If True, load from HuggingFace; otherwise load from local directory
        fallback_data_dir: Local directory to use if HF loading fails
    """
    if use_hf:
        # Try loading from HuggingFace
        checker_data = load_paleka_dataset_from_hf(config_name)
        
        if checker_data is None:
            logger.warning("Failed to load from HuggingFace, falling back to local files")
            use_hf = False
            # Use fallback directory
            if fallback_data_dir:
                data_source = fallback_data_dir
                logger.info(f"Using fallback directory: {data_source}")
        else:
            # Process each checker type
            for checker_name, questions_data in checker_data.items():
                logger.info(f"Processing {checker_name}Checker with {len(questions_data)} questions")
                
                # Generate forecasts
                results = evaluate_paleka_questions(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    questions_data=questions_data,
                    max_new_tokens=max_new_tokens,
                    num_generations=num_generations,
                )
                
                # Create output directory
                model_output_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)
                
                # Create output filename
                output_filename = f"{checker_name}Checker.jsonl"
                output_path = os.path.join(model_output_dir, output_filename)
                
                # Save results
                with open(output_path, 'w') as f:
                    for result in results:
                        output_line = {"line": result["line"]}
                        f.write(json.dumps(output_line) + '\n')
                
                logger.info(f"Saved {len(results)} results to {output_path}")
    
    if not use_hf:
        # Load from local directory
        data_dir = data_source
        jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        
        if not jsonl_files:
            logger.error(f"No JSONL files found in {data_dir}")
            return
        
        logger.info(f"Found {len(jsonl_files)} JSONL files to process")
        
        for jsonl_file in jsonl_files:
            logger.info(f"Processing file: {jsonl_file}")
            
            # Load questions from this file
            questions_data = load_paleka_questions_from_jsonl(jsonl_file)
            
            if not questions_data:
                logger.warning(f"No questions found in {jsonl_file}")
                continue
            
            # Generate forecasts
            results = evaluate_paleka_questions(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                questions_data=questions_data,
                max_new_tokens=max_new_tokens,
                num_generations=num_generations,
            )
            
            # Create output directory
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Create output filename
            input_filename = os.path.basename(jsonl_file)
            output_path = os.path.join(model_output_dir, input_filename)
            
            # Save results
            with open(output_path, 'w') as f:
                for result in results:
                    output_line = {"line": result["line"]}
                    f.write(json.dumps(output_line) + '\n')
            
            logger.info(f"Saved {len(results)} results to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/paleka", 
                       help="Base directory to save outputs")
    
    parser.add_argument('--model_dir', type=str, default="/fast/nchandak/models/Qwen3-1.7B", 
                       help="Model directory")
    parser.add_argument('--model', type=str, default="None", help="Model name")
    
    parser.add_argument('--max_new_tokens', type=int, default=16384, 
                       help="Maximum number of new tokens for generation")
    
    parser.add_argument('--data_dir', type=str, 
                       default="/fast/nchandak/forecasting/datasets/paleka/tuples_2028",
                       help="Directory containing Paleka JSONL files (used if --use_local is set)")
    
    parser.add_argument('--use_local', action='store_true',
                       help="Use local JSONL files instead of HuggingFace dataset")
    
    parser.add_argument('--config_name', type=str, default="tuples_2028",
                       help="HuggingFace dataset config name (e.g., tuples_2028)")
    
    parser.add_argument('--num_generations', type=int, default=1, 
                       help="Number of generations to use per prompt")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.base_save_dir, exist_ok=True)
    logger.info(f"Output directory: {args.base_save_dir}")

    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {gpu_count}")
    
    # Extract model name from model_dir if not provided
    model_name = args.model
    if args.model == "None":
        model_name = args.model_dir.rstrip("/").split("/")[-1]
    
    logger.info(f"Model name: {model_name}")
    
    if args.use_local:
        logger.info(f"Using local data directory: {args.data_dir}")
        data_source = args.data_dir
    else:
        logger.info(f"Using HuggingFace dataset: dpaleka/ccflmf, config: {args.config_name}")
        data_source = args.config_name
    
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_dir, model_name)
    
    # Process dataset
    process_paleka_dataset(
        data_source=data_source,
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        output_dir=args.base_save_dir,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        use_hf=not args.use_local,
        config_name=args.config_name,
        fallback_data_dir=args.data_dir,
    )
    
    logger.info("Processing complete!")
