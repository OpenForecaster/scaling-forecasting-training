"""
Evaluation script for binary forecasting questions.
Evaluates models on binary (yes/no) forecasting tasks from datasets like Metaculus, Manifold, etc.
Extracts probability predictions from model outputs and calculates Brier scores.
Uses vLLM for efficient batch inference.
"""

import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset
from typing import Optional, List, Tuple
from accelerate import Accelerator
from transformers import AutoTokenizer
from tqdm import tqdm 
import json
import os 
import logging
import time 
import sys
from typing import Callable

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

MODEL_DIR = ""
DATA_SPLIT = "train"
ZERO_SHOT = True
OUTPUT_DIR = "/fast/nchandak/forecasting/evals/manual/"
DATA = "halawi"

def add_idx_column(dataset: Dataset) -> Dataset:
    """
    Adds an 'idx' column to the dataset, storing the original row index.
    """
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)


def extract_answer(completion: str) -> Optional[str]:
    """
    Extracts the final answer from the LLM's output.
    Returns the raw answer text without type conversion.
    """
    matches = re.finditer(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
    matches_list = list(matches)
    
    if not matches_list:
        return None
    
    # Get the last match
    last_match = matches_list[-1]
    answer_text = last_match.group(1).strip()
    
    return answer_text

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

def extract_question(choose_best_output: str) -> str:
    """
    Extract the final question content from choose_best response.
    Always takes the last match if multiple matches are found.
    
    Args:
        choose_best_output: The output from choose_best processing
        
    Returns:
        Each block of the question extracted in a dictionary with key as the block name and value as the block content
    """
    import re
    
    if not choose_best_output:
        return ""
    
    # Check for "NO GOOD QUESTION" case
    if "NO GOOD QUESTION" in choose_best_output.upper():
        return ""
    
    # Fallback: For each tag, find the last opening tag and extract from there to its closing tag
    def extract_last_tag_block(text, tag):
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        last_open = text.rfind(open_tag)
        if last_open == -1:
            return ""
        start = last_open
        start += len(open_tag)
        
        end = text.find(close_tag, start)
        if end == -1:
            return ""
        # end += len(close_tag)
        return text[start:end]

    tags = [
        "question_title",
        "background",
        "resolution_criteria",
        "answer",
        "answer_type"
    ]
    return_dict = {tag: "" for tag in tags}
    
    blocks = []
    for tag in tags:
        block = extract_last_tag_block(choose_best_output, tag)
        return_dict[tag] = block
        
    # If no valid question structure found, return empty string
    # logger.warning("Could not extract valid question from choose_best output")
    return return_dict

def load_questions_from_jsonl(file_path: str) -> List[dict]:
    """
    Load articles with questions from JSONL file and extract question components.
    
    Args:
        file_path: Path to the JSONL file containing articles with final_question field
        
    Returns:
        List of dictionaries with extracted question components and article data
    """
    questions_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    article = json.loads(line.strip())
                    question_dict = {}
                    
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
                    
                    if len(list(question_dict.keys())) >= 5:
                        question_dict['resolution_date'] = article.get('resolution_date', '')
                        question_dict['question_start_date'] = article.get('date_begin', '')
                        question_dict['question_close_date'] = article.get('date_close', '')
                        
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
                            'question_close_date': question_dict.get('question_close_date', ''),
                            'nr_forecasters': article.get('nr_forecasters', ''),
                            'resolution': article.get('resolution', ''),
                            'url': article.get('url', ''),
                            'prompt': article.get('prompt', ''),
                        }
                        # Only add if we have a valid question title
                        if question_entry['question_title'].strip():
                            questions_data.append(question_entry)
                    
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx}: {e}")
                    continue
    
    logger.info(f"Loaded {len(questions_data)} valid questions from {file_path}")
    return questions_data

def old_format_forecasting_prompt(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer: str = "",
    answer_type: str = "",
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a forecasting question. You have to come up with the best guess for the final answer. Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = -1.25 whereas if the answer was correct, then your score would be - (1 - 0.5)^2 = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Try hard to come up with the best guess for the final answer. ONLY IF you cannot think of any answer, then just say "UNKNOWN" in the <answer> </answer> tags and assign a probability of 0 to it. REMEMBER THAT YOU SHOULD ALWAYS TRY TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""

    return prompt



def format_forecasting_prompt_no_article(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer: str = "",
    answer_type: str = "",
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best guess for the final answer (YES/NO). Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final answer (YES or NO) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. Thus, the range of the score is [-1, 0]. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be either YES or NO and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""

    return prompt




def format_forecasting_prompt_abstain(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer: str = "",
    answer_type: str = "",
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best guess for the final answer (YES/NO). Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final answer (YES or NO) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. Thus, the range of the score is [-1, 0]. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Try hard to come up with the best guess for the final answer. ONLY IF you cannot decide which resolution is more likely, then just say "UNKNOWN" in the <answer> </answer> tags and assign a probability of *0.5* to it (i.e., both outcomes being equally likely). REMEMBER THAT YOU SHOULD ALWAYS TRY TO MAXIMIZE YOUR SCORE.

Your final answer should be either YES or NO and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""

    return prompt



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
    prompt_fn: Callable = format_forecasting_prompt_no_article,
):
    """
    Run batched inference with multiple generations per prompt using vLLM
    """
    # Create prompts from each row
    all_prompts = []
    all_idxs = []
    all_row_data = []
    
    for i,row in enumerate(dataset):
        # Format the prompt for each example
        if prompts_available and row["prompt"] and row["prompt"].strip():
            local_prompt = row["prompt"]
        else:
            local_prompt = prompt_fn(
                question_title=row["question_title"],
                background=row["background"],
                resolution_criteria=row["resolution_criteria"],
                answer=row["answer"],
                answer_type=row["answer_type"],
            )
        try:
            chat = [
            {
                "role": "user",
                "content": local_prompt,
            },
            # {
            #     "role": "assistant",
            #     "content": "Let me reason about all the information provided step by step.\n<think>"
            # }
            ]
            if 'qwen3' in model_name.lower():
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, 
                                                        add_generation_prompt=True, enable_thinking=True)
            else:
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
        
        except Exception as e:
            logger.info(f"Error in tokenizer.apply_chat_template: {e}")
            prompt = prompt_fn(
                question_title=row["question_title"],
                background=row["background"],
                resolution_criteria=row["resolution_criteria"],
                answer=row["answer"],
                answer_type=row["answer_type"],
            )
            
        all_prompts.append(prompt)
        
        if i < 1:
            logger.info(f"Prompt: {prompt}")
            
        all_idxs.append(row["idx"])
        all_row_data.append(row)
    
    # Configure sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=num_generations,  # Number of generations per prompt
    )
    
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
            
            if "</think>" in answer:
                answer = answer.split("</think>")[1]
                
            final_prob = extract_probability(answer)
            last_ans = "YES"
            if "<answer>" in answer:
                last_ans = extract_answer(answer)
                
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
            # Additional fields requested
            "article_url": row.get("url", ""),
            "question_title": row.get("question_title", ""),
            "resolution_date": row.get("resolution_date", ""),
            "question_start_date": row.get("question_start_date", ""), 
            "question_close_date": row.get("question_close_date", ""),
            "nr_forecasters": row.get("nr_forecasters", ""),
            "resolution": row.get("resolution", ""),
            "background": row.get("background", ""),
            "resolution_criteria": row.get("resolution_criteria", ""),
            "answer": row.get("answer", ""),
            "answer_type": row.get("answer_type", ""),
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
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/binary", help="Base directory to save outputs")
    
    parser.add_argument('--model_dir', type=str, default="/fast/nchandak/models/Qwen3-1.7B", help="Model directory")
    parser.add_argument('--model', type=str, default="None", help="Model name")
    
    # Add max_new_tokens arg
    parser.add_argument('--max_new_tokens', type=int, default=16384, help="Maximum number of new tokens for generation")
    
    parser.add_argument('--data_split', type=str, default="test", help="Data split to use")
    
    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/metaculus_test.jsonl",
                      help="Path to JSONL file containing articles with final_question field")
    
    parser.add_argument('--num_generations', type=int, default=5, help="Number of generations to use per prompt")
    parser.add_argument('--prompts_not_available', action='store_true', help="Whether to use prompts available in the dataset")
    
    args = parser.parse_args()
    
    if "metaculus" in args.questions_file:
        dataset_name = "metaculus"
    elif "manifold" in args.questions_file:
        dataset_name = "manifold"
    else :
        dataset_name = "unknown"
        
    last_folder_name = args.questions_file.split("/")[-2]
    dataset_name = dataset_name + "_" + last_folder_name
    
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
    dataset = Dataset.from_list(questions_data)
    
    logger.info(f"Data split: {DATA_SPLIT}")
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Dataset size: {len(dataset)}") 

    dataset = add_idx_column(dataset)
    new_tokens = args.max_new_tokens
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max new tokens: {new_tokens}")
    logger.info(f"Model directory: {MODEL_DIR}")
    
    model_name = args.model
    
    # Extract model name from model_dir 
    if args.model == "None":
        model_name = MODEL_DIR.rstrip("/").split("/")[-1]
        # Remove any checkpoint suffix after model name
        # if "checkpoint" in MODEL_DIR:
        #     model_name = MODEL_DIR.rstrip("/").split("/")[-2] + "__" + MODEL_DIR.rstrip("/").split("/")[-1]
    
    # prompt_fn = format_forecasting_prompt_abstain
    prompt_fn = format_forecasting_prompt_no_article
    
    prompt_available = not args.prompts_not_available
    
    if not prompt_available:
        if prompt_fn == format_forecasting_prompt_abstain:
            model_name = model_name + "-abstain"
        elif prompt_fn == format_forecasting_prompt_no_article:
            model_name = model_name + "-p2"
    
    
    logger.info(f"Model name: {model_name}")
    logger.info(f"Prompts available: {prompt_available}")
    
    
    # Create output filename 
    output_file = os.path.join(
        output_base_dir, 
        f"{model_name}_{DATA_SPLIT}_size_{len(dataset)}_generations_{args.num_generations}.jsonl"
    )
    
    # if output file already exists, skip
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Skipping...")
        sys.exit(0)
    
    model, tokenizer = load_model_and_tokenizer(args.model_dir, model_name)
    
    # Run evaluation
    all_results = evaluate_model(
        model_name, 
        model, 
        tokenizer, 
        dataset, 
        max_new_tokens=new_tokens, 
        num_generations=args.num_generations, 
        prompts_available=prompt_available,
        prompt_fn=prompt_fn,
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