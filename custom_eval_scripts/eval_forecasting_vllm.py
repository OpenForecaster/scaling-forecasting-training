"""
General forecasting evaluation script using vLLM.
Evaluates models on binary forecasting datasets (Halawi, Metaculus, Manifold).
Extracts probability predictions and calculates Brier scores and accuracy.
Legacy script - prefer eval_binary.py for binary forecasting tasks.
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

def extract_final_answer0(generated_text: str) -> Optional[float]:
    """
    Extract the final answer (probability between 0 and 1) from text containing
    a substring like '*0.XX*'. Returns None if not found.
    """
    pattern = re.compile(r'\*(0(\.\d+)?|1(\.0+)?)\*')
    match = pattern.search(generated_text)
    if match:
        # matched string includes the asterisks, e.g. '*0.75*'
        final_str = match.group(0).strip('*')  # remove leading/trailing '*'
        return float(final_str)
    return None

def extract_answer(completion: str) -> Optional[float]:
    """
    Extracts the final answer from the LLM's output.
    """
    matches = re.finditer(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
    matches_list = list(matches)
    
    if not matches_list:
        return None
    
    # Get the last match
    last_match = matches_list[-1]
    answer_text = last_match.group(1).strip()
    
    try:
        prediction = float(answer_text)
    except:
        return None 
    
    if prediction < 0 or prediction > 1:
        return None 
    
    return prediction

def extract_final_answer(llm_output: str) -> Optional[float]:
    """
    Extracts the first probability prediction from the LLM's output.
    
    The prediction can be:
    - A decimal between 0 and 1, possibly wrapped in asterisks (e.g., *0.75*)
    - A percentage, possibly wrapped in asterisks (e.g., *75%*)
    
    Returns:
        A float between 0 and 1 representing the probability, or None if not found.
    """
    # Define regex patterns for different prediction formats
    patterns = [
        # Pattern for asterisk-wrapped percentage (e.g., *75%*)
        r'\*\s*(\d{1,3}(?:\.\d+)?)\s*%\s*\*',
        # Pattern for standalone percentage (e.g., 75%)
        r'(?<!\w)(\d{1,3}(?:\.\d+)?)\s*%(?!\w)',
        # Pattern for asterisk-wrapped decimal (e.g., *0.75*)
        r'\*\s*(0\.\d+)\s*\*',
        # Pattern for standalone decimal (e.g., 0.75)
        r'(?<!\w)(0\.\d+)(?!\w)',
    ]
    
    matches: List[Tuple[int, float]] = []
    
    for pattern in patterns:
        for match in re.finditer(pattern, llm_output):
            value = match.group(1)
            start_index = match.start()
            try:
                if '%' in match.group(0):
                    # Convert percentage to decimal
                    percentage = float(value)
                    if 0 <= percentage <= 100:
                        decimal = percentage / 100
                        matches.append((start_index, decimal))
                else:
                    # Direct decimal value
                    decimal = float(value)
                    if 0 <= decimal <= 1:
                        matches.append((start_index, decimal))
            except ValueError:
                continue  # If conversion fails, skip to the next match
    
    if not matches:
        return None
    
    # Sort matches based on their position in the text
    matches.sort(key=lambda x: x[0])
    
    # Return the decimal value of the earliest match
    return matches[0][1]

def format_forecasting_prompt(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str,
    zero_shot: bool = False
) -> str:
    """
    Format the prompt given the row data.
    """
    
    if zero_shot:
        return f"""I will ask you a forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. 
        
Question: {question}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}

Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
"""
    
    return f"""Question: {question}
    Question Background: {background}
    Resolution Criteria: {resolution_criteria}
    Question close date: {date_close}

    Instructions:
    1. Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
    {{{{ Insert rephrased and expanded question. }}}}
    2. Using your own knowledge of the world and topic, provide a few
    reasons why the answer might be no. Rate the strength of each reason.
    {{{{ Insert your thoughts }}}}
    3. Using your knowledge of the world and topic, as well as the information provided, provide a few
    reasons why the answer might be yes. Rate the strength of each reason.
    {{{{ Insert your thoughts }}}}
    4. Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver).
    {{{{ Insert your aggregated considerations }}}}
    5. Output an initial probability (prediction) given steps 1-4. It should be a number BETWEEN 0 and 100. For example, 
    if you are 75% confident the answer is yes, you would write 75.  
    {{{{ Insert initial probability }}}}
    6. Evaluate whether your calculated probability is excessively confident or not confident enough. Also,
    consider anything else that might affect the forecast that you did not before consider (e.g. base rate of
    the event).
    {{{{ Insert your thoughts }}}}
    7. Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end
    of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe 
    the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
    {{{{ Insert your answer }}}}
    """

def format_forecasting_prompt_old(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str
) -> str:
    """
    Format the prompt given the row data.
    """
    
    if ZERO_SHOT:
        return f"""Question: {question}
        Question Background: {background}
        Resolution Criteria: {resolution_criteria}
        Today's date: {date_begin}
        Question close date: {date_close}
        
        Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end
        of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe 
        the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
        """
    
    return f"""Question: {question}
    Question Background: {background}
    Resolution Criteria: {resolution_criteria}
    Today's date: {date_begin}
    Question close date: {date_close}

    Instructions:
    1. Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
    {{{{ Insert rephrased and expanded question. }}}}
    2. Using your own knowledge of the world and topic, provide a few
    reasons why the answer might be no. Rate the strength of each reason.
    {{{{ Insert your thoughts }}}}
    3. Using your knowledge of the world and topic, as well as the information provided, provide a few
    reasons why the answer might be yes. Rate the strength of each reason.
    {{{{ Insert your thoughts }}}}
    4. Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver).
    {{{{ Insert your aggregated considerations }}}}
    5. Output an initial probability (prediction) given steps 1-4. It should be a number BETWEEN 0 and 100. For example, 
    if you are 75% confident the answer is yes, you would write 75.  
    {{{{ Insert initial probability }}}}
    6. Evaluate whether your calculated probability is excessively confident or not confident enough. Also,
    consider anything else that might affect the forecast that you did not before consider (e.g. base rate of
    the event).
    {{{{ Insert your thoughts }}}}
    7. Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end
    of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe 
    the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
    {{{{ Insert your answer }}}}
    """

def load_model_and_tokenizer(model_path: str, model_name: str = None):
    if model_name is None:
        model_name = model_path.rstrip("/").split("/")[-1]
    logger.info(f"Using model_name: {model_name}")

    logger.info(f"Loading model with vLLM from local directory: {model_path}")
    
    # Initialize vLLM model
    try:
        # Load tokenizer separately for prompt processing
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize vLLM model with tensor parallelism
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.85,
            tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs for tensor parallelism
        )
    except:
        # Fallback if your particular directory structure requires it
        model_path += "/snapshots/model/"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.85,
            tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs for tensor parallelism
        )
        
    return model, tokenizer

def format_superforecasting_prompt(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str,
    zero_shot: bool = False
) -> str:
    """
    Format the prompt given the row data.
    """
    
    return f"""
Question: {question}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}
"""

def evaluate_model(
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset,
    max_new_tokens: int = 8192,
    batch_size: int = 32,
    format_prompt_fn: callable = format_forecasting_prompt,
    max_prompt_length: int = 4096,
    num_generations: int = 8,  # Added parameter for number of generations
    prompts_available: bool = False,
):
    """
    Run batched inference with multiple generations per prompt using vLLM
    """
    # Create prompts from each row
    all_prompts = []
    all_idxs = []
    all_resolutions = []
    
    for row in dataset:
        # Format the prompt for each example
        if prompts_available:
            local_prompt = row["prompt"]
        else :
            local_prompt = format_prompt_fn(
                question=row["question"],
                background=row["background"],
                resolution_criteria=row["resolution_criteria"],
                date_begin=row["date_begin"],
                date_close=row["date_close"],
                zero_shot=ZERO_SHOT,
            )
            
        try:
            chat = [{ 
                "role": "user",
                "content": f"You will be asked a forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. Show your work (reasoning) in <think> </think> tags. And return only the final answer (probability) in <answer> </answer> tags, for example if you think the event asked is 83% likely, then output <answer>0.83</answer>. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. Think step by step inside <think> tags."
            },
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
            prompt = format_forecasting_prompt(
                question=row["question"],
                background=row["background"],
                resolution_criteria=row["resolution_criteria"],
                date_begin=row["date_begin"],
                date_close=row["date_close"],
                zero_shot=ZERO_SHOT,
            )
            
        all_prompts.append(prompt)
        all_idxs.append(row["idx"])
        all_resolutions.append(float(row["resolution"]))
    
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
    
    # Process results
    all_results = []
    
    for i, outputs in enumerate(all_outputs):
        prompt = all_prompts[i]
        idx = all_idxs[i]
        actual = all_resolutions[i]
        
        # Process each generation for this prompt
        for gen_idx, output in enumerate(outputs.outputs):
            generated_text = output.text
            
            # Find where the prompt ends and the completion begins
            prompt_end_idx = generated_text.find("Let me solve this step by step.\n<think>")
            if prompt_end_idx == -1:
                # Fallback if the expected text isn't found
                prompt_end_idx = len(prompt)
                answer = generated_text
            else:
                answer = generated_text[prompt_end_idx:]
            
            # Extract the final probability
            final_prob = extract_answer(answer)
            skipped = False
            if final_prob is None:
                final_prob = 0.5
                skipped = True
            
            # Calculate token counts (approximate for vLLM)
            prompt_tokens = len(tokenizer.encode(prompt))
            completion_tokens = len(tokenizer.encode(answer))
            
            # Store result
            result = {
                "model": model_name,
                "prompt": prompt,
                "split": DATA_SPLIT,
                "data_type": DATA,
                "idx": idx,
                "generation_idx": gen_idx,
                "response": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "final_answer": float(final_prob),
                "resolution": float(actual),
                "skipped": skipped,
            }
            
            all_results.append(result)
    
    # Calculate metrics
    skipped_questions = len([result for result in all_results if result["skipped"]])
    logger.info(f"Skipped questions: {skipped_questions}")
    # Calculate metrics using all generations for each prompt
    # Group results by idx and generation_idx
    results_by_idx = {}
    for result in all_results:
        idx = result["idx"]
        gen_idx = result["generation_idx"]
        if idx not in results_by_idx:
            results_by_idx[idx] = {}
        results_by_idx[idx][gen_idx] = result
    
    # Get number of generations per prompt
    num_gens = max([max(results_by_idx[idx].keys()) for idx in results_by_idx]) + 1
    
    # Calculate metrics for each generation
    brier_scores = []
    accuracies = []
    skipped_questions = []
    for gen_idx in range(num_gens):
        gen_predictions = []
        gen_actuals = []
        skip_count = 0
        for idx in sorted(results_by_idx.keys()):
            if gen_idx in results_by_idx[idx]:
                gen_predictions.append(results_by_idx[idx][gen_idx]["final_answer"])
                gen_actuals.append(results_by_idx[idx][gen_idx]["resolution"])
                if results_by_idx[idx][gen_idx]["skipped"]:
                    skip_count += 1
        
        gen_predictions = np.array(gen_predictions, dtype=float)
        gen_actuals = np.array(gen_actuals, dtype=float)
        
        brier_score = np.mean((gen_predictions - gen_actuals) ** 2)
        predicted_binary = (gen_predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_binary == gen_actuals)
        
        brier_scores.append(brier_score)
        accuracies.append(accuracy)
        skipped_questions.append(skip_count)
    # Calculate mean and std dev of metrics
    mean_brier = np.mean(brier_scores)
    std_brier = np.std(brier_scores)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    logger.info(f"Brier Score: {mean_brier:.4f} ± {std_brier:.4f}")
    logger.info(f"Accuracy:    {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"Skipped questions: {np.mean(skipped_questions)}")
    logger.info("\n-------------------------------------------------------\n")
    
    # Also log metrics for each generation for backward compatibility
    for gen_idx in range(num_gens):
        logger.info(f"Generation {gen_idx} Brier Score: {brier_scores[gen_idx]:.4f}")
        logger.info(f"Generation {gen_idx} Accuracy:    {accuracies[gen_idx]:.4f}")
        logger.info(f"Generation {gen_idx} Number of skipped questions: {skipped_questions[gen_idx]}")
        logger.info("\n-------------------------------------------------------\n")
    
    # Log mean output token length with standard deviation
    completion_tokens = [result["completion_tokens"] for result in all_results]
    mean_output_length = np.mean(completion_tokens)
    std_output_length = np.std(completion_tokens)
    logger.info(f"Mean output token length: {mean_output_length:.2f} ± {std_output_length:.2f}")
    
    # Make output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save results
    output_file = f"{OUTPUT_DIR}{model_name}_{DATA_SPLIT}_size_{len(dataset)}_generations_{num_generations}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Saved {len(all_results)} generations to {output_file}")
    
    return brier_score, accuracy, skipped_questions

if __name__ == "__main__":
    import argparse
    from data_utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_save_dir', default=None, help="Where to save outputs of the model")
    
    parser.add_argument('--model_dir', type=str, default="/fast/rolmedo/models/qwen2.5-7b-it", help="Model directory")
    parser.add_argument('--model', type=str, default="None", help="Model name")
    
    # Add max_new_tokens arg
    parser.add_argument('--max_new_tokens', type=int, default=16384, help="Maximum number of new tokens for generation")
    
    parser.add_argument('--data_split', type=str, default="test", help="Data split to use")
    
    parser.add_argument('--data', type=str, default="halawi",
                      help="Which dataset to use")
    
    parser.add_argument('--num_generations', type=int, default=1, help="Number of generations to use per prompt")
    
    # Add tensor_parallel_size argument
    # parser.add_argument('--tensor_parallel_size', type=int, default=0, 
    #                   help="Tensor parallel size for vLLM. Set to 0 to use all available GPUs.")
    
    args = parser.parse_args()
    # Ensure the base save directory exists
    os.makedirs(args.base_save_dir, exist_ok=True)
    logger.info(f"Ensuring output directory exists: {args.base_save_dir}")

    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {gpu_count}")
    
    MODEL_DIR = args.model_dir
    DATA_SPLIT = args.data_split
    DATA = args.data
    
    if args.base_save_dir is not None:
        OUTPUT_DIR = args.base_save_dir
        if OUTPUT_DIR[-1] != "/":
            OUTPUT_DIR += "/"
    else:
        OUTPUT_DIR = OUTPUT_DIR + DATA + "/"
        
    base_save_dir = args.base_save_dir
    
    prompts_available = False
    if DATA == "halawi":
        # load training data
        dataset = load_halawi_data(split=DATA_SPLIT)
    elif DATA == "metaculus":
        dataset = load_metaculus_data(split=DATA_SPLIT)
    
    elif DATA == "manifold":
        dataset = load_manifold_data(split=DATA_SPLIT)
        # only keep 100 rows of dataset
        # dataset = dataset.select(range(20))
        prompts_available = True
        
    elif "menge" in DATA:
        data_type = DATA.split("_")[1]
        dataset = load_menge_data(split=DATA_SPLIT, data_type=data_type)
        prompts_available = True
        
    elif "infinitegames" in DATA:
        dataset = load_infinitegames_data(split=DATA_SPLIT)
        prompts_available = True
    
    elif "retrieval" in DATA or "reuters" in DATA:
        dataset = load_retreived_data(split=DATA_SPLIT, data_type=DATA)
        prompts_available = True
    
    logger.info(f"Data split: {DATA_SPLIT}")
    logger.info(f"Data type: {DATA}")
    logger.info(f"Dataset size: {len(dataset)}") 

    # shuffle dataset
    # dataset = dataset.shuffle(seed=SEED)
    # dataset = dataset.select(range(10))
    # logger.info(f"Actual dataset size: {len(train_dataset)}")

    # dataset = add_idx_column(dataset)
    # logger.info(f"Actual dataset size: {len(train_dataset)} Filtered ds size: {len(dataset)}")

    dataset = add_idx_column(dataset)
    new_tokens = args.max_new_tokens
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max new tokens: {new_tokens}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Model directory: {MODEL_DIR}")
    model_name = args.model
    
    # Extract model name from model_dir 
    if args.model == "None":
        model_name = MODEL_DIR.rstrip("/").split("/")[-1]
        # Remove any checkpoint suffix after model name
        if "checkpoint" in MODEL_DIR:
            model_name = MODEL_DIR.rstrip("/").split("/")[-2] + "__" + MODEL_DIR.rstrip("/").split("/")[-1]
        
    logger.info(f"Model name: {model_name}")
    
    model, tokenizer = load_model_and_tokenizer(args.model_dir, model_name)
    evaluate_model(model_name, model, tokenizer, dataset, max_new_tokens=new_tokens, format_prompt_fn=format_superforecasting_prompt, num_generations=args.num_generations, prompts_available=prompts_available)