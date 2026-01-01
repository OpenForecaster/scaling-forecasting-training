import re
import torch
import numpy as np
import asyncio
import aiohttp
from tqdm import tqdm
from datasets import Dataset
from typing import Optional, List
import os 
import logging
import json
import time
import httpx 
import sys

# Import OpenRouter API Key
from open_router_key import API_KEY

logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppresses INFO logs from httpx

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL_DIR = ""
# Update OUTPUT_DIR to be set dynamically based on data argument
OUTPUT_DIR = None  # Will be set based on args.data

# List of Llama-2-70B variants available on OpenRouter
MODEL_VARIANTS = [
    # "meta-llama/llama-3.1-8b-instruct:free",
    # "meta-llama/llama-3.1-8b-instruct",
    # "meta-llama/llama-3.3-70b-instruct",
    
    # "qwen/qwen-2.5-72b-instruct",
    # "qwen/qwen-2.5-7b-instruct",
    
    # "mistralai/mistral-small-24b-instruct-2501",
    
    # "deepseek/deepseek-r1-distill-llama-70b",
    # "deepseek/deepseek-r1-distill-qwen-14b",
    
    # "deepseek/deepseek-chat",
    # "deepseek/deepseek-r1:free",
    # "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3-0324",
    
    # "anthropic/claude-3.7-sonnet",
    
    # "x-ai/grok-3-beta",
    # "x-ai/grok-3-mini-beta",
    
    # "google/gemini-2.5-pro-preview-03-25",
    
    # "meta-llama/llama-4-maverick",
    # "meta-llama/llama-4-scout",
    
    # "qwen/qwq-32b-preview",
    # "deepseek/deepseek-r1-distill-qwen-1.5b",
    # "deepseek/deepseek-r1-distill-llama-8b",
    # "deepseek/deepseek-r1-distill-qwen-14b",
    # "deepseek/deepseek-r1-distill-qwen-32b",
    # "deepseek/deepseek-r1-distill-llama-70b",
    
    # "meta-llama/llama-3.1-405b-instruct",
    # "meta-llama/llama-3.1-405b",
    
    # "meta-llama/llama-2-70b",
    # "togethercomputer/llama-2-70b-chat",
    
    # "google/gemini-2.0-flash-001",
    # "openai/o3-mini",
    # "openai/o3-mini-high",
    # "google/gemma-3-27b-it",
    # "openai/gpt-4o-mini-search-preview"
]



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
    # match = pattern.search(generated_text)
    try :
        matches = pattern.findall(generated_text)
    except Exception as e:
        # logger.warning(f"Error with regex: {e}")
        return None
    
    if matches:
        # matched string includes the asterisks, e.g. '*0.75*'
        # final_str = match.group(-1).strip('*')  # remove leading/trailing '*'
        final_str = matches[-1][0].strip('*')
        return float(final_str)
    return None


def extract_last_decimal(text: str) -> Optional[float]:
    """
    Extract the last decimal number (0.xxx) from the given text.
    Returns None if no valid number is found.
    """
    pattern = re.compile(r'0\.\d+')  # Match numbers like 0.xxx
    matches = pattern.findall(text)  # Find all occurrences
    
    if matches:
        return float(matches[-1])  # Return the last matched decimal
    return None

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


def format_superforecasting_prompt(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str,
) -> str:
    """
    TODO!!!! 
    Format the prompt given the row data.
    """
    
    return f"""I will ask you a forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. 
    
    Question: {question}
    Question Background: {background}
    Resolution Criteria: {resolution_criteria}
    Question close date: {date_close}
    
    Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end
    of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe 
    the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
    """
    
def extract_answer1(generated_text: str) -> Optional[str]:
    """
    Extract the answer1 from the generated text. Find the last occurrence of <answer1> and </answer1> tags.
    """
    pattern = re.compile(r'<answer1>(.*?)</answer1>', re.DOTALL)
    try:
        matches = pattern.findall(generated_text)
    except Exception as e:
        return None
    
    if matches:
        ans = matches[-1].strip()
        # Clean up the answer to handle multiline responses
        ans = ans.strip().upper()
        # Extract just the letter if there's more text
        if ans and ans[0] in ['A', 'B', 'C', 'D']:
            ans = ans[0]
        
        if ans not in ['A', 'B', 'C', 'D']:
            return None
        return ans
    
    return None

def extract_answer2(generated_text: str) -> Optional[float]:
    """
    Extract the answer2 from the generated text. Find the last occurrence of <answer2> and </answer2> tags.
    """
    pattern = re.compile(r'<answer2>(.*?)</answer2>', re.DOTALL)
    try:
        matches = pattern.findall(generated_text)
    except Exception as e:
        return None
    
    if matches:
        ans = matches[-1].strip()
        # Don't convert to uppercase for numerical values
        try:
            return float(ans)
        except Exception as e:
            return None
    
    return None

def detailed_extraction(generated_text: str):
    """
    Extract the answer1 and answer2 from the generated text.
    Find the last occurence of <A>, <B>, <C>, <D> tags and extract the numerical probability between them.
    """
    values = []
    atleast_one_match = False
    for letter in ['A', 'B', 'C', 'D']:
        pattern = re.compile(rf'<{letter}>(.*?)</{letter}>', re.DOTALL)
        try:
            matches = pattern.findall(generated_text)
            if matches:
                values.append(float(matches[-1].strip()))
                atleast_one_match = True
        except Exception as e:
            values.append(-1)
            pass

    if not atleast_one_match:
        return None, None, []
    
    # Find the index with maximum value
    max_index = np.argmax(values)
    answer1 = chr(max_index + ord('A'))
    answer2 = values[max_index]
    
    return answer1, answer2, values

def create_prompts(dataset: Dataset, zero_shot: bool = False) -> List[str]:
    
    # Create prompts from each row
    prompts = [
        format_forecasting_prompt(
            question=row["question"],
            background=row["background"],
            resolution_criteria=row["resolution_criteria"],
            date_begin=row["date_begin"],
            date_close=row["date_close"],
            zero_shot=zero_shot
        )
        for row in dataset
    ]
    return prompts

DELTA = 0

# OpenRouter API URL
API_URL = "https://openrouter.ai/api/v1/chat/completions"

async def async_query_openrouter(session, model, prompt, max_tokens=1024):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    temp = 0.6 if "deepseek" in model else 0.2
    # temp = 0.6 if "claude" in model else temp
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp, # 0, # 0.6,
        "max_tokens": max_tokens,
        "include_reasoning": True,
        "top_p": 0.95,
        # "reasoning": { "effort": "high" },
        # "top_k": 1
    }
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=900.0) as client:
                response = await client.post(API_URL, headers=headers, json=data)
                response.raise_for_status()
                
                try:
                    resp_json = response.json()
                except json.decoder.JSONDecodeError:
                    logger.error(f"Failed to decode JSON. Response text: {response.text}")
                    continue  # Try again
                
                message = resp_json.get("choices", [{}])[0].get("message", {})
                model_ans = message.get("content", "")
                reasoning = message.get("reasoning", "")
                finish_reason = resp_json.get("choices", [{}])[0].get("finish_reason", "")
                usage = resp_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                return {
                    "response": model_ans,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "reasoning": reasoning,
                    "completion_tokens": completion_tokens
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            
        if attempt < max_retries - 1:  # Don't sleep on last attempt
            await asyncio.sleep(retry_delay)
            
    # If all retries failed, return empty dict instead of None
    return {
        "response": "",
        "finish_reason": "error",
        "prompt_tokens": 0,
        "reasoning": "",
        "completion_tokens": 0
    }

def load_existing_results(save_path):
    """
    Load existing results from file if it exists.
    Returns a dictionary mapping idx to result.
    """
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            existing_results = json.load(f)
            # Create a dictionary mapping idx to result
            return {result['idx']: result for result in existing_results}
    return {}

def get_save_path(model, split="test", data_type="curated"):
    """Get the save path for results file."""
    dir_idx = model.find("/")
    model_name = model[dir_idx+1:]
    file_name = f"{model_name}__{split}__{data_type}_results.json"
    return os.path.join(OUTPUT_DIR, file_name)

async def batch_eval_model(model, dataset, prompts, split, data_type, max_tokens=2048, batch_size=10, task_type="binary"):
    save_path = get_save_path(model, split, data_type)
    existing_results = load_existing_results(save_path)
    
    # Filter prompts that haven't been processed yet or were skipped
    pending_indices = []
    pending_prompts = []
    relevant_results = []
    for i, prompt in enumerate(prompts):
        idx = DELTA + i
        should_process = (
            idx not in existing_results or  # New entry
            existing_results[idx].get("skipped", False) or  # Was skipped
            not existing_results[idx].get("response", "").strip()  # Empty response
        )
        if should_process:
            pending_indices.append(i)
            pending_prompts.append(prompt)
            # Remove from existing results if present since it will be reprocessed
            if idx in existing_results:
                del existing_results[idx]

        else :
            if idx in existing_results:
                relevant_results.append(existing_results[idx])
    
    results = relevant_results # list(existing_results.values())  # Convert remaining existing results to list
    
    logger.info(f"Found {len(existing_results)} existing results. Processing {len(pending_prompts)} new or previously skipped prompts.")
    
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(pending_prompts), batch_size)):
            batch_prompts = pending_prompts[i:i+batch_size]
            batch_indices = pending_indices[i:i+batch_size]
            tasks = [async_query_openrouter(session, model, prompt, max_tokens) for prompt in batch_prompts]
            batch_results = await asyncio.gather(*tasks)
            
            for j, info in enumerate(batch_results):
                idx = DELTA + batch_indices[j]
                current_prompt = batch_prompts[j]
                context = {
                    "model": model,
                    "prompt": current_prompt,
                    "split": split,
                    "data_type": data_type,
                    "idx": idx
                }
                context.update(info)
                
                response = context["response"]
                
                if task_type == "binary":
                    final_answer = extract_final_answer0(response)
                    skipped = False 
                    
                    if final_answer is None :
                        final_answer = 0.5 # default to 0.5 if no valid number is found
                        skipped = True 
                        
                    context["final_answer"] = float(final_answer)
                    context["resolution"] = float(dataset["resolution"][context["idx"]])
                    context["skipped"] = skipped
                    
                elif task_type == "mcq":
                    # answer1 = extract_answer1(response)
                    # answer2 = extract_answer2(response)
                    if '<A>' in current_prompt and '<B>' in current_prompt and '<C>' in current_prompt and '<D>' in current_prompt:
                        answer1, answer2, values = detailed_extraction(response)
                        context["values"] = values
                    else:
                        answer1 = extract_answer1(response)
                        answer2 = extract_answer2(response)
                        
                    skipped = False
                    
                    if answer1 is None :
                        skipped = True
                    
                    if answer2 is None :
                        answer2 = 1 
                        
                    context["answer1"] = answer1
                    context["answer2"] = answer2
                    context["resolution"] = dataset["resolution"][context["idx"]]
                    context["skipped"] = skipped
                
                # if id column is present in dataset, add it to context
                if "id" in dataset.column_names:
                    context["id"] = dataset["id"][context["idx"]]
                # if post_id column is present in dataset, add it to context
                if "post_id" in dataset.column_names:
                    context["post_id"] = dataset["post_id"][context["idx"]]
        
                results.append(context)
                
                # Save results incrementally after each batch
                with open(save_path, 'w') as f:
                    json.dump(results, f)
            
            await asyncio.sleep(1)  # Avoid rate limits
    
    return results


def extend_model_outputs(dataset, results, task_type, verbose=False):
    predictions = []
    actuals = []
    skips = 0
    mean_response_token_length = []
    accuracy = 0
    
    for result in results:
        if task_type == "binary":
            predictions.append(result["final_answer"])
            actuals.append(result["resolution"])
                
        elif task_type == "mcq":
            if result["answer1"] is not None and result["answer2"] is not None:
                if result["answer1"] == result["resolution"]:
                    predictions.append(result["answer2"])
                    accuracy += 1
                else:
                    predictions.append(0)
                    
                actuals.append(1)
        
        mean_response_token_length.append(result["completion_tokens"])
        
        if result["skipped"]:
            skips += 1
            
        if verbose:
            logger.info(f"Model: {result['model']}\nResponse: {result['response']}\nActual Resolution: {result['resolution']}\n\n\n")
    
    predictions = np.array(predictions, dtype=float)
    actuals = np.array(actuals, dtype=float)
    mean_response_token_length = np.array(mean_response_token_length, dtype=int)
    
    # Brier Score = mean((p - y)^2)
    brier_score = np.mean((predictions - actuals) ** 2)

    # Accuracy = fraction of correct classifications at threshold 0.5
    if task_type == "binary":
        predicted_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_binary == actuals)
    elif task_type == "mcq":
        # print("MCQ accuracy: ", accuracy)
        accuracy = accuracy / len(predictions)
    
    # Also calculate log odds scoring rule (log(p) if y=1, log(1-p) if y=0)
    log_score = np.mean(np.where(actuals == 1, np.log(predictions), np.log(1 - predictions)))
    
    # Mean response token length
    mean_response_token_length = np.mean(mean_response_token_length)

    logger.info(f"Log Score:   {log_score:.4f}")
    logger.info(f"Brier Score: {brier_score:.4f}")
    logger.info(f"Accuracy:    {accuracy:.4f}")
    logger.info(f"Skipped:     {skips}")
    logger.info(f"Mean Response Token Length: {mean_response_token_length}")
    
    return results


if __name__ == "__main__":
    import argparse
    from data_utils import *
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/")
    parser.add_argument('--split', type=str, default="test", help="Which data split? ['test', 'train', 'validation']")
    parser.add_argument('--data_type', type=str, default="curated", help="Which data type? ['curated', 'raw']")
    parser.add_argument('--zero_shot', type=bool, default=True, help="Zero-shot forecasting?")
    parser.add_argument('--data', type=str, default="metaculus",
                      help="Which dataset to use")
    
    args = parser.parse_args()
    base_save_dir = args.base_save_dir
    split = args.split
    data_type = args.data_type
    zero_shot = args.zero_shot
    
    # Set OUTPUT_DIR based on data argument
    if args.data == "halawi":
        OUTPUT_DIR = f"/fast/nchandak/forecasting/evals/{args.data}/"
    else:
        OUTPUT_DIR = f"/fast/nchandak/forecasting/evals/custom/{args.data}/"
    
    if zero_shot:
        OUTPUT_DIR += "zeroshot/"
        
    # Make output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task_type = "binary" if "mcq" not in args.data else "mcq"
    
    # Load appropriate dataset based on data argument
    if args.data == "halawi":
        dataset = load_halawi_data(split=split, raw=(data_type=="raw"))
        prompts = create_prompts(dataset, zero_shot=zero_shot)
    
    elif args.data == "metaculus":
        dataset = load_metaculus_data(split=split)
        # only keep 5 rows of dataset
        # dataset = dataset.select(range(10))
        prompts = create_prompts(dataset, zero_shot=zero_shot)
        
    elif args.data == "manifold":
        dataset = load_manifold_data(split=split)
        # only keep 100 rows of dataset
        # dataset = dataset.select(range(20))
        
        prompts = create_prompts(dataset, zero_shot=zero_shot)
        # prompts = dataset["prompt"]
        
    elif args.data == "metaculus_mcq":
        dataset = load_mcq_metaculus_data(split=split)
        prompts = dataset["full_prompt"]
        
    elif args.data == "manifold_mcq":
        dataset = load_mcq_manifold_data(split=split, data_type=data_type, volume=1000)
        if split == "train":
            dataset = dataset.select(range(1000))
            
        # dataset = dataset.select(range(100))
        prompts = dataset["full_prompt"]
        
    elif "menge" in args.data:
        which_data_type = args.data.split("_")[1]
        dataset = load_menge_data(split=split, data_type=which_data_type)
        dataset = dataset.select(range(500))
        prompts = dataset["full_prompt"]
        
    elif "infinitegames" in args.data:
        dataset = load_infinitegames_data(split=split)
        # dataset = dataset.select(range(10))
        
        prompts = dataset["full_prompt"]
        
    elif "paleka" in args.data:
        dataset = load_paleka_data()
        # dataset = dataset.select(range(10))
        prompts = dataset["prompt"]
        
        
    logger.info(f"Actual dataset size: {len(dataset)}, Zero-shot: {zero_shot}")
    
    
    for model in MODEL_VARIANTS:
        new_tokens = 16384 if "deepseek" in model or "distill" in model else 2048 # 8192
        new_tokens = 16384 if "openai" in model else new_tokens
        new_tokens = 16384 if "gemini" in model else new_tokens
        new_tokens = 16384 if "claude" in model else new_tokens
        new_tokens = 16384 if "grok" in model else new_tokens
        
        logger.info(f"Querying {model} with token limit {new_tokens}...")
        results = asyncio.run(batch_eval_model(model, dataset, prompts, split, data_type, max_tokens=new_tokens, task_type=task_type, batch_size=200))
        
        results = extend_model_outputs(dataset, results, task_type=task_type, verbose=False)
        
        time.sleep(5)
        
    logger.info("Done!")
