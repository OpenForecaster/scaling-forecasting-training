import re
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from typing import Optional, List
from tqdm import tqdm 
import os 
import logging
import time 

import requests
import json
import time

# Your OpenRouter API Key
# from forecasting.inference.open_router import API_KEY
from inference.open_router_key import API_KEY

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL_DIR = ""

OUTPUT_DIR = "/fast/nchandak/forecasting/evals/halawi/"

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
    "deepseek/deepseek-r1",
    
    # "meta-llama/llama-2-70b",
    # "togethercomputer/llama-2-70b-chat",
]


# OpenRouter API URL
API_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        logger.warn(f"Error with regex: {e}")
        return 0.5
    
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
        Resolution Criteria: {resolution_criteria}
        Question close date: {date_close}
        
        Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end
        of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe 
        the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
        """
    
    return f"""Question: {question}
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



# Function to query OpenRouter
def query_openrouter(model, prompt, max_tokens=1024):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "include_reasoning": True,
        "top_p": 1,
        "top_k": 1  # Forces most probable token selection
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        try:
            resp_json = response.json()
        except json.decoder.JSONDecodeError:
            logger.error(f"Failed to decode JSON. Response text: {response.text}")
            return {}
        
        # logger.info(resp_json)
        message = resp_json.get("choices", [{}])[0].get("message", {})
        model_ans = message.get("content", "")
        
        reasoning = message.get("reasoning", "")
        finish_reason = resp_json.get("choices", [{}])[0].get("finish_reason", "")
        
        usage = resp_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        info = {"response": model_ans, "finish_reason": finish_reason, "prompt_tokens": prompt_tokens, "reasoning": reasoning, "completion_tokens": completion_tokens}
        return info
    else:
        logger.warn(f"Error with {model}: {response.text}")
        return {}


def eval_model(model, prompts, split, data_type, max_tokens=2048):
    results = []
    # Run inference on each model for each prompt
    for i, prompt in tqdm(enumerate(prompts)):
        # logger.info(f"Querying {model} with prompt: {prompt}")
        info = query_openrouter(model, prompt, max_tokens=max_tokens)
        if len(info) > 0:
            context = {"model": model, "prompt": prompt, "split": split, "data_type": data_type,  "idx": i}
            for key, value in info.items():
                context[key] = value
                
            results.append(context)
        
        # print(f"Model: {model}\nPrompt: {prompt}\nResponse: {response}\n\n")
        
        time.sleep(1)  # Avoid hitting rate limits
        
    return results 

def save_results(model, results, split="test", data_type="curated"):
    dir_idx = model.find("/")
    model_name = model[dir_idx+1:]
    
    file_name = f"{model_name}__{split}__{data_type}_results3.json"
    
    # TODO: Add timestamp to file name
    
    save_dir = os.path.join(OUTPUT_DIR, file_name)
    save_path = save_dir 
    
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"results.json")
    logger.info(f"Saving results to {save_path}")
    
    with open(save_path, "w") as f:
        json.dump(results, f)


def extend_model_outputs(dataset, results, verbose=False):
    predictions = []
    actuals = []
    
    for result in results:
        response = result["response"]
        final_answer = extract_final_answer0(response)
        skipped = False 
        
        if final_answer is None :
            final_answer = 0.5 # default to 0.5 if no valid number is found
            skipped = True 
            
        result["final_answer"] = float(final_answer)
        result["resolution"] = float(dataset["resolution"][result["idx"]])
        result["skipped"] = skipped
        
        predictions.append(final_answer)
        actuals.append(result["resolution"])
        
        if verbose:
            logger.info(f"Model: {result['model']}\nResponse: {result['response']}\nFinal Answer: {final_answer}\nActual Resolution: {result['resolution']}\n\n\n")
    
    predictions = np.array(predictions, dtype=float)
    actuals = np.array(actuals, dtype=float)
    
    # Brier Score = mean((p - y)^2)
    brier_score = np.mean((predictions - actuals) ** 2)

    # Accuracy = fraction of correct classifications at threshold 0.5
    predicted_binary = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_binary == actuals)

    logger.info(f"Brier Score: {brier_score:.4f}")
    logger.info(f"Accuracy:    {accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    from data_utils import load_halawi_data, filter_halawi_data
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/")
    
    parser.add_argument('--split', type=str, default="test", help="Which data split? ['test', 'train', 'validation']")
    parser.add_argument('--data_type', type=str, default="curated", help="Which data type? ['curated', 'raw']")
    parser.add_argument('--zero_shot', type=bool, default=False, help="Zero-shot forecasting?")

    args = parser.parse_args()
    
    base_save_dir = args.base_save_dir
    split = args.split
    data_type = args.data_type
    zero_shot = args.zero_shot
    
    if zero_shot:
        OUTPUT_DIR += "zeroshot/"
        
    # load training data
    train_dataset = load_halawi_data(split=split, raw=(data_type=="raw"))

    # Filter or pick only part of the dataset if desired
    dataset = train_dataset
    dataset = train_dataset.select(range(500))
    
    logger.info(f"Actual dataset size: {len(train_dataset)}, Zero-shot: {zero_shot}")
    
    prompts = create_prompts(dataset, zero_shot=zero_shot)
    for model in MODEL_VARIANTS:
        new_tokens = 2048 
        if "deepseek-r1" in model and "distill" not in model : # Actual DeepSeek-R1 model 
            new_tokens = 8192
            
        logger.info(f"Querying {model}...")
        results = eval_model(model, prompts, split=split, data_type=data_type, max_tokens=new_tokens)
        results = extend_model_outputs(dataset, results, verbose=False)
        save_results(model, results, split=split, data_type=data_type)
        # logger.info(f"Saved results for {model} to {OUTPUT_DIR}")
        time.sleep(5)
        
    logger.info("Done!")

    