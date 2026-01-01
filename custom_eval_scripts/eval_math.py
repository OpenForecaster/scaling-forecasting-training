"""
Evaluation script for mathematical reasoning benchmarks (MATH dataset).
Evaluates models on mathematical problem-solving tasks from datasets like
DigitalLearningGmbH/MATH-lighteval and HuggingFaceH4/MATH-500.
Uses math_verify library to check answer correctness.
Uses vLLM for efficient inference.
"""

import re
import torch
import numpy as np
from datasets import Dataset, load_dataset
from typing import Optional
from transformers import AutoTokenizer
from tqdm import tqdm 
import json
import os 
import logging
import time 
from dataclasses import dataclass
from math_verify import parse, verify

# Import vLLM for faster generation
from vllm import LLM, SamplingParams

# Set SEED
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: str = "main"
    torch_dtype: Optional[str] = "bfloat16"
    trust_remote_code: bool = True

@dataclass
class EvalScriptArguments:
    dataset_id_or_path: str = "DigitalLearningGmbH/MATH-lighteval"
    dataset_splits: str = "train"
    tokenizer_name_or_path: Optional[str] = None
    model_checkpoint: str = None
    per_device_eval_batch_size: int = 32
    output_dir: str = "results/"

def add_idx_column(dataset: Dataset) -> Dataset:
    """Adds an 'idx' column to the dataset."""
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)

def extract_answer(completion: str) -> Optional[str]:
    """Extracts the final answer from the LLM's output."""
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()

def load_model_and_tokenizer(model_path: str, model_name: str = None):
    if model_name is None:
        model_name = model_path.rstrip("/").split("/")[-1]
    logger.info(f"Using model_name: {model_name}")

    logger.info(f"Loading model with vLLM from local directory: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.85,
            tensor_parallel_size=torch.cuda.device_count(),
        )
    except:
        model_path += "/snapshots/model/"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.85,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        
    return model, tokenizer

def format_math_prompt(problem: str) -> str:
    """Format the prompt for math questions."""
    return f"""Question (math): {problem}
Find the solution to the above question. Show your work (reasoning) in <think> </think> tags. And return only the final answer in <answer> </answer> tags, for example <answer> \\frac{{15}}{{19}} </answer>. Think step by step inside <think> tags."""

def evaluate_model(
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset,
    which_dataset: str,
    which_split: str,
    output_dir: str,
    max_new_tokens: int = 4096,
    num_generations: int = 1
):
    """Run batched inference using vLLM"""
    all_prompts = []
    all_idxs = []
    all_answers = []
    
    for row in dataset:
        try:
            chat = [{
                "role": "system",
                "content": "You are a helpful assistant. For any query asked by the user, you first think about the reasoning process in the mind and then provides an answer."
            },
            {
                "role": "user",
                "content": format_math_prompt(row["problem"])
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }]
            
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
        except Exception as e:
            logger.info(f"Error in tokenizer.apply_chat_template: {e}")
            prompt = format_math_prompt(row["problem"])
        
        all_prompts.append(prompt)
        all_idxs.append(row["idx"])
        all_answers.append(row["answer"])
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=num_generations,
    )
    logger.info(f"Starting generation with vLLM for {len(all_prompts)} prompts")
    start_time = time.time()
    
    all_outputs = model.generate(all_prompts, sampling_params)
    
    end_time = time.time()
    logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    accuracies = []
    all_results = []
    
    for gen_idx in range(num_generations):
        correct = 0
        wrong = 0
        total = len(all_outputs)
        
        for i, outputs in enumerate(all_outputs):
            prompt = all_prompts[i]
            idx = all_idxs[i]
            expected_answer = all_answers[i]
            
            generated_text = outputs.outputs[gen_idx].text
            
            # Find where the prompt ends and the completion begins
            prompt_end_idx = generated_text.find("Let me solve this step by step.\n<think>")
            if prompt_end_idx == -1:
                completion = generated_text
            else:
                completion = generated_text[prompt_end_idx:]
            
            # Extract the answer
            answer = extract_answer(completion)
            
            if answer is None:
                wrong += 1
                result_type = "no answer"
            else:
                try:
                    # Parse and verify the answer
                    answer_text = "$" + str(answer) + "$"
                    final_answer = parse(answer_text)
                    
                    # Handle boxed answers in the expected answer
                    if "\\boxed" in expected_answer:
                        boxed_content = ""
                        brace_count = 0
                        start_idx = expected_answer.rfind("\\boxed{") + 7
                        
                        for i in range(start_idx, len(expected_answer)):
                            if expected_answer[i] == '{':
                                brace_count += 1
                            elif expected_answer[i] == '}':
                                if brace_count == 0:
                                    break
                                brace_count -= 1
                            boxed_content += expected_answer[i]
                        
                        gold = parse("$" + boxed_content + "$")
                    else:
                        gold = parse("$" + str(expected_answer) + "$")
                    
                    if verify(gold, final_answer, precision=3):
                        correct += 1
                        result_type = "correct"
                    else:
                        wrong += 1
                        result_type = "wrong"
                except Exception as e:
                    logger.warning(f"Error verifying answer: {str(e)}")
                    wrong += 1
                    result_type = "wrong"
                
            result = {
                "model": model_name,
                "split": which_split,
                "data_type": which_dataset,
                "idx": idx,
                "generation_idx": gen_idx,
                "response": completion,
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "completion_tokens": len(tokenizer.encode(completion)),
                "final_answer": answer,
                "expected_answer": expected_answer,
                "skipped": answer is None,
                "correct": 1 if result_type == "correct" else 0,
            }
            
            all_results.append(result)
        
        # Calculate metrics for this generation
        accuracy = (correct / total) * 100
        accuracies.append(accuracy)
    
    # Calculate mean and std dev of accuracies
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    logger.info(f"Results across {num_generations} generations:")
    logger.info(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    
    # Save results if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}{model_name}_{which_split}_size_{len(dataset)}_generations_{num_generations}.json"
            
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved {len(all_results)} generations to {output_file}")
        
    # Save summary metrics
    metrics_file = os.path.join("metrics.csv")
    with open(metrics_file, "a") as f:
        if not os.path.getsize(metrics_file):
            f.write("dataset,split,model,dataset_size,mean_accuracy,std_accuracy,max_new_tokens,num_generations\n")
        f.write(f"{which_dataset},{which_split},{model_name},{len(dataset)},{mean_accuracy:.2f},{std_accuracy:.2f},{max_new_tokens},{num_generations}\n")
    
    logger.info(f"Updated metrics in {metrics_file}")
    
    return mean_accuracy, std_accuracy

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/manual/math/", help="Where to save outputs of the model")
    parser.add_argument('--model_dir', type=str, default="/fast/rolmedo/models/qwen2.5-7b-it", help="Model directory")
    parser.add_argument('--model', type=str, default=None, help="Model name")
    parser.add_argument('--max_new_tokens', type=int, default=16384, help="Maximum number of new tokens")
    parser.add_argument('--data_split', type=str, default="test", help="Dataset split to use")
    parser.add_argument('--num_generations', type=int, default=1, help="Number of generations per prompt")
    parser.add_argument('--data', type=str, default="DigitalLearningGmbH/MATH-lighteval", help="Dataset to evaluate on")
    
    args = parser.parse_args()
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {gpu_count}")
    
    # Load dataset
    dataset = load_dataset(args.data)[args.data_split]
    dataset = add_idx_column(dataset)
    
    # Standardize column names
    if "question" in dataset.column_names:
        dataset = dataset.rename_column("question", "problem")
    if "Question" in dataset.column_names:
        dataset = dataset.rename_column("Question", "problem")
    if "Answer" in dataset.column_names:
        dataset = dataset.rename_column("Answer", "answer")
    if "solution" in dataset.column_names and "answer" not in dataset.column_names:
        dataset = dataset.rename_column("solution", "answer")
    if "Solution" in dataset.column_names and "answer" not in dataset.column_names:
        dataset = dataset.rename_column("Solution", "answer")
    if "Problem" in dataset.column_names:
        dataset = dataset.rename_column("Problem", "problem")
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # only keep 100 rows
    # dataset = dataset.select(range(20))
    
    # Load model and tokenizer
    if args.model == None:
        model_name = args.model_dir.rstrip("/").split("/")[-1]
        if "checkpoint" in args.model_dir:
            model_name = args.model_dir.rstrip("/").split("/")[-2] + "__" + args.model_dir.rstrip("/").split("/")[-1]
    else:
        model_name = args.model
    
    
    # logger.info(f"Model name: {model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_dir, model_name)
    
    # Run evaluation
    evaluate_model(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        which_dataset=args.data,
        which_split=args.data_split,
        dataset=dataset,
        output_dir=args.base_save_dir,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations
    )

if __name__ == "__main__":
    main() 