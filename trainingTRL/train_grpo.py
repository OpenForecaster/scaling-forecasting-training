import os
os.environ["SOFT_FILELOCK"] = "1"
from filelock import SoftFileLock
import filelock
# Override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock
import logging
from dataclasses import dataclass
from datetime import datetime
import logging
import os
import configparser
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# # Parse the config file to get the token
# HF_TOKEN_PATH = "/fast/sgoel/hfcache/huggingface/stored_tokens"
# config = configparser.ConfigParser()
# config.read(HF_TOKEN_PATH)

# # Use the token from mpi_cluster3 section (you can change this if needed)
# token = config['mpi_cluster3']['hf_token']

# os.environ["HF_HOME"] = "/fast/sgoel/hfcache/huggingface"
# os.environ["HUGGING_FACE_HUB_TOKEN"] = token

import random
import re
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
from accelerate import Accelerator
import wandb
import numpy as np
from data_utils import *

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "YuehHanChen/forecasting"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# Initialize Accelerator
accelerator = Accelerator()


########################
# Helper functions
########################

def format_reward_func(completions, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    # Pre-compile regex pattern for better performance
    pattern = re.compile(r"^\s*<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>[\s\S]*?<answer>([\s\S]*?)<\/answer>\s*$")
    
    rewards = []
    for completion in completions:
        try:
            # Add synthetic <think> tag
            completion = "<think>" + completion
            
            # Use pre-compiled pattern to check format
            match = pattern.search(completion)
            
            # Simplified reward assignment
            rewards.append(1.0 if match and len(match.groups()) == 2 else -1.0)
            
        except Exception:
            rewards.append(-1.0)
            
    return rewards

def log_odds_scoring_rule(completions, question, resolution, **kwargs):
    """
    Evaluates completions based on:
    1. If the answer is "I don't know" (inside <answer> tags), reward 0.05.
    2. Otherwise, check for a mathematically correct equation that uses all provided numbers exactly once.
    
    Args:
        completions (list[str]): Generated outputs.
        expecte_answer (list[str]): Expected answers.
    
    Returns:
        list[float]: Reward scores.
    """
    rewards = []
    bce = torch.nn.BCELoss()
    for completion, query, gt in zip(completions, question, resolution):
        try:
            # Prepend synthetic <think> to align with the expected prompt structure.
            completion = "<think>" + completion
            # Extract the content within <answer> tags.
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(np.log(0.5))
                continue
            answer_text = match.group(1).strip()
            
            # Log (on console, like print) completion, problem, gt with 5% probability
            # if random.random() < 0.0001:
            #     logger.info(f"Completion: {completion}")
            #     logger.info(f"Problem: {query}")
            #     logger.info(f"GT: {gt} <-> Extracted: {answer_text}")
            #     logger.info("---------------------------------------\n")
            
            prediction = float(answer_text)
            if prediction < 0 or prediction > 1:
                prediction = 0.5 # Assume 0.5 if the prediction is out of bounds.
                # logger.info(f"Prediction out of bounds. Setting to 0.5")
                # logger.info("---------------------------------------")
                # logger.info(f"Completion: {completion}")
                # logger.info(f"Problem: {query}")
                # logger.info(f"GT: {gt} <-> Extracted: {answer_text}")
                # logger.info("---------------------------------------\n")
                
            y_pred = [prediction]
        except Exception:
            # In case of any errors during processing, assume prediction of 0.5 
            y_pred = [0.5]
            
        # Calculate binary cross entropy loss
        y_true = [float(gt)]
        try :
            bce_loss = bce(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32))
            rewards.append(-bce_loss.item())
        except Exception as e:
            # print the exception
            logger.info(f"Exception: {e}")
            logger.info(f"{y_pred}, {y_true}")
            logger.info(f"Completion: {completion}")
            logger.info(f"Answer: {answer_text}")
            
            rewards.append(np.log(0.5))
        
    return rewards


def brier_scoring_rule(completions, question, resolution, **kwargs):
    """
    Evaluates completions based on:
    1. If the answer is "I don't know" (inside <answer> tags), reward 0.05.
    2. Otherwise, check for a mathematically correct equation that uses all provided numbers exactly once.
    
    Args:
        completions (list[str]): Generated outputs.
        expecte_answer (list[str]): Expected answers.
    
    Returns:
        list[float]: Reward scores.
    """
    rewards = []
    pattern = re.compile(r"<answer>(.*?)<\/answer>", re.DOTALL)
    
    for completion, query, gt in zip(completions, question, resolution):
        try:
            # Find all answer tags and take the last one
            matches = pattern.findall("<think>" + completion)
            if not matches:
                reward = -2 # -0.25
                rewards.append(reward)
                continue
                
            # Get the last answer
            answer_text = matches[-1].strip()
                
            # Parse prediction, defaulting to 0.5 if invalid
            try:
                prediction = float(answer_text)
                if not 0 <= prediction <= 1:
                    prediction = 0.5
            except:
                prediction = 0.5
            # Calculate MSE loss
            y_true = float(gt)
            mse_loss = (prediction - y_true) ** 2
            rewards.append(-mse_loss)
            
            # Log interesting examples
            # close1 = (y_true > 0.99 and prediction < 0.001)
            # close0 = (y_true < 0.001 and prediction > 0.99)
            
            # if random.random() < 0.001 or close1 or close0:
            #     with open("raw_outputs/samples_brier2.txt", "a") as f:
            #         f.write(f"Prompt: {query}\n")
            #         f.write(f"Completion: {completion}\n") 
            #         f.write(f"y_pred: {prediction}\n")
            #         f.write(f"y_true: {y_true}\n")
            #         f.write("--------------------------------\n\n")
                    
        except Exception as e:
            logger.info(f"Exception: {e}")
            logger.info(f"Completion: {completion}")
            reward = -2 # -0.25
            rewards.append(reward)  # Baseline accuracy
            
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def format_forecasting_prompt(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str,
    zero_shot: bool = True,
) -> str:
    """
    Format the prompt given the row data.
    """
    
    if zero_shot:
        return f"""
Question: {question}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}
"""
    

def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face 
    
    
    # ---------------------------------------------
    
    # dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    
    # dataset_path = "YuehHanChen/forecasting"
    # test_dataset = load_dataset(dataset_path)["test"]
    
    # suffix = "curated"
    
    # if "raw" in script_args.dataset_id_or_path:
    #     suffix = "raw"
    #     dataset = load_halawi_data(split="train", raw=True)
    
    # ---------------------------------------------
    
    dataset = load_manifold_and_metaculus_data(split="train", raw=True)
    test_dataset = load_metaculus_data(split="test")
    
    # Print column names 
    # print(dataset.column_names)
    
    dataset = dataset.shuffle(seed=42) 
    # logger info dataset length
    logger.info(f"Dataset length: {len(dataset)}")

    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(row, zero_shot=True):
        if 'prompt' in row:
            local_prompt = row['prompt']
        else:
            local_prompt = format_forecasting_prompt(
                question=row["question"],
                background=row["background"],
                resolution_criteria=row["resolution_criteria"],
                date_begin=row["date_begin"],
                date_close=row["date_close"],
                zero_shot=zero_shot)
        
        r1_prefix = [{ 
            "role": "user",
            "content": f"You will be asked a forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. Show your work (reasoning) in <think> </think> tags. And return only the final answer (probability) in <answer> </answer> tags, for example if you think the event asked is 83% likely, then output <answer>0.83</answer>. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. Think step by step inside <think> tags."
          },
          {
            "role": "user",
            "content": local_prompt,
          },
          {
            "role": "assistant",
            "content": "Let me reason about this step by step.\n<think>"
          }]
        
        return_dict = row 
        return_dict["prompt"] = tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True)
        return return_dict

    # convert our dataset to the r1 prompt
    train_dataset = dataset.map(lambda x: generate_r1_prompt(x))
    test_dataset = test_dataset.map(lambda x: generate_r1_prompt(x))
    
    # Only initialize wandb on the main process.
    if accelerator.is_main_process:
        config_dict = training_args.to_dict()
        # print("init:", config_dict)  # This shows the config in your console.
        run_name = model_args.model_name_or_path + "-" + suffix + "-BRIER-2"
        wandb.init(project="forecasting-rl", name=run_name, config=config_dict)
        wandb.config.update(config_dict)
        

    #########################
    # Instantiate GRPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[format_reward_func, brier_scoring_rule], # log_odds_scoring_rule], # log_odds_scoring_rule],
      args=training_args,
      train_dataset=train_dataset,
      processing_class=tokenizer,
      eval_dataset=test_dataset, # Currently evals run sequentially so your whole training will be stopped while model generates response for eval set
    #   peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    # last_checkpoint = None
    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train() #
    # train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()