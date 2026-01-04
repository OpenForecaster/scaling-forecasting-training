#!/usr/bin/env python3
"""
Script to convert FSDP checkpoints to HuggingFace format and launch evaluation jobs.

This script:
1. Recursively finds all checkpoints in the input directory
2. Converts FSDP checkpoints to HuggingFace format if not already done
3. Launches evaluation jobs for all converted checkpoints

Usage:
    python convert_checkpoints_fixed.py --input_dir /path/to/checkpoints --base_model_path /path/to/base/model
"""

import os
import re
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_checkpoints(input_dir: str) -> List[Dict[str, str]]:
    """
    Recursively find all checkpoints in the input directory.
    
    Args:
        input_dir: Root directory to search for checkpoints
        
    Returns:
        List of dictionaries containing checkpoint information
    """
    checkpoints = []
    
    # Find all global_step_* directories
    pattern = os.path.join(input_dir, "**/global_step_*")
    global_step_dirs = glob.glob(pattern, recursive=True)
    
    logger.info(f"Found {len(global_step_dirs)} global_step directories")
    
    for global_step_dir in global_step_dirs:
        # Extract information from path
        path_parts = Path(global_step_dir).parts
        
        # Find the project name (usually the parent of global_step_*)
        project_dir = Path(global_step_dir).parent
        project_name = project_dir.parent.name
        
        # Extract step number
        step_match = re.search(r'global_step_(\d+)', global_step_dir)
        if not step_match:
            logger.warning(f"Could not extract step number from {global_step_dir}")
            continue
        step_num = step_match.group(1)
        
        if project_name == "rl-test":
            project_name = "data20k"
        elif project_name == "rl-freeform":
            project_name = "datamix70k"
        elif project_name == "rl-freeform-data22k":
            project_name = "data22k"
        
        model_name = project_dir.name
        
        print("Project name: ", project_name)
        print("Model name: ", model_name)
        print("Step number: ", step_num)
        print("Project dir: ", project_dir)
        
        # Check if actor directory exists
        actor_dir = os.path.join(global_step_dir, "actor")
        if not os.path.exists(actor_dir):
            logger.warning(f"Actor directory not found in {global_step_dir}")
            continue
        
        final_model_name = f"{model_name}-{project_name}-checkpoint{step_num}"
        
        # Check if already converted
        expected_hf_dir = os.path.join(global_step_dir, f"{model_name}-{project_name}-checkpoint{step_num}")
        is_converted = os.path.exists(expected_hf_dir) and os.path.exists(os.path.join(expected_hf_dir, "config.json"))
        
        checkpoint_info = {
            'global_step_dir': global_step_dir,
            'actor_dir': actor_dir,
            'project_name': project_name,
            'model_name': model_name,
            'step_num': step_num,
            'expected_hf_dir': expected_hf_dir,
            'is_converted': is_converted,
            'final_model_name': final_model_name
        }
        
        checkpoints.append(checkpoint_info)
        logger.info(f"Found checkpoint: {final_model_name} (converted: {is_converted})")
    
    return checkpoints

def convert_checkpoint_to_hf(checkpoint_info: Dict[str, str]) -> bool:
    """
    Convert a single FSDP checkpoint to HuggingFace format.
    
    Args:
        checkpoint_info: Dictionary containing checkpoint information
        
    Returns:
        True if conversion successful, False otherwise
    """
    fsdp_checkpoint_path = checkpoint_info['actor_dir']
    huggingface_model_path = os.path.join(fsdp_checkpoint_path, "huggingface")
    output_path = checkpoint_info['expected_hf_dir']
    
    # Check if huggingface directory exists in actor
    if not os.path.exists(huggingface_model_path):
        logger.warning(f"HuggingFace directory not found in {fsdp_checkpoint_path}, using base model path")
        return False
    
    logger.info(f"Converting checkpoint: {checkpoint_info['final_model_name']}")
    logger.info(f"FSDP path: {fsdp_checkpoint_path}")
    logger.info(f"HF model path: {huggingface_model_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        # Run the conversion script with the fixed version
        cmd = [
            "python", "libraries/verl/scripts/ours/convert_fsdp_to_hf_fixed.py",
            "--fsdp_checkpoint_path", fsdp_checkpoint_path,
            "--huggingface_model_path", huggingface_model_path,
            "--output_path", output_path
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/nchandak/forecasting")
        
        if result.returncode == 0:
            logger.info(f"Successfully converted {checkpoint_info['final_model_name']}")
            return True
        else:
            logger.error(f"Failed to convert {checkpoint_info['final_model_name']}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Exception during conversion: {e}")
        return False

def launch_simpleqa_evaluation_job(checkpoint_info: Dict[str, str], n_gpus: int = 1) -> bool:
    """
    Launch an evaluation job for a converted checkpoint.
    
    Args:
        checkpoint_info: Dictionary containing checkpoint information
        n_gpus: Number of GPUs to request for evaluation
        
    Returns:
        True if job launched successfully, False otherwise
    """
    model_dir = checkpoint_info['expected_hf_dir']
    model_name = checkpoint_info['model_name']
    
    final_model_name = checkpoint_info['final_model_name']
    
    logger.info(f"Launching evaluation job for {final_model_name}")
    logger.info(f"Model directory: {model_dir}")
    
    try:
        # Run the evaluation job launcher
        cmd = [
            "python", "jobs_eval.py",
            "--model_dir", model_dir,
            "--model", final_model_name,
            "--task", "simpleqa",
            "--n_gpus", str(n_gpus),
            "--max_new_tokens", "16384",
            "--num_generations", "4"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/nchandak/forecasting/custom_eval_scripts/")
        
        if result.returncode == 0:
            logger.info(f"Successfully launched evaluation job for {final_model_name}")
            return True
        else:
            logger.error(f"Failed to launch evaluation job for {final_model_name}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Exception during job launch: {e}")
        return False

def launch_guardian_evaluation_job(checkpoint_info: Dict[str, str], n_gpus: int = 1) -> bool:
    """
    Launch an evaluation job for a converted checkpoint.
    
    Args:
        checkpoint_info: Dictionary containing checkpoint information
        n_gpus: Number of GPUs to request for evaluation
        
    Returns:
        True if job launched successfully, False otherwise
    """
    model_dir = checkpoint_info['expected_hf_dir']
    model_name = checkpoint_info['model_name']
    
    final_model_name = checkpoint_info['final_model_name']
    
    logger.info(f"Launching evaluation job for {final_model_name}")
    logger.info(f"Model directory: {model_dir}")
    
    try:
        # Run the evaluation job launcher
        cmd = [
            "python", "jobs_eval.py",
            "--model_dir", model_dir,
            "--model", final_model_name,
            "--task", "freeform",
            "--n_gpus", str(n_gpus),
            "--max_new_tokens", "32768",
            "--data", "/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian_207_free_3_cleaned.jsonl",
            "--data_split", "eval",
            "--num_generations", "8",
            "--base_save_dir", "/fast/nchandak/forecasting/evals/freeform/manual/"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/nchandak/forecasting/custom_eval_scripts/")
        
        if result.returncode == 0:
            logger.info(f"Successfully launched evaluation job for {final_model_name}")
            return True
        else:
            logger.error(f"Failed to launch evaluation job for {final_model_name}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Exception during job launch: {e}")
        return False

def launch_validation2_evaluation_job(checkpoint_info: Dict[str, str], n_gpus: int = 1) -> bool:
    """
    Launch an evaluation job for a converted checkpoint.
    
    Args:
        checkpoint_info: Dictionary containing checkpoint information
        n_gpus: Number of GPUs to request for evaluation
        
    Returns:
        True if job launched successfully, False otherwise
    """
    model_dir = checkpoint_info['expected_hf_dir']
    model_name = checkpoint_info['model_name']
    final_model_name = checkpoint_info['final_model_name']
    
    logger.info(f"Launching evaluation job for {checkpoint_info['final_model_name']}")
    logger.info(f"Model directory: {model_dir}")
    
    try:
        # Run the evaluation job launcher
        cmd = [
            "python", "jobs_eval.py",
            "--model_dir", model_dir,
            "--model", final_model_name,
            "--task", "freeform",
            "--n_gpus", str(n_gpus),
            "--max_new_tokens", "32768",
            "--data", "/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/combined_non_numeric_all_validation.jsonl",
            "--data_split", "eval",
            "--num_generations", "8",
            "--base_save_dir", "/fast/nchandak/forecasting/evals/freeform/manual/"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/nchandak/forecasting/custom_eval_scripts/")
        
        if result.returncode == 0:
            logger.info(f"Successfully launched evaluation job for {final_model_name}")
            return True
        else:
            logger.error(f"Failed to launch evaluation job for {final_model_name}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Exception during job launch: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoints to HF format and launch evaluation jobs")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing checkpoints")
    parser.add_argument("--n_gpus", type=int, default=2,
                       help="Number of GPUs to request for evaluation jobs")
    parser.add_argument("--skip_conversion", action="store_true",
                       help="Skip checkpoint conversion, only launch evaluation jobs")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation job launch, only convert checkpoints")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    logger.info(f"Processing checkpoints in: {args.input_dir}")
    logger.info(f"Number of GPUs for evaluation: {args.n_gpus}")
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.input_dir)
    
    if not checkpoints:
        logger.warning("No checkpoints found")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    # Convert checkpoints if not skipping
    if not args.skip_conversion:
        logger.info("Starting checkpoint conversion...")
        conversion_success = 0
        conversion_failed = 0
        
        for checkpoint_info in checkpoints:
            if checkpoint_info['is_converted']:
                logger.info(f"Skipping already converted checkpoint: {checkpoint_info['model_name']}-checkpoint{checkpoint_info['step_num']}")
                conversion_success += 1
                continue
            
            if convert_checkpoint_to_hf(checkpoint_info):
                conversion_success += 1
                checkpoint_info['is_converted'] = True
            else:
                conversion_failed += 1
        
        logger.info(f"Conversion complete: {conversion_success} successful, {conversion_failed} failed")
    
    # Launch evaluation jobs if not skipping
    if not args.skip_evaluation:
        logger.info("Starting evaluation job launches...")
        evaluation_success = 0
        evaluation_failed = 0
        
        for checkpoint_info in checkpoints:
            if not checkpoint_info['is_converted']:
                logger.warning(f"Skipping unconverted checkpoint: {checkpoint_info['model_name']}-checkpoint{checkpoint_info['step_num']}")
                evaluation_failed += 1
                continue
            
            # Uncomment the evaluation jobs you want to run
            # if launch_simpleqa_evaluation_job(checkpoint_info, args.n_gpus):
            #     evaluation_success += 1
            # else:
            #     evaluation_failed += 1
                
            # if launch_guardian_evaluation_job(checkpoint_info, args.n_gpus):
            #     evaluation_success += 1
            # else:
            #     evaluation_failed += 1
                
            # if launch_validation2_evaluation_job(checkpoint_info, args.n_gpus):
            #     evaluation_success += 1
            # else:
            #     evaluation_failed += 1
        
        logger.info(f"Evaluation job launches complete: {evaluation_success} successful, {evaluation_failed} failed")
    
    logger.info("Script execution complete")

if __name__ == "__main__":
    main()
