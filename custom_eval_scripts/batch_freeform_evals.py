#!/usr/bin/env python3
"""
Batch evaluation script for running evaluations across multiple model checkpoints.
Finds all checkpoint directories (global_step_*) in a training run and evaluates each one.
Supports multiple evaluation modes: freeform, binary, retrieval, retrievalbinary.
Useful for tracking model performance across training steps.
"""

import os
import sys
import subprocess
import argparse
import time
import glob
import re
from pathlib import Path
from typing import List, Tuple

def find_checkpoints(input_dir: str) -> List[str]:
    """Find all checkpoint directories in the input directory."""
    checkpoint_pattern = os.path.join(input_dir, "global_step_*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    # Sort checkpoints by step number
    def extract_step_number(path):
        match = re.search(r'global_step_(\d+)', path)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=extract_step_number)
    return checkpoints

def find_model_directory(checkpoint_dir: str) -> str:
    """Find the model directory inside a checkpoint directory."""
    model_pattern = os.path.join(checkpoint_dir, "*checkpoint*")
    model_dirs = glob.glob(model_pattern)
    
    if not model_dirs:
        raise ValueError(f"No model directory found in {checkpoint_dir}")
    
    # Return the first model directory found
    return model_dirs[0]

def extract_step_number(checkpoint_path: str) -> int:
    """Extract step number from checkpoint path."""
    match = re.search(r'global_step_(\d+)', checkpoint_path)
    if not match:
        raise ValueError(f"Could not extract step number from {checkpoint_path}")
    return int(match.group(1))

def run_eval_freeform(model_dir: str, output_file: str, log_file: str, mode: str, questions_file: str) -> Tuple[bool, int, str]:
    """Run eval_freeform.py on a model directory."""
    cmd = [
        sys.executable, f"eval_{mode}.py",
        "--model_dir", model_dir,
        "--questions_file", questions_file,
        # "--output_file", output_file
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Run the command and capture output
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Write output to log file
            # log_f.write(result.stdout)
            
            # Also print to console
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return False, -1, f"Command timed out after {duration:.1f}s"
    except Exception as e:
        duration = time.time() - start_time
        return False, -1, f"Exception occurred after {duration:.1f}s: {str(e)}"
    
    duration = time.time() - start_time
    return result.returncode == 0, result.returncode, f"{duration:.1f}s"

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation script for freeform forecasting models")
    parser.add_argument(
        "--input_dir",
        default="/fast/nchandak/forecasting/training/verl/checkpoints/rl-data66k-withbinary2k/Qwen3-8B-2048-8192",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--mode",
        default="freeform",
        help="which script to use for evaluation"
    )
    parser.add_argument(
        "--questions_file",
        default="test",
        help="questions file"
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--log_dir",
        default="./eval_logs",
        help="Directory to save evaluation logs"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip checkpoints that already have results"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Do not skip existing results (overrides --skip_existing)"
    )
    
    args = parser.parse_args()
    
    # Handle skip_existing logic
    skip_existing = args.skip_existing and not args.no_skip_existing
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("Starting batch evaluation...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Log directory: {args.log_dir}")
    print(f"Skip existing: {skip_existing}")
    print()
    
    # Find all checkpoints
    print(f"Searching for checkpoints in {args.input_dir}...")
    checkpoints = find_checkpoints(args.input_dir)
    
    if not checkpoints:
        print(f"No checkpoint directories found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for checkpoint in checkpoints:
        print(f"  - {os.path.basename(checkpoint)}")
    print()
    
    if "valid" in args.questions_file:
        questions_file = "/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/combined_non_numeric_all_validation.jsonl"
    if "fivenew" in args.questions_file:
        # questions_file = "/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_fivenewscombined-retrieval_1000_30.jsonl"
        questions_file = "/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_1000_30.jsonl"
    # elif "test" in args.questions_file:
    #     questions_file = "/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian_207_free_3_cleaned.jsonl"
    else:
        pass
        # raise ValueError(f"Invalid questions file: {args.questions_file}")
    
    if "binary" in args.mode:
        questions_file = "/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/binary_test.jsonl"
    
    if "retrieval" in args.mode:
        if "binary" not in args.mode:
            questions_file = "/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian-retrieval-new-30_207_free_3_cleaned.jsonl"
        if "fivenew" in args.questions_file:
            questions_file = "/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_fivenewscombined-retrieval_1000_30.jsonl"
        if "news5" in args.questions_file:
            questions_file = "/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_1000_30.jsonl"
        if "test5" in args.questions_file:
            questions_file = "/fast/nchandak/forecasting/newsdata/testset/withretrieval/o4-mini-high_test5news_302_30.jsonl"
            
    if "valid" in args.questions_file:
        # questions_file = "/fast/nchandak/forecasting/newsdata/retrieval/data/precompiled/ranked_queries_o4-mini-high_validation-theguardian_207_30.jsonl"
        questions_file = "/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_validation-retrieval_207_30.jsonl"
    
    if "retrievalbinary" in args.mode:
        questions_file = "/fast/nchandak/forecasting/datasets/metaculus/fromOct2025/with_retrieval/metaculusOct_30.jsonl"
        
    # Process each checkpoint
    successful = 0
    failed = 0
    skipped = 0
    
    for checkpoint_dir in checkpoints:
        step_name = os.path.basename(checkpoint_dir)
        print(f"Processing checkpoint: {step_name}")
        
        try:
            step_num = extract_step_number(checkpoint_dir)
            model_dir = find_model_directory(checkpoint_dir)
            
            print(f"  Model directory: {os.path.basename(model_dir)}")
            
            # Check if results already exist
            if skip_existing:
                result_file = os.path.join(args.output_dir, f"eval_results_step_{step_num}.json")
                if os.path.exists(result_file):
                    print(f"  Results already exist for step {step_num}, skipping...")
                    skipped += 1
                    continue
            
            # Prepare output and log file names
            output_file = os.path.join(args.output_dir, f"eval_results_step_{step_num}.json")
            log_file = os.path.join(args.log_dir, f"eval_log_step_{step_num}.log")
            
            print(f"  Output: {output_file}")
            print(f"  Log: {log_file}")
            
            # Run eval_freeform
            success, exit_code, duration_msg = run_eval_freeform(model_dir, output_file, log_file, args.mode, questions_file)
            
            if success:
                print(f"  ✓ Completed successfully in {duration_msg}")
                successful += 1
            else:
                print(f"  ✗ Failed with exit code {exit_code} after {duration_msg}")
                print(f"  Check log file: {log_file}")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error processing {step_name}: {str(e)}")
            failed += 1
        
        print()
    
    # Print summary
    print("Batch evaluation completed!")
    print(f"Results saved in: {args.output_dir}")
    print(f"Logs saved in: {args.log_dir}")
    print()
    print("Summary:")
    print(f"Total checkpoints found: {len(checkpoints)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Results directory: {args.output_dir}")
    print(f"Logs directory: {args.log_dir}")

if __name__ == "__main__":
    main() 