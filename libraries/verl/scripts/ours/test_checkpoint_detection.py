#!/usr/bin/env python3
"""
Test script to verify checkpoint detection logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from convert_checkpoints import find_checkpoints

def test_checkpoint_detection():
    """Test the checkpoint detection function."""
    
    # Test with the actual checkpoint directory
    input_dir = "/fast/nchandak/forecasting/training/verl/checkpoints/"
    input_dir = "/fast/nchandak/forecasting/training/verl/checkpoints/rl-test/Qwen3-1.7B-2048-4096"
    input_dir = "/fast/nchandak/forecasting/training/verl/checkpoints/rl-test/Qwen3-4B-2048-8192"
    
    print(f"Testing checkpoint detection in: {input_dir}")
    
    checkpoints = find_checkpoints(input_dir)
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for i, checkpoint in enumerate(checkpoints):
        print(f"\n{i+1}. {checkpoint['model_name']}-{checkpoint['project_name']}-checkpoint{checkpoint['step_num']}")
        print(f"   Project: {checkpoint['project_name']}")
        print(f"   Global step dir: {checkpoint['global_step_dir']}")
        print(f"   Actor dir: {checkpoint['actor_dir']}")
        print(f"   Expected HF dir: {checkpoint['expected_hf_dir']}")
        print(f"   Already converted: {checkpoint['is_converted']}")
        
        # Show the expected command
        print(f"   Expected conversion command:")
        print(f"   python convert_fsdp_to_hf.py --fsdp_checkpoint_path=\"{checkpoint['actor_dir']}\" --huggingface_model_path=\"{checkpoint['actor_dir']}/huggingface\" --output_path=\"{checkpoint['expected_hf_dir']}\"")

if __name__ == "__main__":
    test_checkpoint_detection() 