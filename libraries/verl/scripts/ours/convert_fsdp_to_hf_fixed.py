#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from glob import glob
from collections import defaultdict
import os


def main(fsdp_checkpoint_path, huggingface_model_path, output_path):
    state_dict = defaultdict(list)

    # First, let's find the actual world size by checking what files exist
    model_files = glob(f"{fsdp_checkpoint_path}/model_world_size_*_rank_*.pt")
    if not model_files:
        raise FileNotFoundError(f"No model files found in {fsdp_checkpoint_path}")
    
    # Extract world size from the first file
    import re
    match = re.search(r'model_world_size_(\d+)_rank_', model_files[0])
    if not match:
        raise ValueError(f"Could not extract world size from filename: {model_files[0]}")
    
    world_size = int(match.group(1))
    print(f"Detected world size: {world_size}")

    # Load all rank files
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
            
        print(f'Loading {filepath}')
        this_state_dict = torch.load(filepath, map_location='cpu')  # Load to CPU first
        
        for key, value in this_state_dict.items():
            # Convert to local tensor and move to CPU to avoid device mismatch
            if hasattr(value, 'to_local'):
                local_tensor = value.to_local().cpu()
            else:
                local_tensor = value.cpu()
            state_dict[key].append(local_tensor)

    # Concatenate tensors (all should be on CPU now)
    print("Concatenating tensors...")
    for key in state_dict:
        if len(state_dict[key]) > 1:
            state_dict[key] = torch.cat(state_dict[key], dim=0)
        else:
            state_dict[key] = state_dict[key][0]

    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(huggingface_model_path)
    
    print("Creating model from config...")
    model = AutoModelForCausalLM.from_config(config)
    
    print("Loading state dict...")
    model.load_state_dict(state_dict)

    print(f"Saving model to {output_path}...")
    model.save_pretrained(output_path, max_shard_size="10GB")

    print("Loading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("Conversion completed successfully!")


if __name__ == "__main__":
    fire.Fire(main)
