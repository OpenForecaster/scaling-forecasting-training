"""
Two-stage approach for downloading and dequantizing GPT-OSS-120B:

STAGE 1: Download quantized files in batches (run multiple times)
STAGE 2: Load with dequantize=True and save bf16 version (run once at end)

Set MODE below to control which stage runs.
"""

# MODE = "DOWNLOAD_BATCH"  # Options: "DOWNLOAD_BATCH" or "DEQUANTIZE"
MODE = "DEQUANTIZE"

import os
import torch
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

model_id = "openai/gpt-oss-120b"
quantized_cache_dir = "/fast/nchandak/models/gpt-oss-120b-quantized"  # Temporary storage
output_dir = "/fast/nchandak/models/gpt-oss-120b-bf16"  # Final dequantized model

# ============================================================================
# STAGE 1: DOWNLOAD IN BATCHES
# ============================================================================
if MODE == "DOWNLOAD_BATCH":
    os.makedirs(quantized_cache_dir, exist_ok=True)
    
    print("Fetching list of files from repository...")
    all_files = list_repo_files(model_id)
    print(f"Total files: {len(all_files)}")
    
    # Filter for model weight files
    model_files = [f for f in all_files if f.endswith(('.safetensors', '.bin'))]
    config_files = [f for f in all_files if f.endswith(('.json', '.txt', '.md', '.py', '.model'))]
    
    print(f"\nModel weight files: {len(model_files)}")
    print(f"Config/tokenizer files: {len(config_files)}")
    print("\nModel files:")
    for i, f in enumerate(model_files):
        print(f"  [{i}] {f}")
    
    # BATCH 1: First 4 model files + all config files
    # batch_1 = model_files[:4] + config_files
    
    # BATCH 2: Next 4 model files (uncomment when ready)
    # batch_2 = model_files[4:8]
    
    # BATCH 3: Next 4 model files (uncomment when ready)
    # batch_3 = model_files[8:12]
    
    # BATCH 4: Remaining files (uncomment when ready)
    batch_4 = model_files[12:]
    
    # Set current batch here
    current_batch = batch_4
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING BATCH - {len(current_batch)} files")
    print(f"{'='*60}")
    for i, filename in enumerate(current_batch):
        print(f"\n[{i+1}/{len(current_batch)}] Downloading: {filename}")
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=quantized_cache_dir,
            local_dir_use_symlinks=False
        )
        print(f"  ✓ Completed: {filename}")
    
    print(f"\n{'='*60}")
    print(f"✓ Batch download complete!")
    print(f"Files saved to: {quantized_cache_dir}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. If more batches remain, update current_batch and run again")
    print("2. After all batches downloaded, set MODE='DEQUANTIZE' to convert to bf16")

# ============================================================================
# STAGE 2: DEQUANTIZE (run once after all files downloaded)
# ============================================================================
elif MODE == "DEQUANTIZE":
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    
    print(f"\n{'='*60}")
    print("LOADING AND DEQUANTIZING MODEL")
    print(f"{'='*60}")
    
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )
    
    print(f"\nLoading quantized model from: {quantized_cache_dir}")
    print("This will dequantize to bf16 format...")
    model = AutoModelForCausalLM.from_pretrained(quantized_cache_dir, **model_kwargs)
    
    # Patch config
    model.config.attn_implementation = "eager"
    
    print(f"\nSaving dequantized model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*60}")
    print("✓ DEQUANTIZATION COMPLETE!")
    print(f"Dequantized bf16 model saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nYou can now delete the quantized cache: {quantized_cache_dir}")
