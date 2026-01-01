from huggingface_hub import snapshot_download
import os

model_names = [
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-14B",
    # "Qwen/Qwen3-32B"
    # "Qwen/Qwen3-4B-Base",
    # "Qwen/Qwen3-4B-Base"
    "meta-llama/Llama-4-Scout-17B-16E-Instruct"
]

base_dir = "/fast/nchandak/models/"

for model in model_names:
    model_id = model.split("/")[-1]
    local_dir = os.path.join(base_dir, model_id)
    print(f"Downloading {model} to {local_dir}...")
    
    # For Llama-4-Scout, ensure we get all tokenizer files
    if "Llama-4" in model:
        # First, try to download tokenizer files specifically
        try:
            snapshot_download(
                repo_id=model,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["tokenizer*", "*.json", "*.model"],
                resume_download=True
            )
            print(f"Downloaded tokenizer files for {model}")
        except Exception as e:
            print(f"Warning: Could not download tokenizer files separately: {e}")
        
        # Then download everything else
        snapshot_download(
            repo_id=model,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
    else:
        snapshot_download(
            repo_id=model,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
