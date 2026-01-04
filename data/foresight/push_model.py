#!/usr/bin/env python3
"""
Script to push a model to HuggingFace Hub
"""
from huggingface_hub import HfApi, login
import os
import tempfile

# Configuration
LOCAL_MODEL_PATH = "/fast/nchandak/forecasting/models/Qwen3-8B-RL"
REPO_ID = "nikhilchandak/OpenForecaster-8B"

MODEL_CARD = """---
license: mit
base_model: Qwen/Qwen3-8B
tags:
  - forecasting
  - reasoning
  - question-answering
  - reinforcement-learning
  - calibration
datasets:
  - nikhilchandak/OpenForesight
language:
  - en
---

# OpenForecaster-8B

**OpenForecaster-8B** is a specialized language model for forecasting and predicting future events. This model is post-trained from **Qwen3-8B** using reinforcement learning on the [OpenForesight dataset](https://huggingface.co/datasets/nikhilchandak/OpenForesight).

## Model Description

OpenForecaster-8B is designed to make calibrated predictions on open-ended questions about future events. The model has been trained to:
- Provide calibrated confidence estimates when asked to do so
- Reason about uncertainty and future scenarios
- Leverage retrieved information (when provided in context) to improve predictions

## Training

This model was trained on the **OpenForesight** dataset, which contains over 52,000 forecasting questions generated from global news events. The training was done using GRPO optimizing a joint reward function combining accuracy and brier score.

**Base Model**: Qwen3-8B  
**Training Dataset**: [OpenForesight](https://huggingface.co/datasets/nikhilchandak/OpenForesight)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "nikhilchandak/OpenForecaster-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# template 
prompt = "What is the likelihood that [future event] will occur by [date]?"
# example
prompt = "Who will become the next Prime Minister of India based on the general election to be held in 2029? Provide specific predictions with probabilities."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=8192)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(prediction)
```

## Performance

OpenForecaster-8B achieves competitive performance with much larger models like DeepSeek-v3 and Qwen3-235B-A22B on forecasting benchmarks. Key improvements include:
- **Improved Accuracy**: Better prediction of future events
- **Better Calibration**: More reliable confidence estimates 
- **Enhanced Consistency**: Reduced logical violations in predictions

## More Information

For more details about the model, training process, and evaluation results, please visit our website:

**🌐 [https://openforecaster.github.io](https://openforecaster.github.io)**

## Citation

```bibtex
@article{openforesight2025,
  title  = {Scaling Open-Ended Reasoning To Predict the Future},
  author = {Chandak, Nikhil and Goel, Shashwat and Prabhu, Ameya and Hardt, Moritz and Geiping, Jonas},
  year   = {2025}
}
```

## License

This model is released under the MIT License.

## Contact

For questions or issues, please visit our [website](https://openforecaster.github.io) or open an issue on the model repository.
"""

def create_model_card(model_path):
    """Create README.md model card in the model directory"""
    readme_path = os.path.join(model_path, "README.md")
    print(f"Creating model card at {readme_path}...")
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD)
    print("✓ Model card created")

def push_model():
    """Push model card (README.md) to HuggingFace Hub"""
    
    # Login to HuggingFace (will use HF_TOKEN from environment or cached token)
    print("Logging in to HuggingFace...")
    try:
        login()
    except Exception as e:
        print(f"Login failed: {e}")
        print("Please set HF_TOKEN environment variable or run 'huggingface-cli login'")
        return
    
    print(f"Target repo: {REPO_ID}")
    
    # Initialize HuggingFace API
    api = HfApi()
    
    try:
        # Create model card in a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(MODEL_CARD)
            temp_readme_path = f.name
        
        print(f"\nUploading model card to {REPO_ID}...")
        
        # Upload just the README.md file
        api.upload_file(
            path_or_fileobj=temp_readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add/update model card for OpenForecaster-8B",
        )
        
        # Clean up temp file
        os.unlink(temp_readme_path)
        
        print(f"\n✓ Successfully pushed model card to https://huggingface.co/{REPO_ID}")
        
    except Exception as e:
        print(f"\n✗ Error pushing model card: {e}")
        raise

if __name__ == "__main__":
    push_model()

