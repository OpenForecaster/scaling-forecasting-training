import requests
import json
import time

# Your OpenRouter API Key
# from forecasting.inference.open_router import API_KEY
from inference.open_router_key import API_KEY

# List of Llama-2-70B variants available on OpenRouter
MODEL_VARIANTS = [
    # "meta-llama/llama-3.3-70b-instruct"
    "meta-llama/llama-3.1-8b-instruct:free"
    # "qwen/qwen-2.5-72b-instruct",
    # "qwen/qwen-2.5-7b-instruct",
    # "mistralai/mistral-small-24b-instruct-2501",
    
    # "deepseek/deepseek-r1-distill-llama-70b",
    # "deepseek/deepseek-r1-distill-qwen-14b",
    
    # "deepseek/deepseek-r1:free",
    
    # "meta-llama/llama-2-70b",
    # "togethercomputer/llama-2-70b-chat",
]

# List of prompts to send
PROMPTS = [
    "What is the capital of France?",
    "Explain the concept of entropy in simple terms.",
    "How does a transformer-based language model work? Keep it short.",
]
OUTPUT_DIR = "/fast/nchandak/forecasting/evals/halawi/"

# OpenRouter API URL
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Store responses
results = []

# Function to query OpenRouter
def query_openrouter(model, prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": 512,
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        print(response.json())
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        print(f"Error with {model}: {response.text}")
        return None

# Run inference on each model for each prompt
for model in MODEL_VARIANTS:
    for prompt in PROMPTS:
        print(f"Querying {model} with prompt: {prompt}")
        response = query_openrouter(model, prompt)
        results.append({"model": model, "prompt": prompt, "response": response})
        print(f"Model: {model}\nPrompt: {prompt}\nResponse: {response}\n\n")
        time.sleep(1)  # Avoid hitting rate limits

# Save results to JSON file
# output_file = "openrouter_llama2_inference_results.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=4)

# print(f"Inference complete. Results saved to {output_file}.")
