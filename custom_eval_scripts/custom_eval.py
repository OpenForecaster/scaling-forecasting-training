"""
Quick model evaluation script for manual testing with custom prompts.
Useful for ad-hoc testing and "vibe checking" models with predefined forecasting questions.
Loads a model with vLLM and generates responses to hardcoded prompts.
"""

import torch
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Manual prompts to test
MANUAL_PROMPTS = [
    """You are a forecasting expert. Based on current trends and historical data, what is your prediction for global AI chip demand growth in 2025?
Think step by step and provide your final answer with a probability estimate.""",
    
    """Question: Will SpaceX successfully land humans on Mars before 2030?
Provide your reasoning and confidence level.""",
    
    """Who will become the next Prime Minister of India based on the general election to be held in 2029? 
Provide specific predictions with probabilities.""",


    """Who will become the next Prime Minister of India?""",

"""You are a forecasting expert. By when will AGI be achieved? Think hard and step by step. Give me your best prediction (exact date -- month and year) with probability.""",
#     """Analyze the following scenario: A major tech company announces a breakthrough in quantum computing. What are the likely market impacts in the next 6 months?
# Give specific predictions with probabilities.""",
    
#     """Forecasting Question: What will be the average global temperature anomaly in 2026 compared to pre-industrial levels?
# Provide your best estimate with reasoning and uncertainty bounds."""
]


Q1 = """You will be asked a forecasting question (which might be from the past). 
You have to come up with the best guess for the final answer. Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).  

Question Title: By when will AGI be achieved?
Question Background: AGI is often defined as highly autonomous systems that outperform humans at most economically valuable work. Microsoft defines it as an system that can generate at least $100 billion in profits.
Resolution Criteria: 
<ul> <li> <b>Source of Truth</b>: The question will resolve based on reports from reputed news outlets and official announcements from the company which declares AGI first. </li> <li> <b>Resolution Date</b>: The resolution occurs on the calendar date when a company officially and publicly announces that they have achieved AGI or by releasing an AGI system. </li> <li> <b>Accepted Answer Format</b>: The exact date (month and year) when AGI will be achieved. </li> </ul> 
Expected Answer Type: string (date)  

Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.  You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.  Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""

MANUAL_PROMPTS.append(Q1)

def load_model_and_tokenizer(model_path: str):
    """Load model with vLLM and tokenizer"""
    logger.info(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Initialize vLLM model
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    
    logger.info("Model loaded successfully")
    return model, tokenizer

def generate_responses(model, tokenizer, prompts, max_tokens=2048, num_generations=1):
    """Generate responses for given prompts"""
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_tokens,
        n=num_generations,
    )
    
    # Format prompts with chat template if available
    formatted_prompts = []
    for prompt in prompts:
        try:
            chat = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)
        except:
            # Fallback to raw prompt if chat template fails
            formatted_prompts.append(prompt)
    
    # Generate
    logger.info(f"Generating responses for {len(prompts)} prompts...")
    outputs = model.generate(formatted_prompts, sampling_params)
    
    return outputs

def print_results(prompts, outputs):
    """Print prompts and responses in a nice format"""
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        print("\n" + "="*80)
        print(f"PROMPT {i}:")
        print("-"*80)
        print(prompt)
        print("\n" + "-"*80)
        print(f"RESPONSE:")
        print("-"*80)
        
        for j, generation in enumerate(output.outputs, 1):
            if len(output.outputs) > 1:
                print(f"\n[Generation {j}]")
            print(generation.text)
            
        print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick model vibe check with manual prompts")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to model directory")
    parser.add_argument('--max_tokens', type=int, default=16384, help="Max tokens to generate")
    parser.add_argument('--num_generations', type=int, default=1, help="Number of generations per prompt")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    
    # Generate responses
    outputs = generate_responses(
        model, 
        tokenizer, 
        MANUAL_PROMPTS, 
        max_tokens=args.max_tokens,
        num_generations=args.num_generations
    )
    
    # Print results
    print_results(MANUAL_PROMPTS, outputs)
    
    logger.info("Done!")

