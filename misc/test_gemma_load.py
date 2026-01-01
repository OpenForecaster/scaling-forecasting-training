#!/usr/bin/env python3
"""
Test script to load and use the Gemma-3-4B-IT model from the specified path using vLLM.
"""

from vllm import LLM, SamplingParams

def load_gemma_model(model_path="/fast/nchandak/models/gemma-3-4b-it-text/"):
    """Load the Gemma model using vLLM."""
    print(f"Loading Gemma model from: {model_path}")
    
    # Load model with vLLM
    # Note: Gemma-3 requires bfloat16 or float32, not float16
    llm = LLM(
        model=model_path,
        dtype="bfloat16",  # Use bfloat16 for Gemma-3
        tensor_parallel_size=1,  # Adjust if using multiple GPUs
    )
    
    print(f"✅ Model loaded successfully!")
    print(f"Model loaded on device(s)")
    
    return llm

def generate_text_batch(llm, prompts, max_tokens=100, temperature=0.7, use_chat_template=True):
    """Generate texts for a batch of prompts using vLLM."""
    batch_formatted_prompts = []
    if use_chat_template:
        tokenizer = llm.get_tokenizer()
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_formatted_prompts.append(formatted)
    else:
        batch_formatted_prompts = prompts
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    
    outputs = llm.generate(batch_formatted_prompts, sampling_params)
    
    # Each output corresponds to an input prompt
    generated_texts = []
    for output in outputs:
        text = output.outputs[0].text if output.outputs else ""
        generated_texts.append(text)
    return generated_texts

if __name__ == "__main__":
    # Load model
    llm = load_gemma_model()
    
    # Test batched generation
    test_prompts = [
        "What is the capital of France?",
        "Who wrote the play Hamlet?",
        "What is the largest planet in our solar system?"
    ]
    print("\n📝 Test prompts:")
    for i, p in enumerate(test_prompts):
        print(f"Prompt {i+1}: {p}")
    print("\nGenerating batched responses...")
    
    responses = generate_text_batch(llm, test_prompts, max_tokens=50)
    for i, (prompt, resp) in enumerate(zip(test_prompts, responses), 1):
        print(f"\n🤖 Model response {i} (prompt: {prompt}):\n{resp}\n")

