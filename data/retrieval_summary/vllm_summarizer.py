"""
This module provides functionality for summarizing text using vLLM with Llama 3.3 70B instruct model.
"""

from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class VLLMSummarizer:
    """
    Class for summarizing text using vLLM with Llama 3.3 70B instruct model.
    """
    
    def __init__(self, model_path: str = "/fast/rolmedo/models/llama-3.3-70b-instruct", 
                 max_tokens: int = 512, 
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 presence_penalty: float = 0.0):
        """
        Initialize the summarizer with the specified model and parameters.
        
        Args:
            model_path (str): Path to the model to use
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            presence_penalty (float): Presence penalty parameter
        """
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens
        )
        
        # Initialize vLLM model
        try:
            # Initialize vLLM model with tensor parallelism
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs for tensor parallelism
            )
            
            # Load tokenizer separately for prompt processing
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        except:
            # Fallback if your particular directory structure requires it
            model_path += "/snapshots/model/"
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs for tensor parallelism
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    def summarize_single(self, prompt: str) -> str:
        """
        Summarize a single article using the prompt provided.
        
        Args:
            prompt (str): The prompt containing the article and instructions
            
        Returns:
            str: The generated summary
        """
        outputs = self.llm.generate(prompt, self.sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text
    
    def summarize_batch(self, prompts: List[str], target_length: int = 100) -> List[str]:
        """
        Summarize a batch of articles using the prompts provided.
        vLLM will handle batching internally.
        
        Args:
            prompts (List[str]): List of prompts, each containing an article and instructions
            
        Returns:
            List[str]: List of generated summaries
        """
        # Create a new SamplingParams object with the same parameters as the original
        # but with adjusted max_tokens based on target length
        current_sampling_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            presence_penalty=self.sampling_params.presence_penalty,
            max_tokens=8192 # target_length * 4  # Modelling each word to be roughly 4 tokens (max) in length
        )
        
        local_prompts = []
        for prompt in prompts:
            
            try:
                chat = [
                {
                    "role": "user",
                    "content": prompt,
                },
                # {
                #     "role": "assistant",
                #     "content": "Let me reason about all the information provided step by step.\n<think>"
                # }
                ]
                if 'qwen3' in self.model_name.lower():
                    local_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, 
                                                            add_generation_prompt=True, enable_thinking=True)
                else:
                    local_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
            except Exception as e:
                print(f"Error processing prompt: {e}")
                local_prompt = prompt
                
            local_prompts.append(local_prompt)
        
        
        outputs = self.llm.generate(local_prompts, current_sampling_params)
        summaries = [output.outputs[0].text.strip() for output in outputs]
        
        num_tokens = []
        actual_summaries = []
        for summary in summaries:
            actual_summary = summary 
            if "</think>" in summary:
                actual_summary = summary.split("</think>")[-1].strip()
                
            if "<summary>" in actual_summary:
                actual_summary = actual_summary.split("<summary>")[-1].split("</summary>")[0].strip()

            actual_summaries.append(actual_summary)
            summary_tokens = len(self.tokenizer.encode(actual_summary))
            num_tokens.append(summary_tokens)
            # print(f"Summary: {summary}")
            # print(f"Actual Summary: {actual_summaries[-1]}")
            # print("--------------------------------")
        
        print(f"Average number of tokens: {sum(num_tokens) / len(num_tokens)}")
        return actual_summaries

# Convenience function for non-async contexts
def summarize_text(prompt: str, model_path: str = "/fast/rolmedo/models/llama-3.3-70b-instruct") -> str:
    """
    Synchronous wrapper for summarizing a single text.
    
    Args:
        prompt (str): The prompt containing the article and instructions
        model_path (str): Path to the model to use
        
    Returns:
        str: The generated summary
    """
    summarizer = VLLMSummarizer(model_path=model_path)
    result = summarizer.summarize_single(prompt)
    return result

# Convenience function for non-async contexts with batch processing
def summarize_batch(prompts: List[str], model_path: str = "/fast/rolmedo/models/llama-3.3-70b-instruct") -> List[str]:
    """
    Synchronous wrapper for summarizing multiple texts.
    
    Args:
        prompts (List[str]): List of prompts, each containing an article and instructions
        model_path (str): Path to the model to use
        
    Returns:
        List[str]: List of generated summaries
    """
    summarizer = VLLMSummarizer(model_path=model_path)
    results = summarizer.summarize_batch(prompts)
    return results 