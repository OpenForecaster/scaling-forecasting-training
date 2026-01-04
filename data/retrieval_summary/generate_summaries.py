#!/usr/bin/env python3
"""
News Article Summarizer for Forecasting with Multiple Prompt Types

Purpose:
    Generates multiple types of summaries for retrieved news articles using vLLM and Llama 3.3 70B.
    Tests 6 different summarization prompts at 3 different target lengths (50, 100, 200 words).

Prompt Types:
    1. Basic Summary - Simple article summarization
    2. Forecast-Focused Summary - Summary relevant to forecasting question
    3. Key Facts Summary - Extract only key factual information
    4. Forecast Evidence Summary - Evidence specifically for forecasting
    5. Timeline-Oriented Summary - Chronological event summary
    6. Halawi - Prompt template from Halawi et al. (2024)

Main Classes:
    - SummaryGenerator: Handles bulk summary generation and dataset creation

Dependencies:
    - vLLM for efficient batch inference
    - prompt_templates.py for different summarization prompts
    - vllm_summarizer.py for vLLM wrapper

Usage:
    python generate_summaries.py --source_path /path/to/data --output_dir /path/to/output --model_path /path/to/model
"""

import os
import json
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from prompt_templates import get_all_prompt_functions, TARGET_LENGTHS
from vllm_summarizer import VLLMSummarizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SummaryGenerator:
    """
    Class for generating article summaries using different prompts and saving them to datasets.
    """
    
    def __init__(self, 
                 source_path: str = "/fast/nchandak/forecasting/newsdata/ameya_retrieval/downloaded_files/o4-mini-high_theguardian-retrieval_207_free_3_cleaned.jsonl",
                 output_dir: str = "/fast/nchandak/forecasting/retrieval_summary",
                 model_path: str = "/fast/rolmedo/models/llama-3.3-70b-instruct"):
        """
        Initialize the summary generator.
        
        Args:
            source_path (str): Path to the source JSON file
            output_dir (str): Directory to save the output files
            model_path (str): Path to the model to use
        """
        self.source_path = source_path
        self.output_dir = output_dir
        self.model_path = model_path
        # Fetch model name from model path
        self.model_name = model_path.split("/")[-1]
        self.prompt_functions = get_all_prompt_functions()
        
        # Initialize summarizer
        self.summarizer = VLLMSummarizer(model_path=self.model_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """
        Load the source JSON file.
        
        Returns:
            dict: The loaded data
        """
        with open(self.source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded data with {len(data)} entries")
        return data
    
    def get_output_filename(self, prompt_name: str, target_length: int):
        """
        Get the output filename for a given prompt and target length.
        
        Args:
            prompt_name (str): Name of the prompt function
            target_length (int): Target length for the summary
            
        Returns:
            str: Output filename
        """
        base_name = os.path.splitext(os.path.basename(self.source_path))[0]
        return f"{base_name}_{prompt_name}_length{target_length}_{self.model_name}.jsonl"
    
    def generate_summaries_for_prompt(self, 
                                      data: Dict[str, Any], 
                                      prompt_name: str, 
                                      prompt_func: callable, 
                                      target_length: int):
        """
        Generate summaries for data using a specific prompt and target length.
        
        Args:
            data (dict): The data to process
            prompt_name (str): Name of the prompt function
            prompt_func (callable): The prompt function
            target_length (int): Target length for the summary
            
        Returns:
            dict: The data with added summaries
        """
        logger.info(f"Generating summaries with {prompt_name} (length: {target_length})...")
        
        # Helper function to generate prompts based on the prompt function signature
        def generate_prompt(entry, doc_idx):
            doc = entry["relevant_docs"][doc_idx]
            article = doc[2]["maintext"]  # doc[2] is the document content
            return prompt_func(article, entry["question_title"], entry["background"], target_length=target_length)
        
        # Prepare all prompts at once
        all_prompts = []
        entry_doc_indices = []  # Keep track of which (entry_key, doc_idx) each prompt corresponds to
        
        logger.info("Preparing all prompts...")
        for entry_key, entry in tqdm(data.items(), desc="Collecting prompts"):
            for doc_idx in range(len(entry["relevant_docs"])):
                prompt = generate_prompt(entry, doc_idx)
                all_prompts.append(prompt)
                entry_doc_indices.append((entry_key, doc_idx))
        
        logger.info(f"Generated {len(all_prompts)} prompts for {prompt_name} (length: {target_length})")
        
        # Generate all summaries at once (vLLM will handle batching internally)
        logger.info("Generating summaries with vLLM...")
        summaries = self.summarizer.summarize_batch(all_prompts, target_length=target_length)
        logger.info("Summarization complete")
        
        # Create a copy of the data to modify
        output_data = {}
        
        # Update the data with the generated summaries
        logger.info("Updating data with summaries...")
        
        # Create a dictionary to store summaries for each entry
        summaries_by_entry = {}
        for i, (entry_key, doc_idx) in enumerate(tqdm(entry_doc_indices, desc="Organizing summaries")):
            if entry_key not in summaries_by_entry:
                summaries_by_entry[entry_key] = [None] * len(data[entry_key]["relevant_docs"])
            
            summary = summaries[i]
            ridx = summary.rfind(".")
            if ridx != -1:
                summary = summary[:ridx+1]
                
            summary_item = {
                "prompt_name": prompt_name,
                "target_length": target_length,
                "model": self.model_name,
                "summary": summary,
                "news_source": data[entry_key]["relevant_docs"][doc_idx][2].get("source", "unknown")
            }
            summaries_by_entry[entry_key][doc_idx] = summary_item
        
        # Apply the summaries to the data
        for entry_key, entry in tqdm(data.items(), desc="Updating data with summaries"):
            output_data[entry_key] = entry.copy()
            
            if entry_key in summaries_by_entry:
                # Add summary entries to relevant_docs
                for doc_idx, summary_item in enumerate(summaries_by_entry[entry_key]):
                    if summary_item is not None:
                        # Create a new entry in relevant_docs for the summary
                        prompt_name = summary_item["prompt_name"]
                        summary_doc = {
                            "type": "summary",
                            "prompt_name": summary_item["prompt_name"],
                            "target_length": summary_item["target_length"],
                            "model": summary_item["model"],
                            "summary": summary_item["summary"],
                            "source": summary_item["news_source"],
                            "original_doc_id": entry["relevant_docs"][doc_idx][1]  # Reference to original doc
                        }
                        
                        output_data[entry_key]["relevant_docs"][doc_idx].append(summary_doc)
                        
                        # # Add summary as a new entry in relevant_docs
                        # output_data[entry_key]["relevant_docs"].append([
                        #     "summary",  # Placeholder score for summary
                        #     f"summary_{entry_key}_{doc_idx}_{prompt_name}_{target_length}",  # Summary ID
                        #     summary_doc
                        # ])
        
        return output_data
    
    def run(self):
        """
        Run the summary generation process for all prompts and target lengths.
        """
        # Load the data
        data = self.load_data()
        
        # Limit to first 10 entries for testing (remove this line for full processing)
        data = {k: v for k, v in list(data.items())[:10]}
    
        # Generate summaries for each prompt and target length
        for prompt_name, prompt_func in self.prompt_functions.items():
            for target_length in TARGET_LENGTHS:
                output_path = os.path.join(self.output_dir, self.get_output_filename(prompt_name, target_length))
                
                # Skip if the output file already exists
                if os.path.exists(output_path):
                    logger.info(f"Output file {output_path} already exists, skipping...")
                    continue
                
                # Generate summaries
                summarized_data = self.generate_summaries_for_prompt(
                    data, prompt_name, prompt_func, target_length
                )
                
                # Save the data
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(summarized_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved data to {output_path}")

def main():
    """
    Main function to run the summary generation process.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate article summaries using different prompts")
    parser.add_argument(
        "--source_path", 
        type=str, 
        default="/fast/nchandak/forecasting/newsdata/ameya_retrieval/downloaded_files/o4-mini-high_theguardian-retrieval_207_free_3_cleaned.jsonl",
        help="Path to the source JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/fast/nchandak/forecasting/retrieval_summary",
        help="Directory to save the output files"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/fast/rolmedo/models/llama-3.3-70b-instruct",
        help="Path to the model to use"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the summary generator
    generator = SummaryGenerator(
        source_path=args.source_path,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
    
    # Run the summary generation process
    generator.run()

if __name__ == "__main__":
    main() 