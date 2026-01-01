#!/usr/bin/env python3
"""
Script to run summarization on input files using specified models and forecasting-relevant prompts.
"""

import json
import os
import argparse
from generate_summaries import SummaryGenerator
from prompt_templates import get_all_prompt_functions, TARGET_LENGTHS

def get_forecasting_prompts():
    """Get only the prompts that are relevant to forecasting questions."""
    all_prompts = get_all_prompt_functions()
    forecasting_prompts = {
        "forecast_focused_summary": all_prompts["forecast_focused_summary"],
        "forecast_evidence_summary": all_prompts["forecast_evidence_summary"],
        "halawi": all_prompts["halawi"],
        "create_forecast_summarization_prompt": all_prompts["create_forecast_summarization_prompt"]
    }
    return forecasting_prompts

def run_summarization(input_file: str, model_dir: str):
    """
    Run summarization on the input file using the specified model.
    Modifies the original file directly by adding summary entries if they don't exist.
    
    Args:
        input_file (str): Path to the input JSON file
        model_dir (str): Path to the model directory
    """
    # Load the data
    print(f"Loading data from {input_file}...")
    if input_file.endswith('.jsonl'):
        data = {}
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                if isinstance(obj, dict):
                    #data.update(obj)
                    data[i] = obj
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    print(f"Loaded {len(data)} entries")
    
    # Get forecasting-relevant prompts
    forecasting_prompts = get_forecasting_prompts()
    print(f"Using {len(forecasting_prompts)} forecasting-relevant prompts: {list(forecasting_prompts.keys())}")
    
    # Create the summary generator
    generator = SummaryGenerator(
        source_path=input_file,
        output_dir=os.path.dirname(input_file),  # Save in same directory as input file
        model_path=model_dir
    )
    
    # Track if any changes were made
    file_modified = False
    
    # Run summarization for each prompt and target length
    # for prompt_name, prompt_func in forecasting_prompts.items():
    #     for target_length in [200]: # TARGET_LENGTHS[2:]:
    
    # prompt_name = "halawi"
    prompt_name = "create_forecast_summarization_prompt"
    prompt_func = forecasting_prompts[prompt_name]
    target_length = 200
    print(f"\n{'='*60}")
    print(f"Processing: {prompt_name} with length {target_length}")
    print(f"{'='*60}")
    
    # Check if summaries already exist for this prompt and length
    summaries_exist = check_summaries_exist(data, prompt_name, target_length, generator.model_name)
    
    if summaries_exist:
        print(f"Summaries for {prompt_name} length {target_length} already exist, skipping...")
        return
    else :
        print(f"Summaries for {prompt_name} length {target_length} do not exist, generating...")
        
    # Generate summaries
    summarized_data = generator.generate_summaries_for_prompt(
        data, prompt_name, prompt_func, target_length
    )
    
    # Update the original data with new summaries
    data = summarized_data
    file_modified = True
    
    # Print summary statistics
    total_summaries = 0
    for entry_key, entry in summarized_data.items():
        summary_count = 0
        for doc in entry['relevant_docs']:
            if isinstance(doc, list) and len(doc) > 3:  # Check if it has summary entries
                summary_count += len([item for item in doc[3:] if isinstance(item, dict) and item.get('type') == 'summary'])
        total_summaries += summary_count
    
    print(f"Generated {total_summaries} summaries across {len(summarized_data)} entries")

    # Save the modified data back to the original file if any changes were made
    if file_modified:
        print(f"\nSaving modified data back to {input_file}...")
        if input_file.endswith('.jsonl'):
            with open(input_file, 'w', encoding='utf-8') as f:
                # Save only the entry values, one per line
                for entry in data.values():
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        print("File updated successfully!")
    else:
        print("\nNo new summaries were generated - all summaries already exist.")

def check_summaries_exist(data: dict, prompt_name: str, target_length: int, model_name: str) -> bool:
    """
    Check if summaries already exist for the given prompt, length, and model.
    
    Args:
        data (dict): The data to check
        prompt_name (str): Name of the prompt
        target_length (int): Target length
        model_name (str): Name of the model
        
    Returns:
        bool: True if summaries exist, False otherwise
    """
    exists = False 
    for entry_key, entry in data.items():
        for doc in entry['relevant_docs']:
            if isinstance(doc, list) and len(doc) > 3:
                for item in doc[3:]:
                    if (isinstance(item, dict) and 
                        item.get('type') == 'summary' and
                        item.get('prompt_name') == prompt_name and
                        item.get('target_length') == target_length and
                        item.get('model') == model_name):
                        exists = True
                    else:
                        return False
    return exists

def main():
    """Main function to parse arguments and run summarization."""
    parser = argparse.ArgumentParser(description="Run summarization on input files using specified models")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="/fast/nchandak/forecasting/newsdata/ameya_retrieval/downloaded_files/o4-mini-test-set/o4-mini-high_theguardian-retrieval_207_free_3_cleaned.jsonl",
        help="Path to the input JSON file"
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="/fast/nchandak/models/Qwen3-32B",
        help="Path to the model directory"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return 1
    
    # Validate model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory {args.model_dir} does not exist")
        return 1
    
    print(f"Input file: {args.input_file}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {os.path.dirname(args.input_file)}")
    print()
    
    try:
        run_summarization(args.input_file, args.model_dir)
        print("\nSummarization completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during summarization: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 