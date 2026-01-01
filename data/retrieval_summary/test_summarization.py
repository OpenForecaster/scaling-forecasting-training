#!/usr/bin/env python3
"""
Test script to verify the summarization pipeline works with the new file structure.
"""

import json
import os
from generate_summaries import SummaryGenerator

def test_summarization():
    """Test the summarization pipeline with a small sample."""
    
    # Test file path
    test_file = "/fast/nchandak/forecasting/newsdata/ameya_retrieval/downloaded_files/o4-mini-test-set/o4-mini-high_theguardian-retrieval_207_free_3_cleaned copy.jsonl"
    output_dir = "/fast/nchandak/forecasting/retrieval_summary/test"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load a small sample of data
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take only the first 2 entries for testing
    test_data = {k: v for k, v in list(data.items())[:2]}
    
    print(f"Testing with {len(test_data)} entries")
    
    # Save test data
    test_file_path = os.path.join(output_dir, "test_data.jsonl")
    with open(test_file_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved test data to {test_file_path}")
    
    # Test with basic summary prompt
    generator = SummaryGenerator(
        source_path=test_file_path,
        output_dir=output_dir,
        # model_path="/fast/rolmedo/models/llama-3.3-70b-instruct"
        model_path="/fast/nchandak/models/Qwen3-32B"
    )
    
    # Test with just one prompt and length
    from prompt_templates import create_forecast_summarization_prompt, TARGET_LENGTHS
    
    print("Testing basic summary generation...")
    
    # Generate summaries for basic_summary with length 50
    summarized_data = generator.generate_summaries_for_prompt(
        test_data, "create_forecast_summarization_prompt", create_forecast_summarization_prompt, 200
    )
    
    # Save the result
    output_file = os.path.join(output_dir, "test_create_forecast_summarization_prompt_length200.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summarized_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved summarized data to {output_file}")
    
    # Check the structure
    print("\nChecking output structure...")
    for entry_key, entry in summarized_data.items():
        print(f"Entry {entry_key}:")
        print(f"  Original docs: {len([d for d in entry['relevant_docs'] if d[0] != 'summary'])}")
        print(f"  Summary docs: {len([d for d in entry['relevant_docs'] if d[0] == 'summary'])}")
        
        # Show a sample summary
        summary_docs = [d for d in entry['relevant_docs'] if d[0] == 'summary']
        if summary_docs:
            print(f"  Sample summary: {summary_docs[0][2]['summary'][:10000]}...")
        print()

if __name__ == "__main__":
    test_summarization() 