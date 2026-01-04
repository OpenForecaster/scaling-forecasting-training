#!/usr/bin/env python3
"""
Relevant Documents Processor for RAG (Retrieval-Augmented Generation)

Purpose:
    Processes multiple JSON files containing retrieved news articles, combines relevant documents,
    ranks by relevance score, and keeps top-K documents for each forecasting question.

Main Functions:
    - process_files(): Processes all JSON files and combines relevant docs
    - extract_relevant_docs(): Extracts and organizes documents by entry
    - load_json_file(): Loads and parses JSON files

Input:
    - Directory containing multiple JSON files with relevant_docs field
    - Each file contains scored document retrievals from different news sources

Output:
    - Combined JSON/JSONL with top-K relevant documents per question
    - Documents sorted by relevance score across all sources

Usage:
    python process_relevant_docs.py --input_dir /path/to/files --output_file output.jsonl --top_k 10
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_relevant_docs(data: Dict[str, Any], source: str) -> Dict[str, List[Tuple[float, str, Dict[str, Any]]]]:
    """
    Extract all relevant_docs from a data dictionary, organized by entry.
    
    Args:
        data: The JSON data
        source: The news source (e.g., 'cnbc', 'cnn', 'dw', 'guardian')
    
    Returns:
        Dictionary mapping entry_key to list of tuples: (score, doc_id, doc_content)
    """
    entry_docs = {}
    
    for entry_key, entry_data in data.items():
        if 'relevant_docs' in entry_data and isinstance(entry_data['relevant_docs'], list):
            docs = []
            for doc in entry_data['relevant_docs']:
                if isinstance(doc, list) and len(doc) >= 3:
                    try:
                        score = float(doc[0])
                        doc_id = doc[1]
                        doc_content = doc[2].copy() if isinstance(doc[2], dict) else doc[2]
                        
                        # Add source field to the document content
                        if isinstance(doc_content, dict):
                            doc_content['source'] = source
                        
                        docs.append((score, doc_id, doc_content))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed doc in entry {entry_key}: {e}")
                        continue
            
            if docs:
                entry_docs[entry_key] = docs
    
    return entry_docs

def get_entry_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the structure of an entry (everything except relevant_docs).
    All entries should have the same structure.
    """
    for entry_key, entry_data in data.items():
        if 'relevant_docs' in entry_data:
            # Return everything except relevant_docs
            return {k: v for k, v in entry_data.items() if k != 'relevant_docs'}
    return {}

def process_files(input_dir: str, output_file: str, top_k: int = 10) -> None:
    """
    Process all JSON files in the input directory.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Output JSON file path
        top_k: Number of top documents to keep per entry
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Load the first file to get entry keys and structure
    first_data = load_json_file(str(json_files[0]))
    entry_keys = list(first_data.keys())
    
    print(f"Found {len(entry_keys)} entries")
    
    # Collect all relevant docs for each entry across all files
    combined_entries = {}
    entry_structures = {}  # Store the structure for each entry
    total_docs = 0
    
    for entry_key in entry_keys:
        combined_entries[entry_key] = []
    
    for json_file in json_files:
        try:
            data = load_json_file(str(json_file))
            
            # Extract source from filename (e.g., "final_queries_o4mini_x_cnbc.json" -> "cnbc")
            source = json_file.stem.split('_x_')[-1] if '_x_' in json_file.stem else json_file.stem
            
            entry_docs = extract_relevant_docs(data, source)
            
            # Add docs to the combined entries and store structure
            for entry_key, docs in entry_docs.items():
                if entry_key in combined_entries:
                    combined_entries[entry_key].extend(docs)
                    total_docs += len(docs)
                    
                    # Store the structure from this file (they should be the same)
                    if entry_key not in entry_structures:
                        entry_data = data[entry_key]
                        entry_structures[entry_key] = {k: v for k, v in entry_data.items() if k != 'relevant_docs'}
            
            print(f"  Added {sum(len(docs) for docs in entry_docs.values())} docs from {json_file.name} (source: {source})")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"\nTotal relevant docs collected: {total_docs}")
    
    # Process each entry: sort by score and keep top K
    print(f"Processing each entry to keep top {top_k} docs...")
    processed_entries = {}
    
    for entry_key, docs in combined_entries.items():
        if docs:
            # Sort by score (descending) and keep top K
            docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = docs[:top_k]
            processed_entries[entry_key] = top_docs
            
            print(f"  {entry_key}: {len(docs)} docs → {len(top_docs)} docs (top score: {top_docs[0][0]:.6f})")
    
    # Create output structure in the same format as input files
    print(f"\nSaving {len(processed_entries)} entries to {output_file}...")
    
    output_data = {}
    jsonl_output_file = os.path.splitext(output_file)[0] + ".jsonl"

    # Build output_data (dict keyed by entry_key) and also a list for jsonl
    jsonl_entries = []
    for entry_key, top_docs in processed_entries.items():
        # Create entry with its original structure + top relevant docs
        entry_data = entry_structures[entry_key].copy()
        entry_data["relevant_docs"] = [
            [str(score), doc_id, doc_content] 
            for score, doc_id, doc_content in top_docs
        ]
        output_data[entry_key] = entry_data

        # For jsonl, add entry_data (without entry_key) to the list
        jsonl_entries.append(entry_data)

    # Save as JSON (with entry_key)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Also save as JSONL (one entry per line, no entry_key)
    with open(jsonl_output_file, 'w', encoding='utf-8') as f_jsonl:
        for entry in jsonl_entries:
            f_jsonl.write(json.dumps(entry, ensure_ascii=False) + '\n')

    total_output_docs = sum(len(docs) for docs in processed_entries.values())
    print(f"✓ Successfully saved {len(processed_entries)} entries with {total_output_docs} total docs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process JSON files and extract top relevant docs")
    parser.add_argument(
        "--input_dir", 
        default="/fast/nchandak/forecasting/newsdata/ameya_retrieval/downloaded_files",
        help="Input directory containing JSON files"
    )
    parser.add_argument(
        "--output_file", 
        default=None,
        help="Output JSONL file path (default: saved in input directory)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10,
        help="Number of top documents to keep"
    )
    
    args = parser.parse_args()
    
    # Set default output file in input directory if not specified
    if args.output_file is None:
        args.output_file = os.path.join(args.input_dir, "top_10_relevant_docs.jsonl")
    
    print("Relevant Docs Processor")
    print("=" * 40)
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Top K: {args.top_k}")
    print()
    
    try:
        process_files(args.input_dir, args.output_file, args.top_k)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 