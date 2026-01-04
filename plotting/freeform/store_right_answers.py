#!/usr/bin/env python3
import os
import json
import argparse
import glob
from collections import defaultdict
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Extract and store correct answers marked by judge models")
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/openrouter/theguardian_207/",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--judge", type=str, default="Qwen3_4B",
                       help="Judge model name (e.g., Qwen3_4B)")
    parser.add_argument("--continuous", action="store_true",
                       help="Use continuous_score fields instead of score_ fields")
    parser.add_argument("--threshold", type=float, default=0.9999,
                       help="Threshold for considering an answer correct (for binary scores)")
    return parser.parse_args()

def load_jsonl_file(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def extract_model_info_from_filename(filename):
    """
    Extract model name and context type from filename.
    
    Expected format: ModelName_evals_size_N_generations_M_context_type.jsonl
    Returns: (model_name, context_type, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract context type (with_article or no_article)
    if 'with_article' in name_without_ext:
        context_type = 'with_article'
    elif 'no_article' in name_without_ext:
        context_type = 'no_article'
    else:
        context_type = 'unknown'
    
    # Extract model name (everything before _evals or _eval)
    model_match = re.match(r'([^_]+(?:_[^_]*?)?(?:-\d+\.?\d*[bB])?)', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = name_without_ext.split('_')[0]
    
    # Extract number of generations
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    if model_name[-1] == '_':
        model_name = model_name[:-1]
    
    return model_name, context_type, num_generations

def get_correct_answers_for_file(file_path, judge, use_continuous=False, threshold=0.9999):
    """
    Extract correct answers marked by the judge model for a single file.
    
    Args:
        file_path: Path to the evaluation JSONL file
        judge: Judge model name (e.g., "Qwen3_4B")
        use_continuous: Whether to use continuous scores
        threshold: Threshold for binary scores to consider correct
        
    Returns:
        List of dictionaries with question info and correct answers
    """
    data = load_jsonl_file(file_path)
    print(f"  Loaded {len(data)} samples from {os.path.basename(file_path)}")
    
    # Determine judge field name
    if use_continuous:
        judge_field = f"continuous_score_{judge}"
    else:
        judge_field = f"score_{judge}"
    
    # Check if judge field exists
    available_fields = set()
    for item in data:
        for key in item.keys():
            if key.startswith("score_") or key.startswith("continuous_score_"):
                available_fields.add(key)
    
    if judge_field not in available_fields:
        print(f"  Warning: {judge_field} not found, available fields: {available_fields}")
        # Try fallback
        if use_continuous:
            fallback_field = f"score_{judge}"
        else:
            fallback_field = f"continuous_score_{judge}"
        if fallback_field in available_fields:
            print(f"  Using fallback field: {fallback_field}")
            judge_field = fallback_field
        else:
            print(f"  No judge field found, skipping file")
            return []
    
    correct_answers_data = []
    
    for item in data:
        question_id = item.get("idx", item.get("question_id"))
        question = item.get("question", item.get("question_title", ""))
        actual_answer = item.get("answer", item.get("solution", ""))
        answer_type = item.get("answer_type", "(no answer type)")
        extracted_answers = item.get("extracted_answer", [])
        judge_scores = item.get(judge_field, [])
        
        if not extracted_answers or not judge_scores:
            continue
            
        # Find answers marked as correct by the judge
        correct_answers = []
        
        for i, (answer, score) in enumerate(zip(extracted_answers, judge_scores)):
            if answer is None or i > 0:
                continue
                
            # Handle different score formats
            if isinstance(score, dict):
                # If score is a dict, extract the value
                score_value = list(score.values())[0] if score else 0
            elif isinstance(score, (int, float)):
                score_value = float(score)
            else:
                try:
                    score_value = float(score)
                except (ValueError, TypeError):
                    continue
            
            # Determine if this answer is correct based on score type and threshold
            is_correct = False
            if use_continuous:
                # For continuous scores, use the score directly
                is_correct = score_value > 0
            else:
                # For binary scores, check against threshold
                is_correct = score_value >= threshold
            
            if is_correct:
                correct_answers.append({
                    "answer": answer,
                    "score": score_value,
                    "generation_index": i
                })
        
        # Only add questions that have at least one correct answer
        if correct_answers:
            result_entry = {
                "question_id": question_id,
                "question": question,
                "actual_answer": actual_answer,
                "answer_type": answer_type,
                "judge_model": judge,
                "correct_answers": correct_answers,
                "total_generations": len(extracted_answers),
                "num_correct": len(correct_answers)
            }
            
            correct_answers_data.append(result_entry)
    
    print(f"  Found {len(correct_answers_data)} questions with correct answers")
    return correct_answers_data

def process_all_files(input_dir, judge, use_continuous=False, threshold=0.9999):
    """
    Process all evaluation files in the input directory and extract correct answers.
    
    Args:
        input_dir: Directory containing evaluation JSONL files
        judge: Judge model name
        use_continuous: Whether to use continuous scores
        threshold: Threshold for binary scores
        
    Returns:
        Dictionary mapping filenames to their correct answers data
    """
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")
    
    all_results = {}
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        
        if "qwen3" not in filename.lower() or "1.7b" not in filename.lower():
            continue
        
        print(f"Processing: {filename}")
        
        # Extract model info from filename
        model_name, context_type, num_generations = extract_model_info_from_filename(filename)
        
        # Skip certain files if needed (following the pattern from clean_passk.py)
        if '8b' in filename.lower() and context_type == 'with_article':
            print(f"  Skipping {filename} (8b with_article)")
            continue
        
        # Get correct answers for this file
        correct_answers = get_correct_answers_for_file(
            file_path, judge, use_continuous, threshold
        )
        
        if correct_answers:
            all_results[filename] = {
                "model_name": model_name,
                "context_type": context_type,
                "num_generations": num_generations,
                "judge_model": judge,
                "use_continuous": use_continuous,
                "threshold": threshold,
                "correct_answers_data": correct_answers
            }
    
    return all_results

def save_results_to_jsonl(all_results, output_dir):
    """
    Save the results to JSONL files in the output directory.
    
    Args:
        all_results: Dictionary of results from process_all_files
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    for filename, file_data in all_results.items():
        # Create output filename
        base_name = filename.replace('.jsonl', '')
        output_filename = f"{base_name}_correct_answers.jsonl"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the correct answers data as JSONL
        with open(output_path, 'w') as f:
            for entry in file_data["correct_answers_data"]:
                f.write(json.dumps(entry) + '\n')
        
        print(f"  Saved {len(file_data['correct_answers_data'])} entries to {output_filename}")
    
    # Also save a summary file
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {
        "judge_model": next(iter(all_results.values()))["judge_model"] if all_results else None,
        "use_continuous": next(iter(all_results.values()))["use_continuous"] if all_results else None,
        "threshold": next(iter(all_results.values()))["threshold"] if all_results else None,
        "files_processed": len(all_results),
        "file_summaries": {}
    }
    
    for filename, file_data in all_results.items():
        summary["file_summaries"][filename] = {
            "model_name": file_data["model_name"],
            "context_type": file_data["context_type"],
            "num_generations": file_data["num_generations"],
            "num_questions": len(file_data["correct_answers_data"]),
            "total_correct_answers": sum(entry["num_correct"] for entry in file_data["correct_answers_data"])
        }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved summary to summary.json")

def main():
    args = parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    print(f"Processing directory: {args.input_dir}")
    print(f"Judge model: {args.judge}")
    print(f"Using {'continuous' if args.continuous else 'binary'} scores")
    print(f"Threshold: {args.threshold}")
    
    # Process all files
    all_results = process_all_files(
        args.input_dir, 
        args.judge, 
        args.continuous, 
        args.threshold
    )
    
    if not all_results:
        print("No valid results found")
        return
    
    # Create output directory
    output_dir = os.path.join(args.input_dir, "answers_marked_correct")
    
    # Save results
    save_results_to_jsonl(all_results, output_dir)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Judge model: {args.judge}")
    print(f"Files processed: {len(all_results)}")
    
    total_questions = 0
    total_correct_answers = 0
    
    for filename, file_data in all_results.items():
        num_questions = len(file_data["correct_answers_data"])
        num_correct = sum(entry["num_correct"] for entry in file_data["correct_answers_data"])
        total_questions += num_questions
        total_correct_answers += num_correct
        
        print(f"  {file_data['model_name']} ({file_data['context_type']}): "
              f"{num_questions} questions, {num_correct} correct answers")
    
    print(f"\nTotal: {total_questions} questions, {total_correct_answers} correct answers")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 