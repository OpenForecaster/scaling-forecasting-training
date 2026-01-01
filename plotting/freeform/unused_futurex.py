#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import glob
from collections import defaultdict
import re
from typing import List, Dict, Any, Tuple

mpl.style.use(['science'])

# Custom display names for models
print_names = {
    'qwen3-4b': 'Qwen3-4B',
    'Qwen3-4B': 'Qwen3-4B',
    'qwen3-8b': 'Qwen3-8B',
    'Qwen3-8B': 'Qwen3-8B',
    'qwen3-1.7b': 'Qwen3-1.7B',
    'Qwen3-1.7B': 'Qwen3-1.7B',
    'qwen3-32b': 'Qwen3-32B',
    'Qwen3-32B': 'Qwen3-32B',
    'deepseek-chat-v3-0324': 'DeepSeek V3',
    'DeepSeek-V3-0324': 'DeepSeek V3',
    'deepseek-r1-0528': 'DeepSeek R1',
    'llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'llama-4-maverick': 'Llama 4 Maverick',
    'llama-4-scout': 'Llama 4 Scout',
    'claude-3.5-haiku': 'Claude 3.5 Haiku',
    'gpt-4o': 'GPT 4o',
    'gpt-4o-mini': 'GPT 4o Mini',
    'o4-mini-high': 'o4 Mini High',
    'grok-3-mini-beta': 'Grok 3 Mini',
    'grok-4': 'Grok 4',
    'kimi-k2': 'Kimi K2',
    'gemini-2.5-pro-preview': 'Gemini 2.5 Pro',
    'gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and plot Brier scores for freeform forecasting evaluation")
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/theguardian_207/",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Output directory for plots")
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
    Extract model name and number of generations from filename.
    
    Expected format: ModelName_eval_size_N_generations_M.jsonl
    Returns: (model_name, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    multiple = "_list" if "multiple" in filename else ""
    
    # Extract model name (everything before _eval)
    model_match = re.match(r'([^_]+(?:_[^_]*?)?(?:-\d+\.?\d*[bB])?)', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = name_without_ext.split('_')[0]
    
    # Extract number of generations
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    if model_name.endswith('_'):
        model_name = model_name[:-1]
        
    model_name = f"{model_name}{multiple}"
    
    return model_name, num_generations

def calculate_brier_score(probability: float, is_correct: bool) -> float:
    """
    Calculate Brier score using the formula from eval_freeform.py.
    
    Args:
        probability: Probability assigned to the answer (0-1)
        is_correct: Whether the answer was correct
        
    Returns:
        Brier score (range: [-2, 0])
    """
    if is_correct:
        # If answer is correct: -(1 - p)^2
        return -((1 - probability) ** 2)
    else:
        # If answer is incorrect: -(1 + p^2)
        # return -(1 + probability ** 2)
        return - (probability ** 2)

def calculate_generation_brier_scores(data: List[Dict[str, Any]], generation_idx: int, judge_field: str) -> List[float]:
    """
    Calculate Brier scores for all questions in a specific generation.
    
    Args:
        data: List of evaluation entries
        generation_idx: Index of the generation to evaluate
        judge_field: Field name for judge scores
        
    Returns:
        List of Brier scores for each question in this generation
    """
    brier_scores = []
    
    for item in data:
        # Skip items without necessary fields
        if "extracted_answer" not in item or judge_field not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        judge_scores = item.get(judge_field, [])
        
        # Skip if generation_idx is out of bounds
        if generation_idx >= len(extracted_answers) or generation_idx >= len(judge_scores):
            continue
            
        generation_answer = extracted_answers[generation_idx]
        generation_scores = judge_scores[generation_idx]
        
        # Handle dictionary format (new probabilistic format)
        if isinstance(generation_answer, dict) and isinstance(generation_scores, dict):
            # For each answer option in this generation
            any_correct = False
            brier_score = 0
            for answer_option, probability in generation_answer.items():
                if not answer_option or not probability:
                    continue 
                
                if probability == None:
                    print(f"Probability is None for {answer_option}, {generation_answer}")
                    
                if answer_option in generation_scores:
                    is_correct = (int(generation_scores[answer_option]) == 1)
                    if is_correct:
                        any_correct = True
                    brier_score += calculate_brier_score(probability, is_correct)
                    
            if not any_correct:
                brier_score -= 1 # Penalize for not having any correct answer so its probability is taken as 0
            
            # brier_score *= -1 
            brier_scores.append(brier_score)
        
        # Handle string format (old format) - less common for probabilistic forecasting
        elif isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
            # For string format, assume probability of 1.0 for the given answer
            is_correct = (int(generation_scores) == 1)
            if not is_correct:
                brier_score = -2 # Penalize for not having any correct answer so its probability is taken as 0
            else :
                brier_score = 0 
            brier_scores.append(brier_score)
    
    print(f"Generation {generation_idx} Brier scores: {brier_scores}")
    return brier_scores

def calculate_model_brier_statistics(data: List[Dict[str, Any]], num_generations: int, judge_field: str) -> Tuple[float, float]:
    """
    Calculate mean Brier score and standard error across all generations for a model.
    
    Args:
        data: List of evaluation entries
        num_generations: Number of generations to evaluate
        judge_field: Field name for judge scores
        
    Returns:
        Tuple of (mean_brier_score, standard_error)
    """
    all_generation_means = []
    
    for gen_idx in range(num_generations):
        generation_brier_scores = calculate_generation_brier_scores(data, gen_idx, judge_field)
        
        if generation_brier_scores:
            # increase each score by 1
            generation_brier_scores = [score + 1 for score in generation_brier_scores]
            # Calculate mean Brier score for this generation across all questions
            generation_mean = np.mean(generation_brier_scores)
            all_generation_means.append(generation_mean)
    
    if not all_generation_means:
        return 0.0, 0.0
    
    # Calculate mean and standard error across generations
    mean_brier = np.mean(all_generation_means)
    std_error = np.std(all_generation_means, ddof=1) / np.sqrt(len(all_generation_means)) if len(all_generation_means) > 1 else 0.0
    
    return mean_brier, std_error

def get_model_brier_data(input_dir: str, judge: str) -> Dict[str, Dict[str, Any]]:
    """Get Brier score data for all models in the input directory."""
    model_data = {}
    
    # Get all JSONL files in the directory (non-recursively)
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")]
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract model info from filename
        model_name, num_generations = extract_model_info_from_filename(filename)
        
        # if "2048" in model_name:
        #     before, after = model_name.split("-2048-")
        #     after = after.split("-")[-1]
        #     model_name = f"{before}-{after}"
        
        if "withbinary" in model_name:
            parts = model_name.split("-")
            model_name = "-".join([part for part in parts if "with" not in part and "binary" not in part])
        
        
        # Create a unique key for model+context combination
        model_key = f"{model_name}"
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name}")
        
        # Check if judge field exists
        judge_field = f"score_{judge}"
        available_fields = set()
        has_judge_field = False
        
        for item in data:
            for key in item.keys():
                if key.startswith("score_"):
                    available_fields.add(key)
                    if key == judge_field:
                        has_judge_field = True
        
        if not has_judge_field:
            print(f"  Warning: {judge_field} not found, available fields: {available_fields}")
            continue
        
        # Calculate Brier statistics
        mean_brier, std_error = calculate_model_brier_statistics(data, num_generations, judge_field)
        
        model_data[model_key] = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'std_error': std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        print(f"  {model_name}: Mean Brier = {mean_brier:.4f} ± {std_error:.4f}")
    
    return model_data

def calculate_generation_accuracy(data: List[Dict[str, Any]], generation_idx: int, judge_field: str) -> float:
    """
    Calculate accuracy for all questions in a specific generation.
    
    Args:
        data: List of evaluation entries
        generation_idx: Index of the generation to evaluate
        judge_field: Field name for judge scores
        
    Returns:
        Accuracy (fraction of correct answers) for this generation
    """
    correct_count = 0
    total_count = 0
    
    for item in data:
        # Skip items without necessary fields
        if "extracted_answer" not in item or judge_field not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        judge_scores = item.get(judge_field, [])
        
        # Skip if generation_idx is out of bounds
        if generation_idx >= len(extracted_answers) or generation_idx >= len(judge_scores):
            continue
            
        generation_answer = extracted_answers[generation_idx]
        generation_scores = judge_scores[generation_idx]
        
        # Handle dictionary format (new probabilistic format)
        if isinstance(generation_answer, dict) and isinstance(generation_scores, dict):
            # For each answer option in this generation, check if any is correct
            any_correct = False
            for answer_option in generation_answer.keys():
                if answer_option in generation_scores:
                    is_correct = (int(generation_scores[answer_option]) == 1)
                    if is_correct:
                        any_correct = True
                        break
            
            if any_correct:
                correct_count += 1
            total_count += 1
        
        # Handle string format (old format)
        elif isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
            is_correct = (int(generation_scores) == 1)
            if is_correct:
                correct_count += 1
            total_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0

def calculate_model_accuracy_statistics(data: List[Dict[str, Any]], num_generations: int, judge_field: str) -> Tuple[float, float]:
    """
    Calculate mean accuracy and standard error across all generations for a model.
    
    Args:
        data: List of evaluation entries
        num_generations: Number of generations to evaluate
        judge_field: Field name for judge scores
        
    Returns:
        Tuple of (mean_accuracy, standard_error)
    """
    all_generation_accuracies = []
    
    for gen_idx in range(num_generations):
        generation_accuracy = calculate_generation_accuracy(data, gen_idx, judge_field) * 100.0
        all_generation_accuracies.append(generation_accuracy)
    
    if not all_generation_accuracies:
        return 0.0, 0.0
    
    # Calculate mean and standard error across generations
    mean_accuracy = np.mean(all_generation_accuracies)
    std_error = np.std(all_generation_accuracies, ddof=1) / np.sqrt(len(all_generation_accuracies)) if len(all_generation_accuracies) > 1 else 0.0
    
    return mean_accuracy, std_error

def get_model_accuracy_data(input_dir: str, judge: str) -> Dict[str, Dict[str, Any]]:
    """Get accuracy data for all models in the input directory."""
    model_data = {}
    
    # Get all JSONL files in the directory (non-recursively)
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")]
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract model info from filename
        model_name, num_generations = extract_model_info_from_filename(filename)
        
        # if "2048" in model_name:
        #     before, after = model_name.split("-2048-")
        #     after = after[after.find("-")+1:]
        #     after = after.split("-")[-1]
        #     model_name = f"{before}-{after}"
        
        if "withbinary" in model_name:
            parts = model_name.split("-")
            model_name = "-".join([part for part in parts if "with" not in part and "binary" not in part])
        
        # Create a unique key for model
        model_key = f"{model_name}"
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name}")
        
        # Check if judge field exists
        judge_field = f"score_{judge}"
        available_fields = set()
        has_judge_field = False
        
        for item in data:
            for key in item.keys():
                if key.startswith("score_"):
                    available_fields.add(key)
                    if key == judge_field:
                        has_judge_field = True
        
        if not has_judge_field:
            print(f"  Warning: {judge_field} not found, available fields: {available_fields}")
            continue
        
        # Calculate accuracy statistics
        mean_accuracy, std_error = calculate_model_accuracy_statistics(data, num_generations, judge_field)
        
        model_data[model_key] = {
            'model_name': model_name,
            'mean_accuracy': mean_accuracy,
            'std_error': std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        print(f"  {model_name}: Mean Accuracy = {mean_accuracy:.4f} ± {std_error:.4f}")
    
    return model_data

def extract_dataset_name(input_dir):
    """Extract and format dataset name from input directory path."""
    # Extract the last part of the path (e.g., theguardian_207, dw_21317)
    dataset_part = os.path.basename(input_dir.rstrip('/'))
    
    # Format it nicely
    if dataset_part.startswith('theguardian_'):
        return f"The Guardian - July 2025 - ({dataset_part.split('_')[1]} Questions)"
    elif dataset_part.startswith('dw_'):
        return f"DW 2024-25 ({dataset_part.split('_')[1]} Questions)"
    else:
        return f"{dataset_part} Forecasting"

def plot_brier_scores(model_data: Dict[str, Dict[str, Any]], judge: str, output_path: str, dataset_name: str = None):
    """Plot Brier scores as a bar chart with error bars, using a different color for models with 'checkpoint' in their name."""
    
    # Get all unique model names
    all_model_names = sorted(set(list(model_data.keys())))
    
    if not all_model_names:
        print("No valid model data found for plotting")
        return
    
    # Prepare data for plotting
    x_positions = np.arange(len(all_model_names))
    
    scores = []
    errors = []
    bar_colors = []
    checkpoint_color = '#ff7f0e'  # orange
    # default_color = '#1f77b4'     # blue
    default_color = '#2ca02c'     # green
    lowest = 0 
    highest = 0
    
    for model_name in all_model_names:
        # With article data
        if model_name in model_data:
            scores.append(model_data[model_name]['mean_brier'])
            errors.append(model_data[model_name]['std_error'])
        else:
            scores.append(0)
            errors.append(0)
            
        lowest = min(lowest, scores[-1] - errors[-1])
        highest = max(highest, scores[-1] + errors[-1])
        # Assign color based on whether "checkpoint" is in the model name
        if "checkpoint" in model_name.lower() or "rl" in model_name.lower():
            bar_colors.append(checkpoint_color)
        else:
            bar_colors.append(default_color)
        
        if 'unfiltered' in model_name.lower():
            bar_colors[-1] = default_color # green color
        elif 'filtered' in model_name.lower():
            bar_colors[-1] = 'red' # red color
            
        
        if 'both' in model_name.lower():
            bar_colors[-1] = default_color # green color
        elif 'freeform' in model_name.lower():
            bar_colors[-1] = 'red' # red color
             
            
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Create bars with custom colors
    bars = plt.bar(x_positions, scores, 
                   yerr=errors, capsize=5, 
                   alpha=0.8, color=bar_colors)
    
    # Adjust the y-axis to start at the lowest score
    plt.ylim(lowest - 0.05, highest + 0.05)
    
    # Customize the plot
    plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Brier Score', fontsize=24, fontweight='bold')
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Brier Score', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Brier Score Performance - {judge} Judge', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in all_model_names], 
               rotation=45, ha='right', fontsize=22)
    plt.yticks(fontsize=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')
            elif value < 0:
                height = bar.get_height()
                print(f"Height: {height}, Error: {error}")
                plt.text(bar.get_x() + bar.get_width()/2., height - error - 0.04,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
    add_value_labels(bars, scores, errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Brier score plot saved to {output_path}")
    plt.close()

def plot_accuracy(model_data: Dict[str, Dict[str, Any]], judge: str, output_path: str, dataset_name: str = None):
    """Plot accuracy as a bar chart with error bars, using a different color for models with 'checkpoint' in their name."""
    
    # Get all unique model names
    all_model_names = sorted(set(list(model_data.keys())))
    
    if not all_model_names:
        print("No valid model data found for plotting")
        return
    
    # Prepare data for plotting
    x_positions = np.arange(len(all_model_names))
    
    scores = []
    errors = []
    bar_colors = []
    checkpoint_color = '#ff7f0e'  # orange
    default_color = '#2ca02c'     # green
    
    for model_name in all_model_names:
        if model_name in model_data:
            scores.append(model_data[model_name]['mean_accuracy'])
            errors.append(model_data[model_name]['std_error'])
        else:
            scores.append(0)
            errors.append(0)
        # Assign color based on whether "checkpoint" is in the model name
        if "checkpoint" in model_name.lower() or "rl" in model_name.lower():
            bar_colors.append(checkpoint_color)
        else:
            bar_colors.append(default_color)
    
        if 'unfiltered' in model_name.lower():
            bar_colors[-1] = default_color # green color
        elif 'filtered' in model_name.lower():
            bar_colors[-1] = 'red' # red color
            
            
        
        if 'both' in model_name.lower():
            bar_colors[-1] = default_color # green color
        elif 'freeform' in model_name.lower():
            bar_colors[-1] = 'red' # red color
             
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Create bars with custom colors
    bars = plt.bar(x_positions, scores, 
                   yerr=errors, capsize=5, 
                   alpha=0.8, color=bar_colors)
    
    # Customize the plot
    plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Accuracy', fontsize=24, fontweight='bold')
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Accuracy Performance', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Accuracy Performance - {judge} Judge', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in all_model_names], 
               rotation=45, ha='right', fontsize=22)
    plt.yticks(fontsize=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Set y-axis limits (accuracy is between 0 and 1)
    # plt.ylim(0, 1.1)
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
    add_value_labels(bars, scores, errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to {output_path}")
    plt.close()

def main():
    args = parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Extract dataset name from input directory
    dataset_name = extract_dataset_name(args.input_dir)
    
    print(f"Processing directory: {args.input_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Judge: {args.judge}")
    
    # Get Brier score data for all models
    print(f"\nCalculating Brier scores for all models in {args.input_dir}")
    brier_model_data = get_model_brier_data(args.input_dir, args.judge)
    
    # Get accuracy data for all models
    print(f"\nCalculating accuracy for all models in {args.input_dir}")
    accuracy_model_data = get_model_accuracy_data(args.input_dir, args.judge)
    
    if not brier_model_data and not accuracy_model_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    
    if brier_model_data:
        brier_output_filename = f"brier_scores_{dataset_suffix}_{args.judge}.png"
        brier_output_path = os.path.join(args.output_dir, brier_output_filename)
        
        # Plot the Brier scores
        plot_brier_scores(brier_model_data, args.judge, brier_output_path, dataset_name)
    
    if accuracy_model_data:
        accuracy_output_filename = f"accuracy_{dataset_suffix}_{args.judge}.png"
        accuracy_output_path = os.path.join(args.output_dir, accuracy_output_filename)
        
        # Plot the accuracy
        plot_accuracy(accuracy_model_data, args.judge, accuracy_output_path, dataset_name)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Judge: {args.judge}")
    
    # Combine model data for comprehensive reporting
    all_models = set()
    if brier_model_data:
        all_models.update(brier_model_data.keys())
    if accuracy_model_data:
        all_models.update(accuracy_model_data.keys())
    
    print(f"Total model variants: {len(all_models)}")
    
    for model_key in sorted(all_models):
        brier_info = ""
        accuracy_info = ""
        samples_info = ""
        
        if model_key in brier_model_data:
            brier_data = brier_model_data[model_key]
            brier_info = f"Brier = {brier_data['mean_brier']:.4f} ± {brier_data['std_error']:.4f}"
            samples_info = f"{brier_data['num_samples']} samples"
        
        if model_key in accuracy_model_data:
            accuracy_data = accuracy_model_data[model_key]
            accuracy_info = f"Accuracy = {accuracy_data['mean_accuracy']:.4f} ± {accuracy_data['std_error']:.4f}"
            if not samples_info:
                samples_info = f"{accuracy_data['num_samples']} samples"
        
        # Combine the information
        metrics = [info for info in [brier_info, accuracy_info] if info]
        metrics_str = ", ".join(metrics)
        
        print(f"  {model_key}: {samples_info}, {metrics_str}")

if __name__ == "__main__":
    main()
