#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
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
    'deepseek-r1-0528': 'DeepSeek R1 0528',
    'deepseek-r1': 'DeepSeek R1',
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

model_sizes = {
    'Qwen3-4B': 4,
    'qwen3-4b': 4,
    'Qwen3-8B': 8,
    'qwen3-8b': 8,
    'Qwen3-1.7B': 1.7,
    'qwen3-1.7b': 1.7,
    'Qwen3-32B': 32,
    'qwen3-32b': 32,
    'DeepSeek V3': 671,
    'deepseek-chat-v3-0324': 671,
    'DeepSeek R1': 671,
    'deepseek-r1-0528': 671,
    'deepseek-r1': 671,
    'llama-3.3-70b-instruct': 70,
    'Llama 3.3 70B': 70,
    'Llama 4 Maverick': 400,
    'llama-4-maverick': 400,
    'gpt-oss-120b': 120,
    'grok-3-mini': 1000
}

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Brier scores and accuracy between two evaluation directories")
    parser.add_argument("--primary", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/theguardian_207",
                       help="Primary directory containing evaluation JSONL files")
    parser.add_argument("--secondary", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/validation-retrieval_207/forcomparison/",
                       help="Secondary directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/comparison",
                       help="Output directory for comparison plots")
    parser.add_argument("--judge", type=str, default="Qwen3_4B",
                       help="Judge model name for score field")
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

def get_model_data_from_dir(input_dir: str, judge: str) -> Dict[str, Dict[str, Any]]:
    """Get both Brier score and accuracy data for all models in the input directory."""
    model_data = {}
    
    # Get all JSONL files in the directory (non-recursively)
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")]
    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract model info from filename
        model_name, num_generations = extract_model_info_from_filename(filename)
        
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
        
        # Calculate both Brier and accuracy statistics
        mean_brier, brier_std_error = calculate_model_brier_statistics(data, num_generations, judge_field)
        mean_accuracy, accuracy_std_error = calculate_model_accuracy_statistics(data, num_generations, judge_field)
        
        model_data[model_key] = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'brier_std_error': brier_std_error,
            'mean_accuracy': mean_accuracy,
            'accuracy_std_error': accuracy_std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        print(f"  {model_name}: Brier = {mean_brier:.4f} ± {brier_std_error:.4f}, Accuracy = {mean_accuracy:.4f} ± {accuracy_std_error:.4f}")
    
    return model_data

def get_secondary_label(primary_path: str, secondary_path: str) -> str:
    """Generate a label for the secondary dataset by removing the primary part."""
    primary_name = os.path.basename(primary_path.rstrip('/'))
    secondary_name = os.path.basename(secondary_path.rstrip('/'))
    return "With Retrieval"
    # Remove the primary name from secondary name
    if primary_name in secondary_name:
        label = secondary_name.replace(primary_name, '').strip('-_')
        if not label:
            label = "secondary"
        return label
    else:
        return secondary_name

def plot_comparison_brier(primary_data: Dict[str, Dict[str, Any]], 
                         secondary_data: Dict[str, Dict[str, Any]], 
                         secondary_label: str,
                         judge: str, 
                         output_path: str, 
                         dataset_name: str = None):
    """Plot comparison of Brier scores between two datasets."""
    
    # Get models that exist in both datasets
    common_models = sorted(set(primary_data.keys()) & set(secondary_data.keys()))
    
    if not common_models:
        print("No common models found between the two datasets for Brier comparison")
        return
    
    # Prepare data for plotting
    x_positions = np.arange(len(common_models))
    width = 0.35  # Width of bars
    
    primary_scores = []
    primary_errors = []
    secondary_scores = []
    secondary_errors = []
    lowest = 0
    highest = 0
    
    for model_name in common_models:
        primary_scores.append(primary_data[model_name]['mean_brier'])
        primary_errors.append(primary_data[model_name]['brier_std_error'])
        secondary_scores.append(secondary_data[model_name]['mean_brier'])
        secondary_errors.append(secondary_data[model_name]['brier_std_error'])
        lowest = min(lowest, primary_scores[-1] - primary_errors[-1], secondary_scores[-1] - secondary_errors[-1])
        highest = max(highest, primary_scores[-1] + primary_errors[-1], secondary_scores[-1] + secondary_errors[-1])
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Adjust the y-axis to start at the lowest score
    plt.ylim(lowest - 0.05, highest + 0.05)
    
    # Create bars
    bars1 = plt.bar(x_positions - width/2, primary_scores, width, 
                   yerr=primary_errors, capsize=5, 
                   alpha=0.8, color='#2ca02c', label='Without Retrieval')
    bars2 = plt.bar(x_positions + width/2, secondary_scores, width,
                   yerr=secondary_errors, capsize=5, 
                   alpha=0.8, color='#ff7f0e', label=f'{secondary_label}')
    
    # Customize the plot
    # plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Brier Score', fontsize=28, fontweight='bold')
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Brier Score Comparison', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Brier Score Comparison - {judge} Judge', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in common_models], 
               rotation=45, ha='right', fontsize=26)
    plt.yticks(fontsize=22)
    
    # Add legend
    plt.legend(fontsize=20, loc='best')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value != 0:  # Only label non-zero bars
                height = bar.get_height()
                if height >= 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.005,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=28, fontweight='bold')
                else:
                    plt.text(bar.get_x() + bar.get_width()/2., height - error - 0.02,
                            f'{value:.3f}', ha='center', va='top', fontsize=28, fontweight='bold')
    
    add_value_labels(bars1, primary_scores, primary_errors)
    add_value_labels(bars2, secondary_scores, secondary_errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Brier score comparison plot saved to {output_path}")
    plt.close()

def plot_comparison_accuracy(primary_data: Dict[str, Dict[str, Any]], 
                            secondary_data: Dict[str, Dict[str, Any]], 
                            secondary_label: str,
                            judge: str, 
                            output_path: str, 
                            dataset_name: str = None):
    """Plot comparison of accuracy between two datasets."""
    
    # Get models that exist in both datasets
    common_models = sorted(set(primary_data.keys()) & set(secondary_data.keys()))
    
    if not common_models:
        print("No common models found between the two datasets for accuracy comparison")
        return
    
    # Prepare data for plotting
    x_positions = np.arange(len(common_models))
    width = 0.35  # Width of bars
    
    primary_scores = []
    primary_errors = []
    secondary_scores = []
    secondary_errors = []
    
    # sort by model size
    common_models = sorted(common_models, key=lambda x: model_sizes[x])
    
    for model_name in common_models:
        primary_scores.append(primary_data[model_name]['mean_accuracy'])
        primary_errors.append(primary_data[model_name]['accuracy_std_error'])
        secondary_scores.append(secondary_data[model_name]['mean_accuracy'])
        secondary_errors.append(secondary_data[model_name]['accuracy_std_error'])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create bars
    bars1 = plt.bar(x_positions - width/2, primary_scores, width, 
                   yerr=primary_errors, capsize=5, 
                   alpha=0.8, color='#2ca02c', label='Without Retrieval')
    bars2 = plt.bar(x_positions + width/2, secondary_scores, width,
                   yerr=secondary_errors, capsize=5, 
                   alpha=0.8, color='#ff7f0e', label=f'{secondary_label}')
    
    fs = 22
    # grok_acc_search = 187.0 / 207 
    # plot this as a dashed horizontal line
    # plt.axhline(y=grok_acc_search * 100, color='black', linestyle='dashed', linewidth=2)
    # add text above the horizontal line
    # plt.text(3, grok_acc_search * 100 + 1, 'grok-4.1-fast with web-search: 90\%', fontsize=fs, fontweight='bold', ha='center', va='bottom')
    # Customize the plot
    # plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Accuracy (\%)', fontsize=26, fontweight='bold', labelpad=4)
    
    # # Set title
    # if dataset_name:
    #     plt.title(f'{dataset_name} - Accuracy Comparison', fontsize=28, fontweight='bold', pad=30)
    # else:
    #     plt.title(f'Accuracy Comparison - {judge} Judge', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in common_models], 
               rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=24)
    
    # Set y-axis limits (accuracy is between 0 and 1)
    plt.ylim(0, max(max(primary_scores), max(secondary_scores)) * 1.2)
    # plt.ylim(0, 100)
    
    # Add legend, shifted down by 20 points in y-axis
    # plt.legend(fontsize=20, loc='best', ncols=1, bbox_to_anchor=(0.67, 0.68), bbox_transform=plt.gcf().transFigure)
    plt.legend(fontsize=18, loc='best', ncols=1)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2 - 0.05, height + error + 0.5,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=24, fontweight='bold')
    
    add_value_labels(bars1, primary_scores, primary_errors)
    add_value_labels(bars2, secondary_scores, secondary_errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison plot saved to {output_path}")
    # also save the plot as a pdf
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
    plt.close()

def extract_dataset_name(input_dir):
    """Extract and format dataset name from input directory path."""
    # Extract the last part of the path (e.g., theguardian_207, dw_21317)
    dataset_part = os.path.basename(input_dir.rstrip('/'))
    
    parts = input_dir.split("/")
    
    for dataset_part in parts:
        # Format it nicely
        if dataset_part.startswith('theguardian'):
            suffix = dataset_part.split("_")[1]
            number_match = re.search(r'(\d+)', suffix)
            number = number_match.group(1) if number_match else "Unknown"
            return f"The Guardian - July 2025 - ({number} Questions)"
        elif dataset_part.startswith('dw_'):
            return f"DW 2024-25 ({dataset_part.split('_')[1]} Questions)"
    
    return f"{dataset_part} Forecasting"

def main():
    args = parse_args()
    
    # Check if input directories exist
    if not os.path.exists(args.primary):
        print(f"Error: Primary directory {args.primary} does not exist")
        return
    
    if not os.path.exists(args.secondary):
        print(f"Error: Secondary directory {args.secondary} does not exist")
        return
    
    # Extract dataset name from primary directory
    dataset_name = extract_dataset_name(args.primary)
    secondary_label = get_secondary_label(args.primary, args.secondary)
    
    print(f"Primary directory: {args.primary}")
    print(f"Secondary directory: {args.secondary}")
    print(f"Dataset: {dataset_name}")
    print(f"Secondary label: {secondary_label}")
    print(f"Judge: {args.judge}")
    
    # Get data from both directories
    print(f"\nProcessing primary directory: {args.primary}")
    primary_data = get_model_data_from_dir(args.primary, args.judge)
    
    print(f"\nProcessing secondary directory: {args.secondary}")
    secondary_data = get_model_data_from_dir(args.secondary, args.judge)
    
    if not primary_data or not secondary_data:
        print("No valid model data found in one or both directories")
        return
    
    # Find common models
    common_models = set(primary_data.keys()) & set(secondary_data.keys())
    print(f"\nFound {len(common_models)} common models: {sorted(common_models)}")
    
    if not common_models:
        print("No common models found between the two directories")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    primary_suffix = os.path.basename(args.primary.rstrip('/'))
    secondary_suffix = os.path.basename(args.secondary.rstrip('/'))
    
    # Plot Brier score comparison
    brier_output_filename = f"brier_comparison_{primary_suffix}_vs_{secondary_suffix}_{args.judge}.png"
    brier_output_path = os.path.join(args.output_dir, brier_output_filename)
    plot_comparison_brier(primary_data, secondary_data, secondary_label, args.judge, brier_output_path, dataset_name)
    
    # Plot accuracy comparison
    accuracy_output_filename = f"accuracy_comparison_{primary_suffix}_vs_{secondary_suffix}_{args.judge}.png"
    accuracy_output_path = os.path.join(args.output_dir, accuracy_output_filename)
    plot_comparison_accuracy(primary_data, secondary_data, secondary_label, args.judge, accuracy_output_path, dataset_name)
    
    # Print summary statistics
    print(f"\nComparison Summary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Primary: {args.primary}")
    print(f"Secondary: {args.secondary} ({secondary_label})")
    print(f"Judge: {args.judge}")
    print(f"Common models: {len(common_models)}")
    
    print(f"\nDetailed Results:")
    for model_key in sorted(common_models):
        primary_info = primary_data[model_key]
        secondary_info = secondary_data[model_key]
        
        print(f"  {model_key}:")
        print(f"    Primary  - Brier: {primary_info['mean_brier']:.4f} ± {primary_info['brier_std_error']:.4f}, "
              f"Accuracy: {primary_info['mean_accuracy']:.2f}% ± {primary_info['accuracy_std_error']:.2f}%")
        print(f"    {secondary_label:8} - Brier: {secondary_info['mean_brier']:.4f} ± {secondary_info['brier_std_error']:.4f}, "
              f"Accuracy: {secondary_info['mean_accuracy']:.2f}% ± {secondary_info['accuracy_std_error']:.2f}%")
        
        # Calculate differences
        brier_diff = secondary_info['mean_brier'] - primary_info['mean_brier']
        accuracy_diff = secondary_info['mean_accuracy'] - primary_info['mean_accuracy']
        print(f"    Difference - Brier: {brier_diff:+.4f}, Accuracy: {accuracy_diff:+.2f}%")

if __name__ == "__main__":
    main()
