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
    parser = argparse.ArgumentParser(description="Calculate and plot Brier scores for dataset evaluation files")
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/gpqa/gpqa_diamond/",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Output directory for plots")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Dataset name for plot title (auto-detected if not provided)")
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
    
    Expected format: ModelName_split_size_N_generations_M.jsonl
    Returns: (model_name, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract model name (everything before _train or _test)
    model_match = re.match(r'([^_]+(?:-[^_]*)*?)(?:_(?:train|test))', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        # Fallback: take everything before the first underscore followed by a known pattern
        parts = name_without_ext.split('_')
        model_name = parts[0]
        # Include additional parts that look like model components
        for i, part in enumerate(parts[1:], 1):
            if part in ['train', 'test', 'size'] or part.isdigit():
                break
            model_name += f"_{part}"
    
    # Extract number of generations
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    return model_name, num_generations

def calculate_brier_score(probability: float, is_correct: bool) -> float:
    """
    Calculate Brier score for GPQA-style evaluation.
    
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
        # If answer is incorrect: -p^2
        return -(probability ** 2)

def calculate_generation_brier_scores(data: List[Dict[str, Any]], generation_idx: int) -> List[float]:
    """
    Calculate Brier scores for all questions in a specific generation for GPQA-style data.
    
    Args:
        data: List of evaluation entries
        generation_idx: Index of the generation to evaluate
        
    Returns:
        List of Brier scores for each question in this generation
    """
    brier_scores = []
    
    for item in data:
        # Skip items without necessary fields
        if "extracted_answer" not in item or "answer" not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        correct_answer = item.get("answer", "")
        
        # Skip if generation_idx is out of bounds
        if generation_idx >= len(extracted_answers):
            continue
            
        generation_answer = extracted_answers[generation_idx]
        
        # Handle dictionary format (answer -> probability)
        if isinstance(generation_answer, dict):
            brier_score = 0
            any_answer_given = False
            any_correct = False
            
            for answer_option, probability in generation_answer.items():
                if answer_option and probability is not None:
                    if not isinstance(probability, float) or probability < 0 or probability > 1:
                        continue
                    any_answer_given = True
                    is_correct = (answer_option == correct_answer)
                    if is_correct:
                        any_correct = True
                    brier_score += calculate_brier_score(probability, is_correct)
            
            # If no valid answer was given, assign worst possible score
            # if not any_answer_given:
            #     brier_score = -2.0
            if not any_correct:
                brier_score -= 1 # Penalize for not having any correct answer so its probability is taken as 0
            
            brier_scores.append(brier_score)
        
        # Handle string format (fallback for older formats)
        elif isinstance(generation_answer, str):
            # For string format, assume probability of 1.0 for the given answer
            is_correct = (generation_answer == correct_answer)
            if is_correct:
                brier_score = 0.0  # Perfect score: -(1-1)^2 = 0
            else:
                brier_score = -2.0  # Wrong with certainty: -(1)^2 = -1
            brier_scores.append(brier_score)
    
    return brier_scores

def calculate_model_brier_statistics(data: List[Dict[str, Any]], num_generations: int) -> Tuple[float, float]:
    """
    Calculate mean Brier score and standard error across all generations for a model.
    
    Args:
        data: List of evaluation entries
        num_generations: Number of generations to evaluate
        
    Returns:
        Tuple of (mean_brier_score, standard_error)
    """
    all_generation_means = []
    
    for gen_idx in range(num_generations):
        generation_brier_scores = calculate_generation_brier_scores(data, gen_idx)
        
        if generation_brier_scores:
            # Shift scores to positive range: add 1 to make range [-, 1]
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

def calculate_generation_accuracy(data: List[Dict[str, Any]], generation_idx: int) -> float:
    """
    Calculate accuracy for all questions in a specific generation.
    
    Args:
        data: List of evaluation entries
        generation_idx: Index of the generation to evaluate
        
    Returns:
        Accuracy (fraction of correct answers) for this generation
    """
    correct_count = 0
    total_count = 0
    
    for item in data:
        # Skip items without necessary fields
        if "extracted_answer" not in item or "answer" not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        correct_answer = item.get("answer", "")
        
        # Skip if generation_idx is out of bounds
        if generation_idx >= len(extracted_answers):
            continue
            
        generation_answer = extracted_answers[generation_idx]
        
        # Handle dictionary format (answer -> probability)
        if isinstance(generation_answer, dict):
            # Check if any answer matches the correct answer
            any_correct = any(answer_option == correct_answer for answer_option in generation_answer.keys() if answer_option)
            if any_correct:
                correct_count += 1
            total_count += 1
        
        # Handle string format
        elif isinstance(generation_answer, str):
            is_correct = (generation_answer == correct_answer)
            if is_correct:
                correct_count += 1
            total_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0

def calculate_model_accuracy_statistics(data: List[Dict[str, Any]], num_generations: int) -> Tuple[float, float]:
    """
    Calculate mean accuracy and standard error across all generations for a model.
    
    Args:
        data: List of evaluation entries
        num_generations: Number of generations to evaluate
        
    Returns:
        Tuple of (mean_accuracy, standard_error)
    """
    all_generation_accuracies = []
    
    for gen_idx in range(num_generations):
        generation_accuracy = calculate_generation_accuracy(data, gen_idx) * 100.0
        all_generation_accuracies.append(generation_accuracy)
    
    if not all_generation_accuracies:
        return 0.0, 0.0
    
    # Calculate mean and standard error across generations
    mean_accuracy = np.mean(all_generation_accuracies)
    std_error = np.std(all_generation_accuracies, ddof=1) / np.sqrt(len(all_generation_accuracies)) if len(all_generation_accuracies) > 1 else 0.0
    
    return mean_accuracy, std_error

def get_model_data(input_dir: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Get Brier score and accuracy data for all models in the input directory."""
    brier_model_data = {}
    accuracy_model_data = {}
    
    # Get all JSONL files in the directory (non-recursively)
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")]
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract model info from filename
        model_name, num_generations = extract_model_info_from_filename(filename)
        
        # # Clean up model name
        # if "checkpoint" in model_name:
        #     # Extract checkpoint number and create cleaner name
        #     checkpoint_match = re.search(r'checkpoint(\d+)', model_name)
        #     if checkpoint_match:
        #         checkpoint_num = checkpoint_match.group(1)
        #         # Simplify the model name
        #         base_name = model_name.split('-')[0]  # Get base model name
        #         model_name = f"{base_name}-RL-CP{checkpoint_num}"
        
        # Create a unique key for model
        model_key = f"{model_name}"
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name}")
        
        # Calculate Brier statistics
        mean_brier, brier_std_error = calculate_model_brier_statistics(data, num_generations)
        
        # Calculate accuracy statistics
        mean_accuracy, accuracy_std_error = calculate_model_accuracy_statistics(data, num_generations)
        
        brier_model_data[model_key] = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'std_error': brier_std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        accuracy_model_data[model_key] = {
            'model_name': model_name,
            'mean_accuracy': mean_accuracy,
            'std_error': accuracy_std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        print(f"  {model_name}: Mean Brier = {mean_brier:.4f} ± {brier_std_error:.4f}")
        print(f"  {model_name}: Mean Accuracy = {mean_accuracy:.2f}% ± {accuracy_std_error:.2f}%")
    
    return brier_model_data, accuracy_model_data

def extract_dataset_name(input_dir):
    """Extract and format dataset name from input directory path."""
    # Extract the last part of the path (e.g., gpqa_diamond, mmlu_pro)
    dataset_part = os.path.basename(input_dir.rstrip('/'))
    
    # Format it nicely
    if dataset_part == 'gpqa_diamond':
        return "GPQA Diamond"
    elif dataset_part == 'gpqa_main':
        return "GPQA Main"
    elif dataset_part == 'mmlu_pro':
        return "MMLU-Pro"
    elif dataset_part.startswith('gpqa_'):
        return f"GPQA {dataset_part.split('_')[1].title()}"
    else:
        return dataset_part.replace('_', ' ').title()

def plot_brier_scores(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
    """Plot Brier scores as a bar chart with error bars."""
    
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
            scores.append(model_data[model_name]['mean_brier'])
            errors.append(model_data[model_name]['std_error'])
        else:
            scores.append(0)
            errors.append(0)
        
        # Assign color based on whether it's a checkpoint model
        if "checkpoint" in model_name.lower() or "rl" in model_name.lower() or "cp" in model_name.lower():
            bar_colors.append(checkpoint_color)
        else:
            bar_colors.append(default_color)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Create bars with custom colors
    bars = plt.bar(x_positions, scores, 
                   yerr=errors, capsize=5, 
                   alpha=0.8, color=bar_colors)
    
    # Customize the plot
    # plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Brier Score', fontsize=28, fontweight='bold')
    
    # # Set title
    # if dataset_name:
    #     plt.title(f'{dataset_name} - Brier Score Performance', fontsize=28, fontweight='bold', pad=30)
    # else:
    #     plt.title(f'Brier Score Performance', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in all_model_names], 
               rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=24)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    add_value_labels(bars, scores, errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Brier score plot saved to {output_path}")
    plt.close()

def plot_accuracy(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
    """Plot accuracy as a bar chart with error bars."""
    
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
        
        # Assign color based on whether it's a checkpoint model
        if "checkpoint" in model_name.lower() or "rl" in model_name.lower() or "cp" in model_name.lower():
            bar_colors.append(checkpoint_color)
        else:
            bar_colors.append(default_color)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Create bars with custom colors
    bars = plt.bar(x_positions, scores, 
                   yerr=errors, capsize=5, 
                   alpha=0.8, color=bar_colors)
    
    # Customize the plot
    plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Accuracy (%)', fontsize=24, fontweight='bold')
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Accuracy Performance', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Accuracy Performance', fontsize=28, fontweight='bold', pad=30)
    
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
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
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
    
    # Extract dataset name from input directory or use provided name
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = extract_dataset_name(args.input_dir)
    
    print(f"Processing directory: {args.input_dir}")
    print(f"Dataset: {dataset_name}")
    
    # Get data for all models
    print(f"\nCalculating scores for all models in {args.input_dir}")
    brier_model_data, accuracy_model_data = get_model_data(args.input_dir)
    
    if not brier_model_data and not accuracy_model_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    
    if brier_model_data:
        brier_output_filename = f"brier_scores_{dataset_suffix}.png"
        brier_output_path = os.path.join(args.output_dir, brier_output_filename)
        
        # Plot the Brier scores
        plot_brier_scores(brier_model_data, brier_output_path, dataset_name)
    
    if accuracy_model_data:
        accuracy_output_filename = f"accuracy_{dataset_suffix}.png"
        accuracy_output_path = os.path.join(args.output_dir, accuracy_output_filename)
        
        # Plot the accuracy
        plot_accuracy(accuracy_model_data, accuracy_output_path, dataset_name)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Dataset: {dataset_name}")
    
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
            accuracy_info = f"Accuracy = {accuracy_data['mean_accuracy']:.2f}% ± {accuracy_data['std_error']:.2f}%"
            if not samples_info:
                samples_info = f"{accuracy_data['num_samples']} samples"
        
        # Combine the information
        metrics = [info for info in [brier_info, accuracy_info] if info]
        metrics_str = ", ".join(metrics)
        
        print(f"  {model_key}: {samples_info}, {metrics_str}")

if __name__ == "__main__":
    main()
