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
    parser = argparse.ArgumentParser(description="Plot binary ablation results with accuracy and Brier scores")
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/theguardian_207/binary-ablations",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/binary_ablations",
                       help="Output directory for plots")
    parser.add_argument("--judge", type=str, default="Llama_4_Scout",
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
    Extract model name and configuration from filename.
    
    Expected format: ModelName_configuration_generations_M.jsonl
    Returns: (model_name, configuration, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract model name (everything before the first underscore followed by configuration)
    model_match = re.match(r'([^_]+(?:_[^_]*?)?(?:-\d+\.?\d*[bB])?)', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = name_without_ext.split('_')[0]
    
    # Extract configuration (binary-ablations, onlybinary, onlyfreeform)
    if 'both' in name_without_ext:
        configuration = 'binary-ablations'
    elif 'onlybinary' in name_without_ext:
        configuration = 'onlybinary'
    elif 'freeform' in name_without_ext:
        configuration = 'onlyfreeform'
    else:
        configuration = 'unknown'
    
    # Extract number of generations
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    if model_name.endswith('_'):
        model_name = model_name[:-1]
    
    return model_name, configuration, num_generations

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

def get_model_data(input_dir: str, judge: str) -> Dict[str, Dict[str, Any]]:
    """Get accuracy and Brier score data for all models in the input directory."""
    model_data = {}
    
    # Get all JSONL files in the directory (non-recursively)
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")]
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract model info from filename
        model_name, configuration, num_generations = extract_model_info_from_filename(filename)
        
        # Create a unique key for model+configuration combination
        model_key = f"{model_name}_{configuration}"
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name} ({configuration})")
        
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
        mean_accuracy, accuracy_std_error = calculate_model_accuracy_statistics(data, num_generations, judge_field)
        
        # Calculate Brier statistics
        mean_brier, brier_std_error = calculate_model_brier_statistics(data, num_generations, judge_field)
        
        model_data[model_key] = {
            'model_name': model_name,
            'configuration': configuration,
            'mean_accuracy': mean_accuracy,
            'accuracy_std_error': accuracy_std_error,
            'mean_brier': mean_brier,
            'brier_std_error': brier_std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        print(f"  {model_name} ({configuration}): Accuracy = {mean_accuracy:.4f} ± {accuracy_std_error:.4f}, Brier = {mean_brier:.4f} ± {brier_std_error:.4f}")
    
    return model_data

def format_model_name_for_display(model_name: str, configuration: str) -> str:
    """Format model name for display with configuration information."""
    # Shorten base model names for better readability
    if 'qwen3-8b' in model_name.lower() or 'qwen3_8b' in model_name.lower():
        if 'checkpoint' in model_name.lower():
            base_name = 'Qwen3-8B (RL)'
        else:
            base_name = 'Qwen3-8B\nThinking'
    else:
        base_name = print_names.get(model_name, model_name)
        # Add RL indicator for checkpoints
        if 'checkpoint' in model_name.lower():
            base_name += ' (RL)'
    
    # if configuration == 'binary-ablations':
    #     return f"{base_name}\nBinary + Freeform"
    # elif configuration == 'onlybinary':
    #     return f"{base_name}\nBinary Only"
    # elif configuration == 'onlyfreeform':
    #     return f"{base_name}\nFreeform Only"
    # else:
    #     return f"{base_name}"
    
    
    if configuration == 'binary-ablations':
        return "\\textbf{+} Binary \& Freeform\n(10K samples each)"
    elif configuration == 'onlybinary':
        return "\\textbf{+} Only Binary\n(20K samples)"
    elif configuration == 'onlyfreeform':
        return "\\textbf{+} Only Freeform\n(20K samples)"
    else:
        return f"{base_name}"

def is_post_rl_model(model_name: str) -> bool:
    """Determine if a model is post-RL (checkpoint/finetuned) or pre-RL (original)."""
    # Check if model name contains checkpoint or other indicators of RL training
    post_rl_indicators = ['checkpoint', 'finetuned', 'ppo', 'rl', 'trained']
    return any(indicator in model_name.lower() for indicator in post_rl_indicators)

def plot_binary_ablations(model_data: Dict[str, Dict[str, Any]], judge: str, output_path: str, dataset_name: str = None):
    """Create horizontal bar charts for accuracy and Brier scores with two subplots."""
    
    if not model_data:
        print("No valid model data found for plotting")
        return
    
    # Prepare data for plotting
    model_keys = list(model_data.keys())
    # Sort so Pre-RL models come first, then reverse to put them at top of plot
    model_keys.sort(key=lambda x: (is_post_rl_model(model_data[x]['model_name']), 
                                   model_data[x]['model_name'], 
                                   model_data[x]['configuration']))
    
    # sort by accuracy
    model_keys.sort(key=lambda x: model_data[x]['mean_accuracy'])
    
    model_keys.reverse()  # Reverse so Pre-RL appears at top
    
    # Create figure with two subplots side by side - more compact
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Create figure with two subplots side by side, making the second subplot slightly bigger
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.13], wspace=0.075)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # Define nicer colors
    pre_rl_color = '#3498db'  # Beautiful blue
    # post_rl_color = '#e74c3c'  # Beautiful red
    post_rl_color = '#ff7f0e'  # Nicer orange-red (matplotlib's tab:orange)
    
    
    # Prepare data arrays
    y_positions = np.arange(len(model_keys))
    accuracies = []
    accuracy_errors = []
    brier_scores = []
    brier_errors = []
    bar_colors = []
    
    for model_key in model_keys:
        data = model_data[model_key]
        accuracies.append(data['mean_accuracy'])
        accuracy_errors.append(data['accuracy_std_error'])
        brier_scores.append(data['mean_brier'])
        brier_errors.append(data['brier_std_error'])
        
        # Determine color based on whether it's post-RL
        if is_post_rl_model(data['model_name']):
            bar_colors.append(post_rl_color)
        else:
            bar_colors.append(pre_rl_color)
    
    # Left subplot: Accuracy
    bars1 = ax1.barh(y_positions, accuracies, xerr=accuracy_errors, 
                     color=bar_colors, alpha=0.8, capsize=4, height=0.5)
    
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([format_model_name_for_display(model_data[key]['model_name'], 
                                                      model_data[key]['configuration']) 
                         for key in model_keys], fontsize=18, ha='right')
    ax1.set_xlabel('Accuracy \% ($\\uparrow$)', fontsize=22, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=20)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add value labels on accuracy bars
    for i, (bar, value, error) in enumerate(zip(bars1, accuracies, accuracy_errors)):
        if value > 0:
            ax1.text(value + error + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left', va='center', fontsize=22, fontweight='bold')
    
    # Right subplot: Brier Scores
    bars2 = ax2.barh(y_positions, brier_scores, xerr=brier_errors,
                     color=bar_colors, alpha=0.8, capsize=4, height=0.5)
    
    ax2.set_yticks(y_positions)
    # Remove y-axis labels from right subplot
    ax2.set_yticklabels([])
    ax2.set_xlabel('Freeform Brier Score ($\\uparrow$)', fontsize=22, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=20)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add value labels on Brier score bars
    for i, (bar, value, error) in enumerate(zip(bars2, brier_scores, brier_errors)):
        if value != 0:
            ax2.text(value + error + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontsize=22, fontweight='bold')
    
    # Create custom legend with better styling - add to both subplots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=pre_rl_color, label='Pre-RL Finetuning', alpha=0.8),
        Patch(facecolor=post_rl_color, label='Post-RL Finetuning', alpha=0.8)
    ]
    
    # Add legend at the top of both subplots
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.48, 1), 
               ncol=2, fontsize=19, frameon=True)
    
    # Remove main title
    
    # Reduce whitespace and improve layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08, top=0.85)  # Reduce gap between subplots and make room for legend
    
    # Set better axis limits for readability
    ax1.set_xlim(0, max(accuracies) * 1.2)
    ax2.set_xlim(min(brier_scores) * 3, max(brier_scores) * 1.3)
    # ax2.set_xlim(0, max(brier_scores) * 1.2)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Binary ablation plot saved to {output_path}")
    plt.close()

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
    
    # Get data for all models
    print(f"\nLoading data for all models in {args.input_dir}")
    model_data = get_model_data(args.input_dir, args.judge)
    
    if not model_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    output_filename = f"binary_ablations_{dataset_suffix}_{args.judge}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Plot the binary ablation results
    plot_binary_ablations(model_data, args.judge, output_path, dataset_name)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Judge: {args.judge}")
    print(f"Total model configurations: {len(model_data)}")
    
    for model_key, data in model_data.items():
        config_info = f" ({data['configuration']})" if data['configuration'] != 'unknown' else ""
        rl_status = "Post RL" if is_post_rl_model(data['model_name']) else "Pre RL"
        print(f"  {data['model_name']}{config_info} [{rl_status}]: "
              f"Accuracy = {data['mean_accuracy']:.2f}% ± {data['accuracy_std_error']:.2f}%, "
              f"Brier = {data['mean_brier']:.4f} ± {data['brier_std_error']:.4f}")

if __name__ == "__main__":
    main()
