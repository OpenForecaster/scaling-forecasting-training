#!/usr/bin/env python3
import os
import json
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
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


# Display names for models
print_names = {
    'qwen3-4b': 'Qwen3-4B',
    'Qwen3-4B': 'Qwen3-4B',
    'qwen3-8b': 'Qwen3-8B',
    'Qwen3-8B': 'Qwen3-8B',
    'qwen3-1.7b': 'Qwen3-1.7B',
    'Qwen3-1.7B': 'Qwen3-1.7B',
    'qwen3-32b': 'Qwen3-32B',
    'Qwen3-32B': 'Qwen3-32B',
    'deepseek-chat-v3-0324': 'deepseek-v3', # 'V3-0324',
    'DeepSeek-V3-0324': 'V3-0324',
    # 'deepseek-r1-0528': 'R1 0528',
    'deepseek-r1-0528': 'deepseek-r1',
    # 'deepseek-r1': 'R1',
    'llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'llama-4-maverick': 'llama-4-maverick',
    'llama-4-scout': 'Scout',
    'claude-3.5-haiku': 'Claude 3.5 Haiku',
    'gpt-4o': 'GPT 4o',
    'gpt-4o-mini': 'GPT 4o Mini',
    'o4-mini-high': 'o4 Mini High',
    'grok-3-mini-beta': 'Grok 3 Mini',
    'grok-4': 'Grok 4',
    'kimi-k2': 'Kimi K2',
    'gemini-2.5-pro-preview': 'Gemini 2.5 Pro',
    'gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
    # 'Qwen3-4B-sft-rl': '\\texttt{OpenForecaster}-4B',
    # 'Qwen3-8B-sft-rl': '\\texttt{OpenForecaster}-8B',
    'Qwen3-4B-RL': '\\texttt{OpenForecaster}-4B',
    'Qwen3-8B-RL': '\\texttt{OpenForecaster}-8B',
    'qwen3-235b-a22b': 'Qwen3-235B-A22B',
    'qwen3-235b-a22b-07-25': 'Qwen3-235B-A22B',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot scaling curves for training sample size")
    parser.add_argument("--input_dir", type=str,
                        default="/fast/nchandak/forecasting/evals/freeform/manual/test5news_302/trainsamples",
                        help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/scaling",
                        help="Output directory for plots")
    parser.add_argument("--judge", type=str, default="Llama_4_Scout",
                        help="Judge model name for score field")
    parser.add_argument("--base-model-name", type=str, default="llama-3.1-8b",
                        help="Base model name (without training samples)")
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
    Extract model name, number of generations, and training samples from filename.
    Returns: (model_name, num_generations, train_samples)
    """
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract training samples if present
    train_match = re.search(r'train(\d+)', name_without_ext)
    train_samples = int(train_match.group(1)) if train_match else None
    
    # Extract model name
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
    
    return model_name, num_generations, train_samples


def calculate_brier_score(probability: float, is_correct: bool) -> float:
    """Calculate Brier score."""
    if is_correct:
        return -((1 - probability) ** 2)
    else:
        return -(probability ** 2)


def calculate_generation_brier_scores(data: List[Dict[str, Any]], generation_idx: int, judge_field: str) -> List[float]:
    """Calculate Brier scores for all questions in a specific generation."""
    brier_scores = []
    
    for item in data:
        if "extracted_answer" not in item or judge_field not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        judge_scores = item.get(judge_field, [])
        
        if generation_idx >= len(extracted_answers) or generation_idx >= len(judge_scores):
            continue
            
        generation_answer = extracted_answers[generation_idx]
        generation_scores = judge_scores[generation_idx]
        
        if isinstance(generation_answer, dict) and isinstance(generation_scores, dict):
            any_correct = False
            brier_score = 0
            for answer_option, probability in generation_answer.items():
                if not answer_option or not probability:
                    continue
                
                if answer_option in generation_scores:
                    is_correct = (int(generation_scores[answer_option]) == 1)
                    if is_correct:
                        any_correct = True
                    brier_score += calculate_brier_score(probability, is_correct)
                    
            if not any_correct:
                brier_score -= 1
            
            brier_scores.append(brier_score)
        
        elif isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
            is_correct = (int(generation_scores) == 1)
            if not is_correct:
                brier_score = -2
            else:
                brier_score = 0
            brier_scores.append(brier_score)
    
    return brier_scores


def calculate_model_brier_statistics(data: List[Dict[str, Any]], num_generations: int, judge_field: str) -> Tuple[float, float]:
    """Calculate mean Brier score and standard error across all generations."""
    all_generation_means = []
    
    for gen_idx in range(num_generations):
        generation_brier_scores = calculate_generation_brier_scores(data, gen_idx, judge_field)
        
        if generation_brier_scores:
            generation_brier_scores = [score + 1 for score in generation_brier_scores]
            generation_mean = np.mean(generation_brier_scores)
            all_generation_means.append(generation_mean)
    
    if not all_generation_means:
        return 0.0, 0.0
    
    mean_brier = np.mean(all_generation_means)
    std_error = np.std(all_generation_means, ddof=1) / np.sqrt(len(all_generation_means)) if len(all_generation_means) > 1 else 0.0
    
    return mean_brier, std_error


def calculate_generation_accuracy(data: List[Dict[str, Any]], generation_idx: int, judge_field: str) -> float:
    """Calculate accuracy for all questions in a specific generation."""
    correct_count = 0
    total_count = 0
    
    for item in data:
        if "extracted_answer" not in item or judge_field not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        judge_scores = item.get(judge_field, [])
        
        if generation_idx >= len(extracted_answers) or generation_idx >= len(judge_scores):
            continue
            
        generation_answer = extracted_answers[generation_idx]
        generation_scores = judge_scores[generation_idx]
        
        if isinstance(generation_answer, dict) and isinstance(generation_scores, dict):
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
        
        elif isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
            is_correct = (int(generation_scores) == 1)
            if is_correct:
                correct_count += 1
            total_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0


def calculate_model_accuracy_statistics(data: List[Dict[str, Any]], num_generations: int, judge_field: str) -> Tuple[float, float]:
    """Calculate mean accuracy and standard error across all generations."""
    all_generation_accuracies = []
    
    for gen_idx in range(num_generations):
        generation_accuracy = calculate_generation_accuracy(data, gen_idx, judge_field) * 100.0
        all_generation_accuracies.append(generation_accuracy)
    
    if not all_generation_accuracies:
        return 0.0, 0.0
    
    mean_accuracy = np.mean(all_generation_accuracies)
    std_error = np.std(all_generation_accuracies, ddof=1) / np.sqrt(len(all_generation_accuracies)) if len(all_generation_accuracies) > 1 else 0.0
    
    return mean_accuracy, std_error


def get_scaling_data(input_dir: str, judge: str, base_model_name: str) -> Tuple[Dict[int, Dict], Dict[str, Dict]]:
    """
    Get scaling data for training samples and baseline models.
    
    Returns:
        Tuple of (scaling_data, baseline_models)
        - scaling_data: Dict mapping train_samples -> metrics
        - baseline_models: Dict mapping model_name -> metrics
    """
    scaling_data = {}
    baseline_models = {}
    
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")]
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        model_name, num_generations, train_samples = extract_model_info_from_filename(filename)
        
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples")
        
        judge_field = f"score_{judge}"
        has_judge_field = any(judge_field in item for item in data)
        
        if not has_judge_field:
            print(f"  Warning: {judge_field} not found in {filename}")
            continue
        
        mean_brier, brier_se = calculate_model_brier_statistics(data, num_generations, judge_field)
        mean_acc, acc_se = calculate_model_accuracy_statistics(data, num_generations, judge_field)
        
        metrics = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'brier_se': brier_se,
            'mean_accuracy': mean_acc,
            'acc_se': acc_se,
            'num_samples': len(data),
            'num_generations': num_generations,
        }
        
        print(f"  Brier = {mean_brier:.4f} ± {brier_se:.4f}, Accuracy = {mean_acc:.2f}% ± {acc_se:.2f}%")
        
        # If this has training samples, add to scaling data
        if train_samples is not None and base_model_name.lower() in model_name.lower():
            scaling_data[train_samples] = metrics
        # Otherwise, it's a baseline model
        elif train_samples is None:
            baseline_models[model_name] = metrics
    
    return scaling_data, baseline_models


def plot_scaling_curve(scaling_data: Dict[int, Dict], baseline_models: Dict[str, Dict], 
                       output_path: str, metric: str, ylabel: str, base_model_name: str):
    """Plot scaling curve for a given metric."""
    
    if not scaling_data:
        print(f"No scaling data found for {metric}")
        return
    
    # Sort scaling data by training samples
    train_samples = sorted(scaling_data.keys())
    metric_key = f'mean_{metric}'
    # Fix: accuracy uses 'acc_se' not 'accuracy_se'
    if metric == 'accuracy':
        error_key = 'acc_se'
    else:
        error_key = f'{metric}_se'
    
    # Get base model metrics (no training)
    base_metrics = None
    for model_name, metrics in baseline_models.items():
        if base_model_name.lower() in model_name.lower() and 'train' not in model_name.lower():
            base_metrics = metrics
            break
    
    # Build the curve data with evenly spaced x positions
    x_values = []  # Actual training sample values for labels
    x_positions = []  # Evenly spaced positions for plotting
    y_values = []
    y_errors = []
    
    position = 0  # Start at position 0
    
    # Add base model point at x=0
    if base_metrics:
        x_values.append(0)  # Actual value is 0
        x_positions.append(position)  # Position 0
        y_values.append(base_metrics[metric_key])
        if metric == 'accuracy':
            y_errors.append(base_metrics.get('acc_se', 0))
        else:
            y_errors.append(base_metrics.get(error_key, 0))
        position += 1
    
    # Add trained model points with evenly spaced positions
    for samples in train_samples:
        x_values.append(samples)
        x_positions.append(position)
        y_values.append(scaling_data[samples][metric_key])
        if metric == 'accuracy':
            y_errors.append(scaling_data[samples].get('acc_se', 0))
        else:
            y_errors.append(scaling_data[samples].get(error_key, 0))
        position += 1
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    red_color   = '#E24A33'  # vibrant red
    green_color = '#47A23F'  # vibrant green
    blue_color  = '#348ABD'  # vibrant blue
    yellow_color = '#E5A50A'  # vibrant yellow
    purple_color = '#9558B2'  # vibrant purple
    orange_color = '#D9730D'  # vibrant orange
    brown_color = '#8A5136'  # vibrant brown
    gray_color = '#666666'  # vibrant gray
    
    # Plot the scaling curve with a vibrant blue color using evenly spaced positions
    ax.errorbar(x_positions, y_values, yerr=y_errors, 
                fmt='o-', linewidth=5, markersize=10,
                capsize=5, capthick=2, label=base_model_name,
                color=green_color, zorder=10)
    
    # Plot baseline models as horizontal dashed lines
    # Vibrant color palette - bright and distinct colors
    colors = ['#00D084', '#FF1744', '#AA00FF', '#00E5FF', '#FF6E40', '#FFEA00', '#00BFA5']
    color_idx = 0
    
    # Vibrant, contrasting color palette (cycled as needed)
    VIBRANT_COLORS = [
        # '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#17becf',
        '#66c2a5', '#8da0cb',  '#fc8d62', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'
    ]
    colors = [red_color, blue_color, purple_color, yellow_color, orange_color, brown_color, gray_color]

    # Get the left x position for baseline model labels
    if len(x_positions) > 0:
        min_x = min(x_positions)
        max_x = max(x_positions)
        # Position labels on the left side
        label_x_position = min_x + (max_x - min_x) * 0.15
    else:
        label_x_position = 0.5
    
    for model_name, metrics in baseline_models.items():
        # Skip the base model (already plotted as part of curve)
        if base_model_name.lower() in model_name.lower():
            continue
        
        y_val = metrics[metric_key]
        display_name = print_names.get(model_name, model_name)
        
        ax.axhline(y=y_val, linestyle='--', linewidth=4, 
                  color=colors[color_idx % len(colors)], 
                  alpha=0.7, zorder=5)
        
        # Position label above or below the line based on model name
        if 'gpt-oss' in model_name.lower() or 'v3' in model_name.lower():
            # For gpt-oss models, place label below the line
            v_align = 'top'
            # Calculate a small downward offset (as percentage of y-range)
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_offset = -0.01 * y_range  # Push down by 2% of y-range
            label_y = y_val + y_offset
        else:
            # For other models, place label above the line
            v_align = 'bottom'
            label_y = y_val #* 1.001
        
        # Add label centered on the line
        ax.text(label_x_position, label_y, display_name,
               verticalalignment=v_align, horizontalalignment='center',
               fontsize=24, color=colors[color_idx % len(colors)], 
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
               fc='white', ec='none', alpha=0.8))
        
        color_idx += 1
    
    # Use linear scale with custom tick positions and labels
    ax.set_xlabel('Dataset Size (Samples from \\texttt{OpenForesight})', fontsize=24, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20, length=6, width=1.2)
    ax.grid(True, alpha=0.4, linestyle='--', which='major')
    ax.legend(fontsize=20, loc='lower right')
    
    # Set x-axis limits - no padding after last point
    if x_positions:
        min_pos = min(x_positions)
        max_pos = max(x_positions)
        ax.set_xlim(min_pos - 0.3, max_pos + 0.3)  # Small symmetric padding only
        
        # Set custom x-ticks to show actual training sample values at evenly spaced positions
        ax.set_xticks(x_positions)
        
        # Format labels with K suffix
        def format_k_label(value):
            if value == 0:
                return '0'
            k_value = value / 1000
            # If it's a whole number, don't show decimals
            if k_value == int(k_value):
                return f'{int(k_value)}K'
            else:
                # Show decimals but strip trailing zeros
                formatted = f'{k_value:.10f}'.rstrip('0').rstrip('.')
                return f'{formatted}K'
        
        ax.set_xticklabels([format_k_label(x) for x in x_values])
    
    # Adjust y-axis limits to extend higher for label visibility
    current_ylim = ax.get_ylim()
    y_range = current_ylim[1] - current_ylim[0]
    ax.set_ylim(current_ylim[0], current_ylim[1] + y_range * 0.05)
    
    # Add axis break symbol between 0 and first training sample (after ylim is set)
    if x_positions and len(x_positions) > 1:
        # Position the break marks between first two points
        break_x = (x_positions[0] + x_positions[1]) / 2
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Draw diagonal break lines on the x-axis
        break_height = y_range * 0.015
        break_width = 0.08
        
        # # Left diagonal
        # ax.plot([break_x - break_width, break_x - break_width/3], 
        #        [y_min - break_height, y_min + break_height],
        #        'k-', linewidth=2.5, clip_on=False, zorder=100)
        # # Right diagonal
        # ax.plot([break_x + break_width/3, break_x + break_width], 
        #        [y_min - break_height, y_min + break_height],
        #        'k-', linewidth=2.5, clip_on=False, zorder=100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved {metric} plot to {output_path}")
    plt.close()


def main():
    args = parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing directory: {args.input_dir}")
    print(f"Judge: {args.judge}")
    print(f"Base model: {args.base_model_name}")
    
    # Get scaling and baseline data
    scaling_data, baseline_models = get_scaling_data(args.input_dir, args.judge, args.base_model_name)
    
    if not scaling_data:
        print("No scaling data found")
        return
    
    print(f"\nFound {len(scaling_data)} training sample points")
    print(f"Found {len(baseline_models)} baseline models")
    
    # Generate output filenames
    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    
    # Plot Brier score scaling
    brier_output = os.path.join(args.output_dir, f"brier_scores_trainsamples_{args.base_model_name}.png")
    plot_scaling_curve(scaling_data, baseline_models, brier_output, 
                      'brier', 'Brier Score ($\\uparrow$)', args.base_model_name)
    
    # Plot accuracy scaling
    accuracy_output = os.path.join(args.output_dir, f"accuracy_trainsamples_{args.base_model_name}.png")
    plot_scaling_curve(scaling_data, baseline_models, accuracy_output,
                      'accuracy', 'Accuracy (\\%) ($\\uparrow$)', args.base_model_name)


if __name__ == "__main__":
    main()

