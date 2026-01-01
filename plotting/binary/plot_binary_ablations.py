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
                       default="/fast/nchandak/forecasting/evals/binary/metaculus_fromMay2025/default/binary-ablations",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/binary_ablations",
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
    Extract model name and configuration from filename.
    
    Expected format: ModelName_configuration_test_size_N_generations_M.jsonl
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
    
    # Extract configuration (both, onlybinary, freeform)
    if 'both' in name_without_ext:
        configuration = 'both'
    elif 'onlybinary' in name_without_ext:
        configuration = 'onlybinary'
    elif 'freeform' in name_without_ext:
        configuration = 'freeform'
    else:
        configuration = 'unknown'
    
    # Extract number of generations
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    if model_name.endswith('_'):
        model_name = model_name[:-1]
    
    return model_name, configuration, num_generations

def calculate_binary_brier_score(probability: float, is_correct: bool) -> float:
    """
    Calculate Brier score for binary forecasting.
    
    Args:
        probability: Probability assigned to the answer (0-1)
        is_correct: Whether the answer was correct
        
    Returns:
        Brier score (range: [0, 1])
    """
    if is_correct:
        # If answer is correct: (1 - p)^2
        return (1 - probability) ** 2
    else:
        # If answer is incorrect: p^2
        return (probability) ** 2

def calculate_binary_accuracy(probability: float, resolution: int) -> int:
    """
    Calculate accuracy for binary forecasting.
    
    Args:
        probability: Probability assigned to the answer (0-1)
        resolution: Ground truth resolution (0 or 1)
        
    Returns:
        Accuracy (1 if correct prediction, 0 otherwise)
    """
    # Model predicts YES if probability >= 0.5
    model_prediction = 1 if probability >= 0.5 else 0
    
    # Accuracy is 1 if prediction matches resolution
    return 1 if model_prediction == resolution else 0

def calculate_generation_binary_scores(data: List[Dict[str, Any]], generation_idx: int) -> Tuple[List[float], List[int]]:
    """
    Calculate Brier scores and accuracy for all questions in a specific generation.
    
    Args:
        data: List of evaluation entries
        generation_idx: Index of the generation to evaluate
        
    Returns:
        Tuple of (brier_scores, accuracy_scores) for each question in this generation
    """
    brier_scores = []
    accuracy_scores = []
    
    abstain_counts = 0
    correct_abstain_prob = 0
    
    for item in data:
        # Skip items without necessary fields
        if "extracted_answer" not in item or "resolution" not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        responses = item.get("response", [])
        resolution = item.get("resolution", None)
        
        # Skip if generation_idx is out of bounds or resolution is missing
        if generation_idx >= len(extracted_answers) or resolution is None:
            continue
            
        generation_answer = extracted_answers[generation_idx]
        response = responses[generation_idx]
        resolution = int(resolution)
        
        # Handle dictionary format (probability format)
        if isinstance(generation_answer, dict):
            # Extract probability for "YES" answer
            probability = generation_answer.get("YES", None)
            
            if "<answer>" in response:
                actual_answer = "yes" if resolution == 1 else "no"
                model_ans = list(generation_answer.keys())[0]
                model_prob = generation_answer[model_ans]
                
                if model_ans and model_prob:
                    if model_ans.strip().lower() == actual_answer:
                        brier_scores.append((1 - model_prob) ** 2)
                        accuracy_scores.append(1)
                    else:
                        if model_ans.strip().lower() == "unknown":
                            abstain_counts += 1
                            if model_prob >= 0.499 and model_prob <= 0.501:
                                correct_abstain_prob += 1
                            
                        brier_scores.append(model_prob ** 2)
                        accuracy_scores.append(0)
                else:
                    brier_scores.append(0.25)
                    accuracy_scores.append(0)
            
            else:
                if probability is not None and isinstance(probability, (int, float)):
                    # Convert resolution to boolean (1 = True, 0 = False)
                    is_correct = (resolution == 1)
                    
                    # Calculate scores
                    brier_score = calculate_binary_brier_score(probability, is_correct)
                    accuracy = calculate_binary_accuracy(probability, int(resolution))
                    
                    brier_scores.append(brier_score)
                    accuracy_scores.append(accuracy)
    
    if abstain_counts > 0:
        print(f"  Abstain counts: {abstain_counts}, Correct abstain prob: {correct_abstain_prob / abstain_counts:.3f}")
    return brier_scores, accuracy_scores

def calculate_baseline_performance(data: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Calculate baseline performance using constant classifier (always predict 1 or always predict 0).
    
    Args:
        data: List of evaluation entries
        
    Returns:
        Tuple of (baseline_accuracy, baseline_brier_score)
    """
    resolutions = []
    
    # Collect all resolutions
    for item in data:
        resolution = item.get("resolution", None)
        if resolution is not None:
            resolutions.append(int(resolution))
    
    if not resolutions:
        return 50.0, 0.25  # Default baseline if no data
    
    # Calculate accuracy for always predicting 1 and always predicting 0
    always_1_accuracy = sum(resolutions) / len(resolutions) * 100.0
    always_0_accuracy = (len(resolutions) - sum(resolutions)) / len(resolutions) * 100.0
    
    # Choose the better accuracy
    baseline_accuracy = max(always_1_accuracy, always_0_accuracy)
    
    # Calculate optimal Brier score using np.bin to find best constant probability
    # Create probability bins from 0 to 1
    prob_bins = np.linspace(0, 1, 101)  # 101 bins from 0.00 to 1.00
    
    # Calculate Brier score for each probability bin
    brier_scores = []
    for prob in prob_bins:
        brier_score = np.mean([(prob - res) ** 2 for res in resolutions])
        brier_scores.append(brier_score)
    
    # Find the optimal constant probability (lowest Brier score)
    optimal_prob_idx = np.argmin(brier_scores)
    optimal_prob = prob_bins[optimal_prob_idx]
    baseline_brier = brier_scores[optimal_prob_idx]
    
    print(f"  Optimal constant classifier: probability = {optimal_prob:.3f}, Brier = {baseline_brier:.4f}")
    
    return baseline_accuracy, baseline_brier

def calculate_model_binary_statistics(data: List[Dict[str, Any]], num_generations: int) -> Tuple[float, float, float, float]:
    """
    Calculate mean Brier score and accuracy with standard errors across all generations for a model.
    
    Args:
        data: List of evaluation entries
        num_generations: Number of generations to evaluate
        
    Returns:
        Tuple of (mean_brier_score, brier_std_error, mean_accuracy, accuracy_std_error)
    """
    all_generation_brier_means = []
    all_generation_accuracy_means = []
    
    for gen_idx in range(num_generations):
        generation_brier_scores, generation_accuracy_scores = calculate_generation_binary_scores(data, gen_idx)
        
        if generation_brier_scores:
            # Calculate mean Brier score for this generation across all questions
            generation_brier_mean = np.mean(generation_brier_scores)
            all_generation_brier_means.append(generation_brier_mean)
        
        if generation_accuracy_scores:
            # Calculate mean accuracy for this generation across all questions
            generation_accuracy_mean = np.mean(generation_accuracy_scores) * 100.0  # Convert to percentage
            all_generation_accuracy_means.append(generation_accuracy_mean)
    
    # Calculate Brier statistics
    if all_generation_brier_means:
        mean_brier = np.mean(all_generation_brier_means)
        brier_std_error = np.std(all_generation_brier_means, ddof=1) / np.sqrt(len(all_generation_brier_means)) if len(all_generation_brier_means) > 1 else 0.0
    else:
        mean_brier = 0.0
        brier_std_error = 0.0
    
    # Calculate accuracy statistics
    if all_generation_accuracy_means:
        mean_accuracy = np.mean(all_generation_accuracy_means)
        accuracy_std_error = np.std(all_generation_accuracy_means, ddof=1) / np.sqrt(len(all_generation_accuracy_means)) if len(all_generation_accuracy_means) > 1 else 0.0
    else:
        mean_accuracy = 0.0
        accuracy_std_error = 0.0
    
    return mean_brier, brier_std_error, mean_accuracy, accuracy_std_error

def get_model_data(input_dir: str) -> Tuple[Dict[str, Dict[str, Any]], float, float]:
    """Get accuracy and Brier score data for all models in the input directory."""
    model_data = {}
    all_data_for_baseline = []
    
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
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
        
        # Check if resolution field exists
        has_resolution_field = False
        for item in data:
            if "resolution" in item and item["resolution"] is not None:
                has_resolution_field = True
                break
        
        if not has_resolution_field:
            print(f"  Warning: resolution field not found in {filename}")
            continue
        
        # Collect data for baseline calculation (use first file's data)
        if not all_data_for_baseline:
            all_data_for_baseline = data
        
        # Calculate binary statistics
        mean_brier, brier_std_error, mean_accuracy, accuracy_std_error = calculate_model_binary_statistics(data, num_generations)
        
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
        
        print(f"  {model_name} ({configuration}): Accuracy = {mean_accuracy:.2f}% ± {accuracy_std_error:.2f}%, Brier = {mean_brier:.4f} ± {brier_std_error:.4f}")
    
    # Calculate baseline performance
    baseline_accuracy, baseline_brier = calculate_baseline_performance(all_data_for_baseline)
    print(f"Baseline constant classifier: Accuracy = {baseline_accuracy:.2f}%, Brier = {baseline_brier:.4f}")
    
    return model_data, baseline_accuracy, baseline_brier

# def format_model_name_for_display(model_name: str, configuration: str) -> str:
#     """Format model name for display with configuration information."""
#     # Shorten base model names for better readability
#     if 'qwen3-8b' in model_name.lower() or 'qwen3_8b' in model_name.lower():
#         if 'checkpoint' in model_name.lower():
#             base_name = 'Qwen3-8B (RL)'
#         else:
#             base_name = 'Qwen3-8B'
#     else:
#         base_name = print_names.get(model_name, model_name)
#         # Add RL indicator for checkpoints
#         if 'checkpoint' in model_name.lower():
#             base_name += ' (RL)'
    
#     if configuration == 'both':
#         return f"{base_name}\nBinary + Freeform"
#     elif configuration == 'onlybinary':
#         return f"{base_name}\nBinary Only"
#     elif configuration == 'freeform':
#         return f"{base_name}\nFreeform Only"
#     else:
#         return f"{base_name}"

# def is_post_rl_model(model_name: str) -> bool:
#     """Determine if a model is post-RL (checkpoint/finetuned) or pre-RL (original)."""
#     # Check if model name contains checkpoint or other indicators of RL training
#     post_rl_indicators = ['checkpoint', 'finetuned', 'ppo', 'rl', 'trained']
#     return any(indicator in model_name.lower() for indicator in post_rl_indicators)

# def plot_binary_ablations(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
#     """Create horizontal bar charts for accuracy and Brier scores with two subplots."""
    
#     if not model_data:
#         print("No valid model data found for plotting")
#         return
    
#     # Prepare data for plotting
#     model_keys = list(model_data.keys())
#     # Sort so Pre-RL models come first, then reverse to put them at top of plot
#     model_keys.sort(key=lambda x: (is_post_rl_model(model_data[x]['model_name']), 
#                                    model_data[x]['model_name'], 
#                                    model_data[x]['configuration']))
    
#     # Sort by accuracy
#     model_keys.sort(key=lambda x: model_data[x]['mean_accuracy'])
#     model_keys.reverse()  # Reverse so highest accuracy appears at top
    
#     # Create figure with two subplots side by side - more compact
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Define nicer colors
#     pre_rl_color = '#3498db'  # Beautiful blue
#     post_rl_color = '#e74c3c'  # Beautiful red
    
#     # Prepare data arrays
#     y_positions = np.arange(len(model_keys))
#     accuracies = []
#     accuracy_errors = []
#     brier_scores = []
#     brier_errors = []
#     bar_colors = []
    
#     for model_key in model_keys:
#         data = model_data[model_key]
#         accuracies.append(data['mean_accuracy'])
#         accuracy_errors.append(data['accuracy_std_error'])
#         brier_scores.append(data['mean_brier'])
#         brier_errors.append(data['brier_std_error'])
        
#         # Determine color based on whether it's post-RL
#         if is_post_rl_model(data['model_name']):
#             bar_colors.append(post_rl_color)
#         else:
#             bar_colors.append(pre_rl_color)
    
#     # Left subplot: Accuracy
#     bars1 = ax1.barh(y_positions, accuracies, xerr=accuracy_errors, 
#                      color=bar_colors, alpha=0.8, capsize=4, height=0.4)
    
#     ax1.set_yticks(y_positions)
#     ax1.set_yticklabels([format_model_name_for_display(model_data[key]['model_name'], 
#                                                       model_data[key]['configuration']) 
#                          for key in model_keys], fontsize=16, ha='right')
#     ax1.set_xlabel('Accuracy (%)', fontsize=18, fontweight='bold')
#     ax1.tick_params(axis='x', labelsize=16)
#     ax1.grid(True, alpha=0.3, linestyle='--', axis='x')
    
#     # Add value labels on accuracy bars
#     for i, (bar, value, error) in enumerate(zip(bars1, accuracies, accuracy_errors)):
#         if value > 0:
#             ax1.text(value + error + 1, bar.get_y() + bar.get_height()/2,
#                     f'{value:.1f}%', ha='left', va='center', fontsize=15, fontweight='bold')
    
#     # Right subplot: Brier Scores
#     bars2 = ax2.barh(y_positions, brier_scores, xerr=brier_errors,
#                      color=bar_colors, alpha=0.8, capsize=4, height=0.4)
    
#     ax2.set_yticks(y_positions)
#     # Remove y-axis labels from right subplot
#     ax2.set_yticklabels([])
#     ax2.set_xlabel('Brier Score', fontsize=18, fontweight='bold')
#     ax2.tick_params(axis='x', labelsize=16)
#     ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    
#     # Add value labels on Brier score bars
#     for i, (bar, value, error) in enumerate(zip(bars2, brier_scores, brier_errors)):
#         if value != 0:
#             ax2.text(value + error + 0.005, bar.get_y() + bar.get_height()/2,
#                     f'{value:.3f}', ha='left', va='center', fontsize=15, fontweight='bold')
    
#     # Create custom legend with better styling
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor=pre_rl_color, label='Pre-RL', alpha=0.8),
#         Patch(facecolor=post_rl_color, label='Post-RL', alpha=0.8)
#     ]
    
#     # Add legend at the top of both subplots
#     fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
#                ncol=2, fontsize=16, frameon=True, fancybox=True, shadow=True)
    
#     # Reduce whitespace and improve layout
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0.05, top=0.85)  # Reduce gap between subplots and make room for legend
    
#     # Set better axis limits for readability
#     ax1.set_xlim(0, max(accuracies) * 1.2)
#     ax2.set_xlim(0, max(brier_scores) * 1.2)
    
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     print(f"Binary ablation plot saved to {output_path}")
#     plt.close()



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
    
    
    if configuration == 'both':
        return "\\textbf{+} Binary \& Freeform\n(10K samples each)"
    elif configuration == 'onlybinary':
        return "\\textbf{+} Only Binary\n(20K samples)"
    elif configuration == 'freeform':
        return "\\textbf{+} Only Freeform\n(20K samples)"
    else:
        return f"{base_name}"

def is_post_rl_model(model_name: str) -> bool:
    """Determine if a model is post-RL (checkpoint/finetuned) or pre-RL (original)."""
    # Check if model name contains checkpoint or other indicators of RL training
    post_rl_indicators = ['checkpoint', 'finetuned', 'ppo', 'rl', 'trained', 'both', 'freeform', 'binary']
    return any(indicator in model_name.lower() for indicator in post_rl_indicators)

def plot_binary_ablations(model_data: Dict[str, Dict[str, Any]], baseline_accuracy: float, baseline_brier: float, output_path: str, dataset_name: str = None):
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
    # model_keys.sort(key=lambda x: model_data[x]['mean_accuracy'])
    
    # model_keys.reverse()  # Reverse so Pre-RL appears at top
    print("MODEL KEYS: ", model_keys)
    model_keys1 = model_keys[:1]
    model_keys2 = model_keys[1:]
    model_keys2.sort(key=lambda x: model_data[x]['mean_accuracy'])
    # model_keys2.reverse()
    model_keys = model_keys1 + model_keys2
    
    model_keys.reverse()  # Reverse so Pre-RL appears at top
    print("KEYS: ", model_keys)
    
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
    
    # # Add baseline line for accuracy
    # ax1.axvline(x=baseline_accuracy, color='gray', linestyle='--', alpha=0.8, linewidth=3)
    # ax1.text(baseline_accuracy + 2, len(model_keys) - 0.5, f'Baseline\n{baseline_accuracy:.1f}%', 
    #          ha='left', va='top', fontsize=14, color='gray', fontweight='bold')
    
    # Add value labels on accuracy bars
    for i, (bar, value, error) in enumerate(zip(bars1, accuracies, accuracy_errors)):
        if value > 0:
            ax1.text(value + error + 2, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left', va='center', fontsize=22, fontweight='bold')
    
    # Right subplot: Brier Scores
    bars2 = ax2.barh(y_positions, brier_scores, xerr=brier_errors,
                     color=bar_colors, alpha=0.8, capsize=4, height=0.5)
    
    ax2.set_yticks(y_positions)
    # Remove y-axis labels from right subplot
    ax2.set_yticklabels([])
    ax2.set_xlabel('Brier Score ($\\downarrow$)', fontsize=22, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=20)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # # Add baseline line for Brier score
    # ax2.axvline(x=baseline_brier, color='gray', linestyle='--', alpha=0.8, linewidth=3)
    # ax2.text(baseline_brier + 0.01, len(model_keys) - 0.5, f'Baseline\n{baseline_brier:.3f}', 
    #          ha='left', va='top', fontsize=14, color='gray', fontweight='bold')
    
    # Add value labels on Brier score bars
    for i, (bar, value, error) in enumerate(zip(bars2, brier_scores, brier_errors)):
        if value != 0:
            ax2.text(value + error + 0.01, bar.get_y() + bar.get_height()/2,
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
    ax2.set_xlim(0, max(brier_scores) * 1.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Binary ablation plot saved to {output_path}")
    plt.close()



def extract_dataset_name(input_dir):
    """Extract and format dataset name from input directory path."""
    # Extract the last part of the path (e.g., metaculus_fromMay2025)
    dataset_part = os.path.basename(input_dir.rstrip('/'))
    
    # Format it nicely
    if dataset_part.startswith('metaculus_'):
        return f"Metaculus Binary Forecasting ({dataset_part.split('_', 1)[1]})"
    elif dataset_part.startswith('manifold_'):
        return f"Manifold Binary Forecasting ({dataset_part.split('_', 1)[1]})"
    else:
        return f"{dataset_part} Binary Forecasting"

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
    
    # Get data for all models
    print(f"\nLoading data for all models in {args.input_dir}")
    model_data, baseline_accuracy, baseline_brier = get_model_data(args.input_dir)
    
    if not model_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    output_filename = f"binary_ablations_{dataset_suffix}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Plot the binary ablation results
    plot_binary_ablations(model_data, baseline_accuracy, baseline_brier, output_path, dataset_name)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Total model configurations: {len(model_data)}")
    
    for model_key, data in model_data.items():
        config_info = f" ({data['configuration']})" if data['configuration'] != 'unknown' else ""
        rl_status = "Post-RL" if is_post_rl_model(data['model_name']) else "Pre-RL"
        print(f"  {data['model_name']}{config_info} [{rl_status}]: "
              f"Accuracy = {data['mean_accuracy']:.2f}% ± {data['accuracy_std_error']:.2f}%, "
              f"Brier = {data['mean_brier']:.4f} ± {data['brier_std_error']:.4f}")

if __name__ == "__main__":
    main()

