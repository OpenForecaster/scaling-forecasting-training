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
    parser = argparse.ArgumentParser(description="Plot training progression line charts for accuracy and Brier scores")
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/theguardian_207/filtered10k",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/lineplots/arxiv",
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
    Extract model name and checkpoint information from filename.
    
    Expected format: filtered-ModelName-...-checkpoint200_eval_... or unfiltered-ModelName-...-checkpoint200_eval_...
    Returns: (model_variant, checkpoint_number, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract checkpoint number
    checkpoint_match = re.search(r'checkpoint(\d+)', name_without_ext)
    checkpoint_number = int(checkpoint_match.group(1)) if checkpoint_match else 0
    num_articles = int(re.search(r'num_articles_(\d+)', name_without_ext).group(1)) if re.search(r'num_articles_(\d+)', name_without_ext) else 11
    
    # Extract filtered/unfiltered prefix
    if name_without_ext.startswith('filtered-'):
        model_variant = 'Filtered'
    elif name_without_ext.startswith('unfiltered-leakage'):
        model_variant = 'Leakage'
    elif name_without_ext.startswith('past'):
        model_variant = 'Past'
    elif name_without_ext.startswith('unfiltered-'):
        model_variant = 'Unfiltered'
    elif "data66k-withbinary2k" in name_without_ext:
        if "rlvr" in name_without_ext:
            model_variant = "Only Accuracy"
        elif 'acc' in name_without_ext:
            model_variant = "Accuracy + Freeform Brier"
        else:
            model_variant = "Only Freeform Brier"
    else:
        parts = name_without_ext.split('_')
        relevant_parts = parts[0].split('-')[:-1]
        model_variant = '-'.join(relevant_parts)
        
        model_variant = parts[0]
        
        print("Variant:", model_variant)
        # if 'shuffled' in name_without_ext.lower():
        #     model_variant = 'Random Order'
        # elif 'nokl' in name_without_ext.lower():
        #     model_variant = 'No KL'
        # else :
        #     model_variant = 'Temporal Order'
            
        # model_variant = 'Unknown'
    
    # Extract number of generations
    print(model_variant, checkpoint_number)
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    return model_variant, checkpoint_number, num_generations, num_articles

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
        # If answer is incorrect: -(p^2)
        return - (probability ** 2)

def calculate_generation_brier_scores(data: List[Dict[str, Any]], generation_idx: int, judge_field: str) -> List[float]:
    """
    Calculate Brier scores for all questions in a specific generation.
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
                
                if answer_option in generation_scores:
                    is_correct = (int(generation_scores[answer_option]) == 1)
                    if is_correct:
                        any_correct = True
                    brier_score += calculate_brier_score(probability, is_correct)
                    
            if not any_correct:
                brier_score -= 1 # Penalize for not having any correct answer so its probability is taken as 0
            
            brier_scores.append(brier_score)
        
        # Handle string format (old format)
        elif isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
            is_correct = (int(generation_scores) == 1)
            if not is_correct:
                brier_score = -2
            else:
                brier_score = 0 
            brier_scores.append(brier_score)
    
    return brier_scores

def calculate_generation_accuracy(data: List[Dict[str, Any]], generation_idx: int, judge_field: str) -> float:
    """
    Calculate accuracy for all questions in a specific generation.
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

def calculate_model_statistics(data: List[Dict[str, Any]], num_generations: int, judge_field: str) -> Tuple[float, float, float, float]:
    """
    Calculate mean accuracy and Brier score with standard errors across all generations for a model.
    
    Returns:
        Tuple of (mean_accuracy, accuracy_std_error, mean_brier, brier_std_error)
    """
    all_generation_accuracies = []
    all_generation_brier_means = []
    
    for gen_idx in range(num_generations):
        # Calculate accuracy
        generation_accuracy = calculate_generation_accuracy(data, gen_idx, judge_field) * 100.0
        all_generation_accuracies.append(generation_accuracy)
        
        # Calculate Brier score
        generation_brier_scores = calculate_generation_brier_scores(data, gen_idx, judge_field)
        if generation_brier_scores:
            # Adjust Brier scores (add 1 to shift range from [-2,0] to [-1,1])
            adjusted_brier_scores = [score + 1 for score in generation_brier_scores]
            generation_brier_mean = np.mean(adjusted_brier_scores)
            all_generation_brier_means.append(generation_brier_mean)
    
    # Calculate accuracy statistics
    if all_generation_accuracies:
        mean_accuracy = np.mean(all_generation_accuracies)
        accuracy_std_error = np.std(all_generation_accuracies, ddof=1) / np.sqrt(len(all_generation_accuracies)) if len(all_generation_accuracies) > 1 else 0.0
    else:
        mean_accuracy = 0.0
        accuracy_std_error = 0.0
    
    # Calculate Brier statistics
    if all_generation_brier_means:
        mean_brier = np.mean(all_generation_brier_means)
        brier_std_error = np.std(all_generation_brier_means, ddof=1) / np.sqrt(len(all_generation_brier_means)) if len(all_generation_brier_means) > 1 else 0.0
    else:
        mean_brier = 0.0
        brier_std_error = 0.0
    
    return mean_accuracy, accuracy_std_error, mean_brier, brier_std_error

def get_checkpoint_data(input_dir: str, judge: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Get checkpoint progression data for all models in the input directory."""
    model_checkpoint_data = defaultdict(dict)
    
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # if "qwen" not in filename.lower():
        #     continue
        
        # Extract model info from filename
        model_variant, checkpoint_number, num_generations, num_articles = extract_model_info_from_filename(filename)
        
        # Skip if no checkpoint number found
        if checkpoint_number == 0 and num_articles == 11:
            print(f"  Skipping {filename} - no checkpoint number found")
            continue
        
        
        # For plotting accuracy with number of articles:
        if num_articles != 11:
            checkpoint_number = num_articles
        
        # Load data for this file
        try:
            data = load_jsonl_file(file_path)
            print(f"  Loaded {len(data)} samples for {model_variant} checkpoint {checkpoint_number}")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
        
        # Check if judge field exists
        judge_field = f"score_{judge}"
        has_judge_field = False
        
        for item in data:
            if judge_field in item:
                has_judge_field = True
                break
        
        if not has_judge_field:
            print(f"  Warning: {judge_field} not found in {filename}")
            continue
        
        # Calculate statistics
        mean_accuracy, accuracy_std_error, mean_brier, brier_std_error = calculate_model_statistics(data, num_generations, judge_field)
        
        model_checkpoint_data[model_variant][checkpoint_number] = {
            'mean_accuracy': mean_accuracy,
            'accuracy_std_error': accuracy_std_error,
            'mean_brier': mean_brier,
            'brier_std_error': brier_std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        print(f"  {model_variant} checkpoint {checkpoint_number}: Accuracy = {mean_accuracy:.2f}% ± {accuracy_std_error:.2f}%, Brier = {mean_brier:.4f} ± {brier_std_error:.4f}")
    
    return dict(model_checkpoint_data)

def plot_training_progression(model_checkpoint_data: Dict[str, Dict[int, Dict[str, Any]]], judge: str, output_dir: str, dataset_name: str = None):
    """Create line plots showing training progression for accuracy and Brier scores (no std dev bars)."""
    
    if not model_checkpoint_data:
        print("No valid model data found for plotting")
        return
    
    # Define nice colors for filtered vs unfiltered
    colors = {
        'Filtered': '#3498db',    # Beautiful blue
        'Unfiltered': '#e74c3c',   # Beautiful red
        'Leakage': '#9b59b6'   # Beautiful purple
    }
    
    xlabel = 'Training Iterations'
    max_checkpoint = 1 
    # plt.figure(figsize=(10, 6))
    
    # Define nice colors for filtered vs unfiltered
    color_list = ['#9b59b6', '#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#34495e', '#95a5a6']
    colors = {}
    for model_variant in model_checkpoint_data.keys():
        colors[model_variant] = color_list.pop(0)
    
    mapping = {
        'Filtered': 'Filtered Data',
        'Unfiltered': 'Unfiltered Data',
        'Leakage': 'Unfiltered w Leakage Data',
    }
    mapping = {
        'Filtered': 'Filtering with leakage removal',
        'Unfiltered': 'Leakage removal but no filtering ',
        'Leakage': 'No filtering',
    }
    # For 0 iteration, add 19.3 to accuracy and -0.05 to brier
    # for model_variant, checkpoint_data in model_checkpoint_data.items():
    #     if 0 not in checkpoint_data:
    #         checkpoint_data[0] = {}
            
    #         checkpoint_data[0]['mean_accuracy'] = 19.3
    #         checkpoint_data[0]['mean_brier'] = 0 # -0.009
    #         checkpoint_data[0]['accuracy_std_error'] = 0.0
    #         checkpoint_data[0]['brier_std_error'] = 0.0
            
    for model_variant, checkpoint_data in model_checkpoint_data.items():
        if not checkpoint_data:
            continue
        
        max_checkpoint = max(max_checkpoint, max(checkpoint_data.keys()))
        # Sort checkpoints by number
        checkpoints = sorted(list(map(int, checkpoint_data.keys())))
        accuracies = [checkpoint_data[cp]['mean_accuracy'] for cp in checkpoints]
        acc_errors = [checkpoint_data[cp]['accuracy_std_error'] for cp in checkpoints]
        brier_scores = [checkpoint_data[cp]['mean_brier'] for cp in checkpoints]
        brier_errors = [checkpoint_data[cp]['brier_std_error'] for cp in checkpoints]
        
        label = mapping.get(model_variant, model_variant)
        # Get color
        color = colors.get(model_variant, '#34495e')
        
        # Plot accuracy (no error bars)
        plt.plot(checkpoints, accuracies, 
                 label=label, color=color, marker='o', markersize=4, 
                 linewidth=2, alpha=0.8)
        
        # plt.errorbar(
        #     checkpoints, accuracies, yerr=acc_errors,
        #     label=label, color=color, marker='o', markersize=4,
        #     linewidth=2, alpha=0.8, capsize=3
        # )
        # ax1.errorbar(checkpoints, a# The above code snippet is plotting error bars on a graph using
        # the `errorbar` function in Matplotlib. It is likely part of a
        # larger script that is visualizing data related to accuracies
        # and errors. The code is currently commented out with `#`, so it
        # is not being executed.
        # ax1.errorbar(checkpoints, accuracies, yerr=acc_errors, color=color, marker='o', markersize=10, 
        #          linewidth=4, alpha=0.8)
        # ax2.errorbar(checkpoints, brier_scores, yerr=brier_errors, color=color, marker='o', markersize=10, 
        #          linewidth=4, alpha=0.8)
    
    print(f"Max checkpoint: {max_checkpoint}")
    if max_checkpoint <= 20:
        xlabel = 'Number of Articles in Prompt'
        
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.xticks([0, 2, 4, 6, 8, 10])
    plt.ylabel('Accuracy (\%)', fontsize=10, fontweight='bold', labelpad=6)
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tick_params(axis='both', labelsize=10)
    
    # Save accuracy plot
    accuracy_output_path = os.path.join(output_dir, f"training_progression_accuracy_{judge}.png")
    plt.tight_layout()
    plt.savefig(accuracy_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Accuracy training progression plot saved to {accuracy_output_path}")
    plt.savefig(accuracy_output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Accuracy training progression plot saved to {accuracy_output_path.replace('.png', '.pdf')}")
    plt.close()
    
    # # Create Brier score plot
    # plt.figure(figsize=(10, 6))
    
    # for model_variant, checkpoint_data in model_checkpoint_data.items():
    #     if not checkpoint_data:
    #         continue
        
    #     label = mapping.get(model_variant, model_variant)
    #     # Sort checkpoints by number
    #     checkpoints = sorted(checkpoint_data.keys())
    #     brier_scores = [checkpoint_data[cp]['mean_brier'] for cp in checkpoints]
        
    #     # Get color
    #     color = colors.get(model_variant, '#34495e')
        
    #     # Plot line without error bars
    #     plt.plot(checkpoints, brier_scores, 
    #              label=label, color=color, marker='o', markersize=10, 
    #              linewidth=4, alpha=0.8)
    
    # plt.xlabel(xlabel, fontsize=18, fontweight='bold')
    # plt.ylabel('Freeform Brier Score', fontsize=18, fontweight='bold')
    # plt.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, loc='best')
    # plt.grid(True, alpha=0.3, linestyle='--')
    # plt.tick_params(axis='both', labelsize=16)
    
    # # Save Brier score plot
    # brier_output_path = os.path.join(output_dir, f"training_progression_brier_{judge}.png")
    # plt.tight_layout()
    # plt.savefig(brier_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # print(f"Brier score training progression plot saved to {brier_output_path}")
    # plt.close()

def plot_combined_progression(model_checkpoint_data: Dict[str, Dict[int, Dict[str, Any]]], judge: str, output_dir: str, dataset_name: str = None):
    """Create combined subplot showing both accuracy and Brier score progression (no std dev bars)."""
    
    if not model_checkpoint_data:
        print("No valid model data found for plotting")
        return
    
    # Create figure with two subplots - compact and clean
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Define nice colors for filtered vs unfiltered
    colors = {
        'Filtered': '#3498db',    # Beautiful blue
        'Unfiltered': '#e74c3c',   # Beautiful red
        'Leakage': '#9b59b6'   # Beautiful purple
    }
    
    xlabel = 'Training Iterations'
    max_checkpoint = 1 
    
    # Define nice colors for filtered vs unfiltered
    color_list = ['#9b59b6', '#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#34495e', '#95a5a6']
    colors = {}
    for model_variant in model_checkpoint_data.keys():
        colors[model_variant] = color_list.pop(0)
    
    mapping = {
        'Filtered': 'Filtered Data',
        'Unfiltered': 'Unfiltered Data',
        'Leakage': 'Unfiltered w Leakage Data',
    }
    mapping = {
        'Filtered': 'Filtering with leakage removal',
        'Unfiltered': 'Leakage removal but no filtering ',
        'Leakage': 'No filtering',
    }
    # # For 0 iteration, add 19.3 to accuracy and -0.05 to brier
    # for model_variant, checkpoint_data in model_checkpoint_data.items():
    #     if 0 not in checkpoint_data:
    #         checkpoint_data[0] = {}
            
    #         checkpoint_data[0]['mean_accuracy'] = 19.3
    #         checkpoint_data[0]['mean_brier'] = 0 # -0.009
    #         checkpoint_data[0]['accuracy_std_error'] = 0.0
    #         checkpoint_data[0]['brier_std_error'] = 0.0
            
    for model_variant, checkpoint_data in model_checkpoint_data.items():
        if not checkpoint_data:
            continue
        
        max_checkpoint = max(max_checkpoint, max(checkpoint_data.keys()))
        # Sort checkpoints by number
        checkpoints = sorted(checkpoint_data.keys())
        accuracies = [checkpoint_data[cp]['mean_accuracy'] for cp in checkpoints]
        acc_errors = [checkpoint_data[cp]['accuracy_std_error'] for cp in checkpoints]
        brier_scores = [checkpoint_data[cp]['mean_brier'] for cp in checkpoints]
        brier_errors = [checkpoint_data[cp]['brier_std_error'] for cp in checkpoints]
        
        label = mapping.get(model_variant, model_variant)
        # Get color
        color = colors.get(model_variant, '#34495e')
        
        # Plot accuracy (no error bars)
        ax1.plot(checkpoints, accuracies, 
                 label=label, color=color, marker='o', markersize=10, 
                 linewidth=4, alpha=0.8)
    
        # Plot Brier scores (no error bars)
        ax2.plot(checkpoints, brier_scores,
                 label=label, color=color, marker='o', markersize=10, 
                 linewidth=4, alpha=0.8)
        
        # ax1.errorbar(checkpoints, a# The above code snippet is plotting error bars on a graph using
        # the `errorbar` function in Matplotlib. It is likely part of a
        # larger script that is visualizing data related to accuracies
        # and errors. The code is currently commented out with `#`, so it
        # is not being executed.
        # ccuracies, yerr=acc_errors, color=color, marker='o', markersize=10, 
        #          linewidth=4, alpha=0.8)
        # ax2.errorbar(checkpoints, brier_scores, yerr=brier_errors, color=color, marker='o', markersize=10, 
        #          linewidth=4, alpha=0.8)
    
    print(f"Max checkpoint: {max_checkpoint}")
    if max_checkpoint <= 20:
        xlabel = 'Number of Articles'
        
    # color: e74c3c
    # Add horizontal reference lines to show filtered data reaches performance faster
    # Accuracy: horizontal line at unfiltered checkpoint 300 performance, extending to checkpoint 100
    if 'Unfiltered' in model_checkpoint_data and 300 in model_checkpoint_data['Unfiltered']:
        unfiltered_300_accuracy = model_checkpoint_data['Unfiltered'][300]['mean_accuracy']
        ax1.axhline(y=unfiltered_300_accuracy, color='black', linestyle='--', 
                   alpha=0.7, linewidth=3, xmin=0.33, xmax=0.82)  # xmax=0.33 corresponds to checkpoint 100 if max is 300
        ax1.text(150, unfiltered_300_accuracy + 0.1, f'3x less data', 
                fontsize=22, color='black', fontweight='bold')
    
    # Brier: horizontal line at unfiltered checkpoint 350 performance, extending to checkpoint 150
    if 'Unfiltered' in model_checkpoint_data and 350 in model_checkpoint_data['Unfiltered']:
        unfiltered_350_brier = model_checkpoint_data['Unfiltered'][350]['mean_brier']
        ax2.axhline(y=unfiltered_350_brier, color='black', linestyle='--', 
                   alpha=0.7, linewidth=3, xmin=0.43, xmax=0.82)  # xmax=0.43 corresponds to checkpoint 150 if max is 350
        ax2.text(280, unfiltered_350_brier + 0.003, f'2x faster', 
                fontsize=20, color='black', fontweight='bold')
    
    # Customize accuracy subplot
    ax1.set_xlabel(xlabel, fontsize=20, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Accuracy (\%)', fontsize=20, fontweight='bold', labelpad=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=20)
    # ax1.legend(fontsize=16, frameon=True, fancybox=True, loc='best')
    
    # Customize Brier score subplot
    ax2.set_xlabel(xlabel, fontsize=20, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Freeform Brier Score', fontsize=20, fontweight='bold', labelpad=-2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=20)

    # Create a single legend above both subplots, in a single row (ncols=2)
    handles, labels = ax1.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(labels),
        fontsize=18,
        frameon=True,
        fancybox=True
    )
    # Clean layout without main title
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)
    
    # Save combined plot
    combined_output_path = os.path.join(output_dir, f"training_progression_combined_{judge}.png")
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined training progression plot saved to {combined_output_path}")
    plt.savefig(combined_output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined training progression plot saved to {combined_output_path.replace('.png', '.pdf')}")
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
    
    last_folder = os.path.basename(args.input_dir)
    args.output_dir = os.path.join(args.output_dir, last_folder)
    # Extract dataset name from input directory
    dataset_name = extract_dataset_name(args.input_dir)
    
    print(f"Processing directory: {args.input_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Judge: {args.judge}")
    
    # Get checkpoint progression data for all models
    print(f"\nLoading checkpoint data for all models in {args.input_dir}")
    model_checkpoint_data = get_checkpoint_data(args.input_dir, args.judge)
    
    if not model_checkpoint_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot individual progression charts
    plot_training_progression(model_checkpoint_data, args.judge, args.output_dir, dataset_name)
    
    # Plot combined progression chart
    plot_combined_progression(model_checkpoint_data, args.judge, args.output_dir, dataset_name)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Judge: {args.judge}")
    print(f"Total models: {len(model_checkpoint_data)}")
    print(f"Output directory: {args.output_dir}")
    
    for model_name, checkpoint_data in model_checkpoint_data.items():
        checkpoints = sorted(checkpoint_data.keys())
        print(f"\n  {model_name}: {len(checkpoints)} checkpoints")
        for cp in checkpoints:
            data = checkpoint_data[cp]
            print(f"    Checkpoint {cp}: Accuracy = {data['mean_accuracy']:.2f}% ± {data['accuracy_std_error']:.2f}%, "
                  f"Brier = {data['mean_brier']:.4f} ± {data['brier_std_error']:.4f}")

if __name__ == "__main__":
    main()
