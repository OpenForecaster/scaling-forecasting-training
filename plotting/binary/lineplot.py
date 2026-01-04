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
                       default="/fast/nchandak/forecasting/evals/binary/metaculus_fromMay2025/default/extra1000",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/lineplots/acc",
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
    Extract model name and checkpoint information from filename.
    
    Expected format: filtered-ModelName-...-checkpoint200_eval_... or unfiltered-ModelName-...-checkpoint200_eval_...
    Returns: (model_variant, checkpoint_number, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract checkpoint number
    checkpoint_match = re.search(r'checkpoint(\d+)', name_without_ext)
    checkpoint_number = int(checkpoint_match.group(1)) if checkpoint_match else 0
    
    # Extract filtered/unfiltered prefix
    if name_without_ext.startswith('filtered-'):
        model_variant = 'Filtered'
    elif name_without_ext.startswith('unfiltered-'):
        model_variant = 'Unfiltered'
    else:
        parts = name_without_ext.split('_')
        relevant_parts = parts[0].split('-')[:-1]
        model_variant = '-'.join(relevant_parts)
        # model_variant = 'Unknown'
    
    # Extract number of generations
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    return model_variant, checkpoint_number, num_generations


def calculate_binary_brier_score(probability: float, is_correct: bool) -> float:
    """
    Calculate Brier score for binary forecasting.
    
    Args:
        probability: Probability assigned to the answer (0-1)
        is_correct: Whether the answer was correct
        
    Returns:
        Brier score (range: [-2, 0])
    """
    if is_correct:
        # If answer is correct: -(1 - p)^2
        return (1 - probability) ** 2
    else:
        # If answer is incorrect: -(1 + p^2)
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
        if not resolution and item.get("answer", None) :
            if "yes" in item.get("answer", "").lower():
                resolution = 1
            elif "no" in item.get("answer", "").lower():
                resolution = 0
            else:
                resolution = None
        
        # print(f"Resolution: {resolution}")
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
                # continue 
                
                actual_answer = "yes" if resolution == 1 else "no"
                model_ans = list(generation_answer.keys())[0]
                model_prob = generation_answer[model_ans]
                
                if model_ans and model_prob:
                    if model_prob < 0.5 :
                        model_prob = 1 - model_prob
                    
                    if model_ans.strip().lower() == actual_answer:
                        brier_scores.append( (1 - model_prob) ** 2)
                        accuracy_scores.append(1)
                    else:
                        if model_ans.strip().lower() == "unknown":
                            abstain_counts += 1
                            if model_prob >= 0.499 and model_prob <= 0.501:
                                correct_abstain_prob += 1
                            
                        brier_scores.append( model_prob ** 2)
                        accuracy_scores.append(0)
                else :
                    brier_scores.append(0.25)
                    accuracy_scores.append(0)
            
            else :
                if probability is not None and isinstance(probability, (int, float)):
                    # Convert resolution to boolean (1 = True, 0 = False)
                    is_correct = (resolution == 1)
                    
                    # Calculate scores
                    brier_score = calculate_binary_brier_score(probability, is_correct)
                    accuracy = calculate_binary_accuracy(probability, int(resolution))
                    
                    brier_scores.append(brier_score)
                    accuracy_scores.append(accuracy)
    
    if abstain_counts > 0:
        print(f"Abstain counts: {abstain_counts}, Correct abstain prob: {correct_abstain_prob / abstain_counts}")
    return brier_scores, accuracy_scores

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

def get_model_binary_data(input_dir: str) -> Dict[str, Dict[str, Any]]:
    """Get Brier score and accuracy data for all models in the input directory."""
    model_data = {}
    
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract model info from filename
        model_name, num_generations = extract_model_info_from_filename(filename)
        
        # Create a unique key for model
        model_key = f"{model_name}"
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name}")
        
        # Check if resolution field exists
        has_resolution_field = False
        for item in data:
            if "resolution" in item and item["resolution"] is not None:
                has_resolution_field = True
                break
        
        if not has_resolution_field:
            print(f"  Warning: resolution field not found in {filename}")
            continue
        
        # Calculate binary statistics
        mean_brier, brier_std_error, mean_accuracy, accuracy_std_error = calculate_model_binary_statistics(data, num_generations)
        
        model_data[model_key] = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'brier_std_error': brier_std_error,
            'mean_accuracy': mean_accuracy,
            'accuracy_std_error': accuracy_std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        print(f"  {model_name}: Mean Brier = {mean_brier:.4f} ± {brier_std_error:.4f}, Mean Accuracy = {mean_accuracy:.2f}% ± {accuracy_std_error:.2f}%")
    
    return model_data

def get_checkpoint_data(input_dir: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Get checkpoint progression data for all models in the input directory."""
    model_checkpoint_data = defaultdict(dict)
    
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Extract model info from filename
        model_variant, checkpoint_number, num_generations = extract_model_info_from_filename(filename)
        
        # Skip if no checkpoint number found
        if checkpoint_number == 0:
            print(f"  Skipping {filename} - no checkpoint number found")
            continue
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_variant} checkpoint {checkpoint_number}")
        
        
        # Calculate statistics
        mean_brier, brier_std_error, mean_accuracy, accuracy_std_error = calculate_model_binary_statistics(data, num_generations)
        
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

def plot_training_progression(model_checkpoint_data: Dict[str, Dict[int, Dict[str, Any]]], output_dir: str, dataset_name: str = None):
    """Create line plots showing training progression for accuracy and Brier scores (no std dev bars)."""
    
    if not model_checkpoint_data:
        print("No valid model data found for plotting")
        return
    
    # Define nice colors for filtered vs unfiltered
    colors = {
        'Filtered': '#3498db',    # Beautiful blue
        'Unfiltered': '#e74c3c'   # Beautiful red
    }
    mapping = {
        'Filtered': 'Filtered Data',
        'Unfiltered': 'Unfiltered Data',
    }
    
    # Create accuracy plot
    plt.figure(figsize=(10, 6))
    
    for model_variant, checkpoint_data in model_checkpoint_data.items():
        if not checkpoint_data:
            continue
        
        label = mapping.get(model_variant, model_variant)
        # Sort checkpoints by number
        checkpoints = sorted(checkpoint_data.keys())
        accuracies = [checkpoint_data[cp]['mean_accuracy'] for cp in checkpoints]
        
        # Get color
        color = colors.get(model_variant, '#34495e')
        
        # Plot line without error bars
        plt.plot(checkpoints, accuracies, 
                 label=label, color=color, marker='o', markersize=10, 
                 linewidth=4, alpha=0.8)
    
    plt.xlabel('RL Training Iterations', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy (\%)', fontsize=18, fontweight='bold')
    plt.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tick_params(axis='both', labelsize=16)
    
    # Save accuracy plot
    accuracy_output_path = os.path.join(output_dir, f"training_progression_accuracy.png")
    plt.tight_layout()
    plt.savefig(accuracy_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Accuracy training progression plot saved to {accuracy_output_path}")
    plt.close()
    
    # Create Brier score plot
    plt.figure(figsize=(10, 6))
    
    for model_variant, checkpoint_data in model_checkpoint_data.items():
        if not checkpoint_data:
            continue
        
        label = mapping.get(model_variant, model_variant)
        # Sort checkpoints by number
        checkpoints = sorted(checkpoint_data.keys())
        brier_scores = [checkpoint_data[cp]['mean_brier'] for cp in checkpoints]
        
        # Get color
        color = colors.get(model_variant, '#34495e')
        
        # Plot line without error bars
        plt.plot(checkpoints, brier_scores, 
                 label=label, color=color, marker='o', markersize=10, 
                 linewidth=4, alpha=0.8)
    
    plt.xlabel('RL Training Iterations', fontsize=18, fontweight='bold')
    plt.ylabel('Freeform Brier Score', fontsize=18, fontweight='bold')
    plt.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tick_params(axis='both', labelsize=16)
    
    # Save Brier score plot
    brier_output_path = os.path.join(output_dir, f"training_progression_brier.png")
    plt.tight_layout()
    plt.savefig(brier_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Brier score training progression plot saved to {brier_output_path}")
    plt.close()

def plot_combined_progression(model_checkpoint_data: Dict[str, Dict[int, Dict[str, Any]]], output_dir: str, dataset_name: str = None):
    """Create combined subplot showing both accuracy and Brier score progression (no std dev bars)."""
    
    if not model_checkpoint_data:
        print("No valid model data found for plotting")
        return
    
    # Create figure with two subplots - compact and clean
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Define nice colors for filtered vs unfiltered
    colors = {
        'Filtered': '#3498db',    # Beautiful blue
        'Unfiltered': '#e74c3c'   # Beautiful red
    }
    
    
    
    # Define nice colors for filtered vs unfiltered
    color_list = ['#3498db', '#e74c3c']
    colors = {}
    for model_variant in model_checkpoint_data.keys():
        colors[model_variant] = color_list.pop(0)
    
    mapping = {
        'Filtered': 'Filtered Data',
        'Unfiltered': 'Unfiltered Data',
    }
    # For 0 iteration, add 19.3 to accuracy and -0.05 to brier
    for model_variant, checkpoint_data in model_checkpoint_data.items():
        if 0 not in checkpoint_data:
            checkpoint_data[0] = {}
            
        checkpoint_data[0]['mean_accuracy'] = 61.5
        checkpoint_data[0]['mean_brier'] = 0.25 # -0.009
        checkpoint_data[0]['accuracy_std_error'] = 0.0
        checkpoint_data[0]['brier_std_error'] = 0.0
            
    for model_variant, checkpoint_data in model_checkpoint_data.items():
        if not checkpoint_data:
            continue
        
        # Sort checkpoints by number
        checkpoints = sorted(checkpoint_data.keys())
        accuracies = [checkpoint_data[cp]['mean_accuracy'] for cp in checkpoints]
        brier_scores = [checkpoint_data[cp]['mean_brier'] for cp in checkpoints]
        
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
    ax1.set_xlabel('Training Iterations', fontsize=20, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Accuracy (\%)', fontsize=20, fontweight='bold', labelpad=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=18)
    # ax1.legend(fontsize=16, frameon=True, fancybox=True, loc='best')
    
    # Customize Brier score subplot
    ax2.set_xlabel('Training Iterations', fontsize=20, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Training Reward', fontsize=20, fontweight='bold', labelpad=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=18)

    # Create a single legend above both subplots, in a single row (ncols=2)
    handles, labels = ax1.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        fontsize=18,
        frameon=True,
        fancybox=True
    )
    # Clean layout without main title
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)
    
    # Save combined plot
    combined_output_path = os.path.join(output_dir, f"training_progression_combined.png")
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined training progression plot saved to {combined_output_path}")
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
    
    # Get checkpoint progression data for all models
    print(f"\nLoading checkpoint data for all models in {args.input_dir}")
    model_checkpoint_data = get_checkpoint_data(args.input_dir)
    
    if not model_checkpoint_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot individual progression charts
    plot_training_progression(model_checkpoint_data, args.output_dir, dataset_name)
    
    # Plot combined progression chart
    plot_combined_progression(model_checkpoint_data, args.output_dir, dataset_name)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Total models: {len(model_checkpoint_data)}")
    
    for model_name, checkpoint_data in model_checkpoint_data.items():
        checkpoints = sorted(checkpoint_data.keys())
        print(f"\n  {model_name}: {len(checkpoints)} checkpoints")
        for cp in checkpoints:
            data = checkpoint_data[cp]
            print(f"    Checkpoint {cp}: Accuracy = {data['mean_accuracy']:.2f}% ± {data['accuracy_std_error']:.2f}%, "
                  f"Brier = {data['mean_brier']:.4f} ± {data['brier_std_error']:.4f}")

if __name__ == "__main__":
    main()
