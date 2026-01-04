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
    parser = argparse.ArgumentParser(description="Calculate and plot accuracy and Brier scores for folktexts evaluation")
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/folktexts/ACSTravelTime_1000/",
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
    
    Expected format: ModelName_test_size_N_generations_M_numeric.jsonl
    Returns: (model_name, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract model name (everything before _test)
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
    
    return model_name, num_generations

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
        return probability ** 2

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

def calculate_generation_folktexts_scores(data: List[Dict[str, Any]], generation_idx: int) -> Tuple[List[float], List[int]]:
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
    
    for item in data:
        # Skip items without necessary fields
        if "extracted_answer" not in item or "label" not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        label = item.get("label", None)
        
        # Skip if generation_idx is out of bounds or label is missing
        if generation_idx >= len(extracted_answers) or label is None:
            continue
            
        generation_answer = extracted_answers[generation_idx]
        label = int(label)
        
        # Handle numeric probability format
        if isinstance(generation_answer, (int, float)):
            probability = float(generation_answer)
            
            # Ensure probability is in valid range [0, 1]
            probability = max(0.0, min(1.0, probability))
            
            # Convert label to boolean (1 = True, 0 = False)
            is_correct = (label == 1)
            
            # Calculate scores
            brier_score = calculate_binary_brier_score(probability, is_correct)
            accuracy = calculate_binary_accuracy(probability, label)
            
            brier_scores.append(brier_score)
            accuracy_scores.append(accuracy)
    
    return brier_scores, accuracy_scores

def calculate_model_folktexts_statistics(data: List[Dict[str, Any]], num_generations: int) -> Tuple[float, float, float, float]:
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
        generation_brier_scores, generation_accuracy_scores = calculate_generation_folktexts_scores(data, gen_idx)
        
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

def get_model_folktexts_data(input_dir: str) -> Dict[str, Dict[str, Any]]:
    """Get Brier score and accuracy data for all models in the input directory."""
    model_data = {}
    baseline_data = None
    
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
        
        # Check if label field exists
        has_label_field = False
        for item in data:
            if "label" in item and item["label"] is not None:
                has_label_field = True
                break
        
        if not has_label_field:
            print(f"  Warning: label field not found in {filename}")
            continue
        
        # Calculate folktexts statistics
        mean_brier, brier_std_error, mean_accuracy, accuracy_std_error = calculate_model_folktexts_statistics(data, num_generations)
        
        model_data[model_key] = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'brier_std_error': brier_std_error,
            'mean_accuracy': mean_accuracy,
            'accuracy_std_error': accuracy_std_error,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        # Calculate baseline statistics from the first file (assuming all files have same labels)
        if baseline_data is None:
            baseline_data = calculate_baseline_statistics(data)
        
        print(f"  {model_name}: Mean Brier = {mean_brier:.4f} ± {brier_std_error:.4f}, Mean Accuracy = {mean_accuracy:.2f}% ± {accuracy_std_error:.2f}%")
    
    # Add baseline data to the model_data dictionary
    if baseline_data:
        model_data['baseline'] = baseline_data
        print(f"  Baseline: Majority Accuracy = {baseline_data['majority_accuracy']:.2f}%, Constant Predictor Accuracy = {baseline_data['constant_accuracy']:.2f}%")
    
    return model_data

def calculate_baseline_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate baseline statistics from the dataset."""
    labels = []
    for item in data:
        if "label" in item and item["label"] is not None:
            labels.append(int(item["label"]))
    
    if not labels:
        return None
    
    # Calculate majority class baseline
    majority_class = 1 if sum(labels) > len(labels) / 2 else 0
    majority_correct = sum(1 for label in labels if label == majority_class)
    majority_accuracy = (majority_correct / len(labels)) * 100.0
    
    # Calculate constant predictor baseline (always predict 0.5)
    # This gives us the Brier score for always predicting 0.5
    constant_probability = 0.5
    constant_brier_scores = []
    for label in labels:
        is_correct = (label == 1)
        brier_score = calculate_binary_brier_score(constant_probability, is_correct)
        constant_brier_scores.append(brier_score)
    
    constant_brier = np.mean(constant_brier_scores)
    constant_accuracy = 50.0  # Always predicting 0.5 gives 50% accuracy
    
    return {
        'model_name': 'baseline',
        'majority_accuracy': majority_accuracy,
        'constant_accuracy': constant_accuracy,
        'constant_brier': constant_brier,
        'majority_class': majority_class,
        'num_samples': len(labels)
    }

def extract_dataset_name(input_dir):
    """Extract and format dataset name from input directory path."""
    # Extract the last part of the path (e.g., ACSEmployment_1000)
    dataset_part = os.path.basename(input_dir.rstrip('/'))
    
    # Format it nicely
    if dataset_part.startswith('ACSEmployment_'):
        return f"Folktexts ACS Employment ({dataset_part.split('_', 1)[1]} samples)"
    elif dataset_part.startswith('ACSIncome_'):
        return f"Folktexts ACS Income ({dataset_part.split('_', 1)[1]} samples)"
    elif dataset_part.startswith('ACSMobility_'):
        return f"Folktexts ACS Mobility ({dataset_part.split('_', 1)[1]} samples)"
    elif dataset_part.startswith('ACSPublicCoverage_'):
        return f"Folktexts ACS Public Coverage ({dataset_part.split('_', 1)[1]} samples)"
    elif dataset_part.startswith('ACSTravelTime_'):
        return f"Folktexts ACS Travel Time ({dataset_part.split('_', 1)[1]} samples)"
    else:
        return f"Folktexts {dataset_part}"

def plot_brier_scores(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
    """Plot Brier scores as a bar chart with error bars."""
    
    # Get all unique model names (excluding baseline)
    all_model_names = sorted([name for name in model_data.keys() if name != 'baseline'])
    
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
            errors.append(model_data[model_name]['brier_std_error'])
        else:
            scores.append(0)
            errors.append(0)
        # Assign color based on whether "checkpoint" is in the model name
        if "checkpoint" in model_name.lower():
            bar_colors.append(checkpoint_color)
        else:
            bar_colors.append(default_color)
    
    # Create the plot   
    plt.figure(figsize=(14, 10))
    
    # Create bars
    bars = plt.bar(x_positions, scores, 
                   yerr=errors, capsize=5, 
                    alpha=0.8, color=bar_colors)
    
    # # Add baseline lines if available
    # if 'baseline' in model_data:
    #     baseline_data = model_data['baseline']
    #     constant_brier = baseline_data.get('constant_brier', 0.25)  # Default to 0.25 if not available
        
    #     # Add constant predictor baseline line
    #     plt.axhline(y=constant_brier, color='red', linestyle='--', linewidth=3, 
    #                label=f'Constant Predictor (0.5) = {constant_brier:.3f}')
        
    #     # Add legend
    #     plt.legend(fontsize=16, loc='upper right')
    
    # Customize the plot
    plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Brier Score', fontsize=24, fontweight='bold')
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Brier Score Performance', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Folktexts - Brier Score Performance', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in all_model_names], 
               rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value != 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.003,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
    add_value_labels(bars, scores, errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Brier score plot saved to {output_path}")
    plt.close()

def plot_accuracy(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
    """Plot accuracy as a bar chart with error bars."""
    
    # Get all unique model names (excluding baseline)
    all_model_names = sorted([name for name in model_data.keys() if name != 'baseline'])
    
    if not all_model_names:
        print("No valid model data found for plotting")
        return
    
    # Prepare data for plotting
    x_positions = np.arange(len(all_model_names))
    
    checkpoint_color = '#ff7f0e'  # orange
    default_color = '#2ca02c'     # green
    
    scores = []
    errors = []
    bar_colors = []
    
    for model_name in all_model_names:
        if model_name in model_data:
            scores.append(model_data[model_name]['mean_accuracy'])
            errors.append(model_data[model_name]['accuracy_std_error'])
        else:
            scores.append(0)
            errors.append(0)
        # Assign color based on whether "checkpoint" is in the model name
        if "checkpoint" in model_name.lower():
            bar_colors.append(checkpoint_color)
        else:
            bar_colors.append(default_color)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Create bars
    bars = plt.bar(x_positions, scores, 
                   yerr=errors, capsize=5, 
                   alpha=0.8, color=bar_colors)
    
    # Add baseline lines if available
    if 'baseline' in model_data:
        baseline_data = model_data['baseline']
        majority_accuracy = baseline_data.get('majority_accuracy', 50.0)
        
        # Add majority class baseline line
        plt.axhline(y=majority_accuracy, color='red', linestyle='--', linewidth=3, 
                   label=f'Majority Class = {majority_accuracy:.1f}%')
        
        # Add legend
        plt.legend(fontsize=16, loc='upper right')
    
    # Customize the plot
    plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Accuracy (%)', fontsize=24, fontweight='bold')
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Accuracy Performance', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Folktexts - Accuracy Performance', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in all_model_names], 
               rotation=45, ha='right', fontsize=22)
    plt.yticks(fontsize=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Set y-axis limits (accuracy is between 0 and 100)
    plt.ylim(0, 105)
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
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
    
    # Get folktexts forecasting data for all models
    print(f"\nCalculating folktexts forecasting metrics for all models in {args.input_dir}")
    model_data = get_model_folktexts_data(args.input_dir)
    
    if not model_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    
    brier_output_filename = f"folktexts_brier_scores_{dataset_suffix}.png"
    brier_output_path = os.path.join(args.output_dir, brier_output_filename)
    
    accuracy_output_filename = f"folktexts_accuracy_{dataset_suffix}.png"
    accuracy_output_path = os.path.join(args.output_dir, accuracy_output_filename)
    
    # Plot the Brier scores
    plot_brier_scores(model_data, brier_output_path, dataset_name)
    
    # Plot the accuracy
    plot_accuracy(model_data, accuracy_output_path, dataset_name)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Total model variants: {len([k for k in model_data.keys() if k != 'baseline'])}")
    
    # Print baseline statistics first
    if 'baseline' in model_data:
        baseline_data = model_data['baseline']
        print(f"\nBaseline Statistics:")
        print(f"  Majority Class Accuracy: {baseline_data['majority_accuracy']:.2f}%")
        print(f"  Constant Predictor (0.5) Accuracy: {baseline_data['constant_accuracy']:.2f}%")
        print(f"  Constant Predictor (0.5) Brier Score: {baseline_data['constant_brier']:.4f}")
        print(f"  Majority Class: {baseline_data['majority_class']}")
        print(f"  Total samples: {baseline_data['num_samples']}")
    
    # Print model statistics
    print(f"\nModel Statistics:")
    for model_key in sorted([k for k in model_data.keys() if k != 'baseline']):
        data = model_data[model_key]
        print(f"  {model_key}: {data['num_samples']} samples, "
              f"Brier = {data['mean_brier']:.4f} ± {data['brier_std_error']:.4f}, "
              f"Accuracy = {data['mean_accuracy']:.2f}% ± {data['accuracy_std_error']:.2f}%")

if __name__ == "__main__":
    main()
