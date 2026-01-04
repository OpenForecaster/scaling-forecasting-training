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
import glob

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
    parser = argparse.ArgumentParser(description="Compare Brier scores and accuracy between two evaluation directories")
    parser.add_argument("--primary", type=str, 
                       default="/fast/nchandak/forecasting/evals/binary/metaculus_fromMay2025/default",
                       help="Primary directory containing evaluation JSONL files")
    parser.add_argument("--secondary", type=str, 
                       default="/fast/nchandak/forecasting/evals/binary/metaculus_fromMay2025/retrieval_30",
                       help="Secondary directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/comparison",
                       help="Output directory for comparison plots")
    return parser.parse_args()

def load_jsonl_file(file_path):
    """Load data from a JSONL file."""
    data = []
    try :
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
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
        if "extracted_answer" not in item : #or "resolution" not in item:
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
                    # assert probability >= 0 and probability <= 1, f"Probability is {probability}"
                    
                    if probability < 0 or probability > 1:
                        continue 
                    
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
        
        # if not has_resolution_field:
        #     print(f"  Warning: resolution field not found in {filename}")
        #     continue
        
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
        
        print(f"  {model_name}: Brier = {mean_brier:.4f} ± {brier_std_error:.4f}, Accuracy = {mean_accuracy:.4f} ± {accuracy_std_error:.4f}")
    
    return model_data

def get_secondary_label(primary_path: str, secondary_path: str) -> str:
    """Generate a label for the secondary dataset by removing the primary part."""
    primary_name = os.path.basename(primary_path.rstrip('/'))
    secondary_name = os.path.basename(secondary_path.rstrip('/'))
    
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
        plt.title(f'Brier Score Comparison', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in common_models], 
               rotation=45, ha='right', fontsize=26)
    plt.yticks(fontsize=22)
    
    # Add legend
    plt.legend(fontsize=20)
    
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
    
    for model_name in common_models:
        primary_scores.append(primary_data[model_name]['mean_accuracy'])
        primary_errors.append(primary_data[model_name]['accuracy_std_error'])
        secondary_scores.append(secondary_data[model_name]['mean_accuracy'])
        secondary_errors.append(secondary_data[model_name]['accuracy_std_error'])
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Create bars
    bars1 = plt.bar(x_positions - width/2, primary_scores, width, 
                   yerr=primary_errors, capsize=5, 
                   alpha=0.8, color='#2ca02c', label='Without Retrieval')
    bars2 = plt.bar(x_positions + width/2, secondary_scores, width,
                   yerr=secondary_errors, capsize=5, 
                   alpha=0.8, color='#ff7f0e', label=f'{secondary_label}')
    
    # Customize the plot
    # plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Accuracy (%)', fontsize=28, fontweight='bold')
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Accuracy Comparison', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Accuracy Comparison', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in common_models], 
               rotation=45, ha='right', fontsize=26)
    plt.yticks(fontsize=22)
    
    # Set y-axis limits (accuracy is between 0 and 1)
    plt.ylim(0, max(max(primary_scores), max(secondary_scores)) * 1.1)
    
    # Add legend
    plt.legend(fontsize=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.5,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    add_value_labels(bars1, primary_scores, primary_errors)
    add_value_labels(bars2, secondary_scores, secondary_errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison plot saved to {output_path}")
    plt.close()

def extract_dataset_name(input_dir):
    """Extract and format dataset name from input directory path."""
    # Extract the last part of the path (e.g., theguardian_207, dw_21317)
    dataset_part = os.path.basename(input_dir.rstrip('/'))
    
    parts = input_dir.split("/")
    
    for dataset_part in parts:
        # Format it nicely
        if dataset_part.startswith('theguardian'):
            number_match = re.search(r'(\d+)', dataset_part)
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
    
    # Get data from both directories
    print(f"\nProcessing primary directory: {args.primary}")
    primary_data = get_model_binary_data(args.primary)
    
    print(f"\nProcessing secondary directory: {args.secondary}")
    secondary_data = get_model_binary_data(args.secondary)
    
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
    brier_output_filename = f"brier_comparison_{primary_suffix}_vs_{secondary_suffix}.png"
    brier_output_path = os.path.join(args.output_dir, brier_output_filename)
    plot_comparison_brier(primary_data, secondary_data, secondary_label, brier_output_path, dataset_name)
    
    # Plot accuracy comparison
    accuracy_output_filename = f"accuracy_comparison_{primary_suffix}_vs_{secondary_suffix}.png"
    accuracy_output_path = os.path.join(args.output_dir, accuracy_output_filename)
    plot_comparison_accuracy(primary_data, secondary_data, secondary_label, accuracy_output_path, dataset_name)
    
    # Print summary statistics
    print(f"\nComparison Summary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Primary: {args.primary}")
    print(f"Secondary: {args.secondary} ({secondary_label})")
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
