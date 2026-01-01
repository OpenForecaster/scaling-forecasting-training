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
import ast
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
    'grok-3-mini': 'Grok 3 Mini',
    'grok-4': 'Grok 4',
    'kimi-k2': 'Kimi K2',
    'gemini-2.5-pro-preview': 'Gemini 2.5 Pro',
    'gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and plot Brier scores for FutureX-Past evaluation")
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/futurex-past86-retrieval",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output_dir", type=str, default="plots/",
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
    
    Expected format: ModelName_train_level_X_size_N_generations_M.jsonl
    Returns: (model_name, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract model name (everything before _train)
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
    
    
def calculate_generation_brier_scores_futurex(data: List[Dict[str, Any]], generation_idx: int) -> List[float]:
    """
    Calculate Brier scores for all questions in a specific generation for FutureX-Past data.
    
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
        ground_truth_raw = item.get("answer", "")
        is_binary_list = item.get("is_binary", [])
        if len(is_binary_list) == 0:
            is_binary = "no" in ground_truth_raw.lower() or "yes" in ground_truth_raw.lower()
            is_binary_list = [is_binary] * len(extracted_answers)
            
        level = int(item.get("level", 0))
        if level > 3 :
            continue # only consider level 1 questions for brier score
        
        # Skip if generation_idx is out of bounds
        if generation_idx >= len(extracted_answers):
            continue
            
            
        # Parse ground truth
        try:
            if isinstance(ground_truth_raw, str):
                # Handle string format like "['A']" or "['Yes']"
                if ground_truth_raw.startswith('[') and ground_truth_raw.endswith(']'):
                    ground_truth_list = ast.literal_eval(ground_truth_raw)
                    ground_truth = ground_truth_list[0].lower() if ground_truth_list else ""
                else:
                    ground_truth = ground_truth_raw.lower()
            else:
                ground_truth = str(ground_truth_raw).lower()
        except Exception as e:
            # print(f"Error in parsing ground truth: {e}")
            continue
            
        
        # print(f"generation_idx: {generation_idx}, is_binary_list: {is_binary_list}")
        # Only process binary questions
        if generation_idx < len(is_binary_list) and is_binary_list[generation_idx] == 1:
            generation_answer = extracted_answers[generation_idx]
            
            # Handle dictionary format (answer: probability)
            if isinstance(generation_answer, dict) and len(generation_answer) > 0:
                # Get the answer and probability
                answer_key = list(generation_answer.keys())[0]
                probability = list(generation_answer.values())[0]
                
                if answer_key and probability is not None:
                    # Determine if the answer is correct
                    predicted_answer = answer_key.lower().strip()
                    
                    # Check correctness
                    is_correct = False
                    if ground_truth in ["yes", "y", "true", "1"]:
                        is_correct = predicted_answer in ["yes", "y", "true", "1"]
                    elif ground_truth in ["no", "n", "false", "0"]:
                        is_correct = predicted_answer in ["no", "n", "false", "0"]
                    else:
                        # For other answers, do exact match
                        is_correct = predicted_answer == ground_truth
                    
                    if not isinstance(probability, float):
                        continue
                        
                    if probability > 1 or probability < 0:
                        continue
                        
                    #print(f"Predicted: {predicted_answer}, Ground truth: {ground_truth}, Is correct: {is_correct}, Probability: {probability}")
                    # Calculate Brier score
                    brier_score = calculate_brier_score(float(probability), is_correct)
                    brier_scores.append(brier_score)
                    
        else :
            # calculate brier score for freeform questions
            generation_answer = extracted_answers[generation_idx]
            
            # Handle dictionary format (new probabilistic format)
            if isinstance(generation_answer, dict) :
                # For each answer option in this generation
                any_correct = False
                brier_score = 0
                for answer_option, probability in generation_answer.items():
                    if not answer_option or not probability:
                        continue 
                    
                    if probability == None:
                        print(f"Probability is None for {answer_option}, {generation_answer}")
                        
                    if not isinstance(probability, float):
                        continue
                        
                    if probability > 1 or probability < 0:
                        continue
                        
                    correctness = 0
                    predicted = answer_option.lower()
                    # Check if prediction matches any ground truth answer
                    if isinstance(ground_truth, list):
                        if predicted.lower() in ground_truth or any(pred.lower().strip() in gt for gt in ground_truth for pred in [predicted]):
                            correctness = 1
                    else:
                        if predicted.lower() == ground_truth or predicted.lower().strip() in str(ground_truth):
                            correctness = 1
                                    
                    is_correct = (correctness == 1)
                    if is_correct:
                        any_correct = True
                    brier_score += calculate_brier_score(probability, is_correct)
                        
                if not any_correct:
                    brier_score -= 1 # Penalize for not having any correct answer so its probability is taken as 0
                
                # brier_score *= -1 
                brier_scores.append(brier_score)
            else :
                assert False, "Generation answer is not a dictionary"
                
            # print(f"brier_score: {brier_score}")
    
    return [1 + score for score in brier_scores]

def calculate_model_brier_statistics_futurex(data: List[Dict[str, Any]], num_generations: int) -> Tuple[float, float]:
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
        generation_brier_scores = calculate_generation_brier_scores_futurex(data, gen_idx)
        
        if generation_brier_scores:
            # Calculate mean Brier score for this generation across all questions
            generation_mean = np.mean(generation_brier_scores)
            all_generation_means.append(generation_mean)
    
    if not all_generation_means:
        return 0.0, 0.0
    
    # Calculate mean and standard error across generations
    mean_brier = np.mean(all_generation_means)
    std_error = np.std(all_generation_means, ddof=1) / np.sqrt(len(all_generation_means)) if len(all_generation_means) > 1 else 0.0
    
    return mean_brier, std_error

def get_model_brier_data_futurex(input_dir: str) -> Dict[str, Dict[str, Any]]:
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
        
        # Create a unique key for model
        model_key = f"{model_name}"
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name} with num generations {num_generations}")
        
        # Calculate Brier statistics
        mean_brier, std_error = calculate_model_brier_statistics_futurex(data, num_generations)
        
        # Count binary questions
        binary_count = 0
        total_generations = 0
        for item in data:
            is_binary_list = item.get("is_binary", [])
            for gen_idx in range(min(num_generations, len(is_binary_list))):
                total_generations += 1
                if is_binary_list[gen_idx] == 1:
                    binary_count += 1
        
        model_data[model_key] = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'std_error': std_error,
            'num_samples': len(data),
            'num_generations': num_generations,
            'binary_count': binary_count,
            'total_generations': total_generations
        }
        
        print(f"  {model_name}: Mean Brier = {mean_brier:.4f} ± {std_error:.4f} (Binary: {binary_count}/{total_generations})")
    
    return model_data

def calculate_generation_accuracy_futurex(data: List[Dict[str, Any]], generation_idx: int) -> float:
    """
    Calculate accuracy for all questions in a specific generation for FutureX-Past data.
    
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
        ground_truth_raw = item.get("answer", "")
        level = int(item.get("level", 0))
        is_binary_list = item.get("is_binary", [])
        if len(is_binary_list) == 0:
            is_binary = "no" in ground_truth_raw.lower() or "yes" in ground_truth_raw.lower()
            is_binary_list = [is_binary] * len(extracted_answers)
        
        
        # Skip if generation_idx is out of bounds
        if generation_idx >= len(extracted_answers):
            continue
            
        generation_answer = extracted_answers[generation_idx]
        
        # Parse ground truth
        try:
            if isinstance(ground_truth_raw, str):
                if ground_truth_raw.startswith('[') and ground_truth_raw.endswith(']'):
                    ground_truth_list = ast.literal_eval(ground_truth_raw)
                    if isinstance(ground_truth_list, list) and len(ground_truth_list) > 0 and isinstance(ground_truth_list[0], str):
                        ground_truth = ground_truth_list[0].lower()
                else:
                    ground_truth = ground_truth_raw.lower()
            else:
                ground_truth = str(ground_truth_raw).lower()
        except Exception as e:
            print(f"Error in parsing ground truth for accuracy: {e}. Ground truth raw: {ground_truth_raw}")
            continue
        
        
        if level <= 1 :
            # Handle dictionary format (answer: probability)
            if isinstance(generation_answer, dict) and len(generation_answer) > 0:
                # Get the predicted answer
                predicted = list(generation_answer.keys())[0].lower() if generation_answer else None
                if predicted:
                    # Check if prediction matches ground truth
                    is_correct = False
                    if ground_truth in ["yes", "y", "true", "1"]:
                        is_correct = predicted in ["yes", "y", "true", "1"]
                    elif ground_truth in ["no", "n", "false", "0"]:
                        is_correct = predicted in ["no", "n", "false", "0"]
                    else:
                        # For other answers, do exact match or substring match
                        is_correct = predicted == ground_truth or predicted.strip() in ground_truth
                    
                    if is_correct:
                        correct_count += 1
                    total_count += 1
        elif level == 4 :
            ground_truth = ground_truth_list[0]
            # print(f"ground_truth: {ground_truth}")
            if not isinstance(ground_truth, float) and not isinstance(ground_truth, int):
                # print(f"ground_truth: {ground_truth}, generation_answer: {generation_answer}")
                # print(f"question: {item.get('question')}")
                continue 
            
            # Handle dictionary format (answer: probability)
            if isinstance(generation_answer, dict) and len(generation_answer) > 0:
                # Get the predicted answer
                predicted = list(generation_answer.keys())[0].lower() if generation_answer else None
                if predicted:
                    # Check if prediction matches ground truth
                    try:
                        estimation = float(predicted)
                        ground_truth = float(ground_truth)
                        # print(f"estimation: {estimation}, ground_truth: {ground_truth}")
                        # score = 1 - relative error where relative error is |estimation - ground_truth| / ground_truth
                        relative_error = abs(estimation - ground_truth) / ground_truth
                        score = 1 - relative_error
                        # print(f"score: {score}")
                    except:
                        print(f"Error in calculating relative error for {predicted} and {ground_truth}")
                        score = 0
                        # continue
                    correct_count += max(0, score)
                    total_count += 1
        else :
            continue
    
    print(f"correct_count: {correct_count}, total_count: {total_count}")
    return correct_count / total_count if total_count > 0 else 0.0

def calculate_model_accuracy_statistics_futurex(data: List[Dict[str, Any]], num_generations: int) -> Tuple[float, float]:
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
        generation_accuracy = calculate_generation_accuracy_futurex(data, gen_idx) * 100.0
        all_generation_accuracies.append(generation_accuracy)
    
    if not all_generation_accuracies:
        return 0.0, 0.0
    
    # Calculate mean and standard error across generations
    mean_accuracy = np.mean(all_generation_accuracies)
    std_error = np.std(all_generation_accuracies, ddof=1) / np.sqrt(len(all_generation_accuracies)) if len(all_generation_accuracies) > 1 else 0.0
    
    return mean_accuracy, std_error

def get_model_accuracy_data_futurex(input_dir: str) -> Dict[str, Dict[str, Any]]:
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
        
        # Create a unique key for model
        model_key = f"{model_name}"
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name}")
        
        # Calculate accuracy statistics
        mean_accuracy, std_error = calculate_model_accuracy_statistics_futurex(data, num_generations)
        
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
    # Extract the last part of the path
    dataset_part = os.path.basename(input_dir.rstrip('/'))
    
    # Format it nicely
    if 'futurex' in dataset_part.lower():
        return "FutureX-Past Level 1"
    else:
        return f"{dataset_part} Forecasting"

def plot_brier_scores2(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
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
    default_color = '#2ca02c'     # green
    
    for model_name in all_model_names:
        if model_name in model_data:
            scores.append(model_data[model_name]['mean_brier'])
            errors.append(model_data[model_name]['std_error'])
        else:
            scores.append(0)
            errors.append(0)
        
        # Use default color for all models
        bar_colors.append(default_color)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Create bars with custom colors
    bars = plt.bar(x_positions, scores, 
                   yerr=errors, capsize=5, 
                   alpha=0.8, color=bar_colors)
    
    # Customize the plot
    plt.xlabel('Model', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Brier Score', fontsize=24, fontweight='bold')
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Brier Score', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Brier Score Performance', fontsize=28, fontweight='bold', pad=30)
    
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
    
    add_value_labels(bars, scores, errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Brier score plot saved to {output_path}")
    # also save the plot to a pdf
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
    print(f"Brier score plot saved to {output_path.replace('.png', '.pdf')}")
    plt.close()
    

def plot_brier_scores(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
    """Plot Brier scores as a bar chart with error bars, using a different color for models with 'checkpoint' in their name."""
    
    # Get all unique model names
    all_model_names = sorted(set(list(model_data.keys())))
    # all_model_names = sorted(set(list(modified_model_names)))
    
    if not all_model_names:
        print("No valid model data found for plotting")
        return
    
    
    modified_model_names = []
    for model_name in all_model_names:
        # if "rl" in model_name[-2:].lower():
        #     model_name = model_name[:-3] + " \\textbf{+ RL}"
        # if "checkpoint" in model_name.lower():
        #     parts = model_name.split("-")
        #     prefix = "-".join(parts[:2])
        #     model_name = prefix + " \\textbf{+ RL}"
        # if "checkpoint" in model_name.lower():
        modified_model_names.append(model_name)
    
    print(f"Model names: {modified_model_names}")
    
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
    plt.xlabel('Model', fontsize=26, fontweight='bold')
    plt.ylabel('Freeform Brier Score', fontsize=26, fontweight='bold', labelpad=12)
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Brier Score', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Brier Score Performance', fontsize=28, fontweight='bold', pad=30)
    
    
    # plt.title(f'Brier Score Performance - {judge} Judge', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in modified_model_names], 
               rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=24)
    
    # add legend based on the bar colors
    # plt.legend(bars, ['Default', 'Trained on our data'], fontsize=24, loc='best', frameon=True, fancybox=True)
    
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=28, fontweight='bold')
            elif value < 0:
                height = bar.get_height()
                print(f"Height: {height}, Error: {error}")
                plt.text(bar.get_x() + bar.get_width()/2., height - error - 0.04,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    add_value_labels(bars, scores, errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Brier score plot saved to {output_path}")   
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
    print(f"Brier score plot saved to {output_path.replace('.png', '.pdf')}")
    plt.close()

def plot_accuracy2(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
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
    default_color = '#2ca02c'     # green
    
    for model_name in all_model_names:
        if model_name in model_data:
            scores.append(model_data[model_name]['mean_accuracy'])
            errors.append(model_data[model_name]['std_error'])
        else:
            scores.append(0)
            errors.append(0)
        
        # Use default color for all models
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
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
    add_value_labels(bars, scores, errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to {output_path}")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to {output_path.replace('.png', '.pdf')}")
    plt.close()


def plot_accuracy(model_data: Dict[str, Dict[str, Any]], output_path: str, dataset_name: str = None):
    """Plot accuracy as a bar chart with error bars, using a different color for models with 'checkpoint' in their name."""
    
    # Get all unique model names
    # all_model_names = sorted(set(list(model_data.keys())))
    
    # Get all unique model names
    all_model_names = sorted(set(list(model_data.keys())))
    # all_model_names = sorted(set(list(modified_model_names)))
    
    if not all_model_names:
        print("No valid model data found for plotting")
        return
    
    
    modified_model_names = []
    for model_name in all_model_names:
        # if "rl" in model_name[-2:].lower():
        #     model_name = model_name[:-3] + " \\textbf{+ RL}"
        # if "checkpoint" in model_name.lower():
        #     parts = model_name.split("-")
        #     prefix = "-".join(parts[:2])
        #     model_name = prefix + " \\textbf{+ RL}"
        # if "checkpoint" in model_name.lower():
        modified_model_names.append(model_name)
    
    print(f"Model names: {modified_model_names}")
    
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
    plt.xlabel('Model', fontsize=26, fontweight='bold')
    plt.ylabel('Accuracy (\%)', fontsize=26, fontweight='bold', labelpad=10)
    
    # Set title
    if dataset_name:
        plt.title(f'{dataset_name} - Accuracy Performance', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Accuracy Performance', fontsize=28, fontweight='bold', pad=30)
    
    # Set x-axis labels
    plt.xticks(x_positions, [print_names.get(name, name) for name in modified_model_names], 
               rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=26)
    
    # add legend based on the bar colors
    # plt.legend(bars, ['Default', 'Trained on our data'], fontsize=24, loc='best', frameon=True, fancybox=True)
    
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Set y-axis limits (accuracy is between 0 and 1)
    plt.ylim(0, max(scores) * 1.1)
    
    # Add value labels on bars
    def add_value_labels(bars, values, errors):
        for bar, value, error in zip(bars, values, errors):
            if value > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    add_value_labels(bars, scores, errors)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to {output_path}")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to {output_path.replace('.png', '.pdf')}")
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
    
    # Get Brier score data for all models
    print(f"\nCalculating Brier scores for all models in {args.input_dir}")
    brier_model_data = get_model_brier_data_futurex(args.input_dir)
    
    # Get accuracy data for all models
    print(f"\nCalculating accuracy for all models in {args.input_dir}")
    accuracy_model_data = get_model_accuracy_data_futurex(args.input_dir)
    
    if not brier_model_data and not accuracy_model_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    final_output_dir = os.path.join(args.output_dir, dataset_suffix)
    os.makedirs(final_output_dir, exist_ok=True)
    
    if brier_model_data:
        brier_output_filename = f"brier_scores_{dataset_suffix}.png"
        brier_output_path = os.path.join(final_output_dir, brier_output_filename)
        
        # Plot the Brier scores
        plot_brier_scores(brier_model_data, brier_output_path, dataset_name)
    
    if accuracy_model_data:
        accuracy_output_filename = f"accuracy_{dataset_suffix}.png"
        accuracy_output_path = os.path.join(final_output_dir, accuracy_output_filename)
        
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
        binary_info = ""
        
        if model_key in brier_model_data:
            brier_data = brier_model_data[model_key]
            brier_info = f"Brier = {brier_data['mean_brier']:.4f} ± {brier_data['std_error']:.4f}"
            samples_info = f"{brier_data['num_samples']} samples"
            binary_info = f"Binary: {brier_data['binary_count']}/{brier_data['total_generations']}"
        
        if model_key in accuracy_model_data:
            accuracy_data = accuracy_model_data[model_key]
            accuracy_info = f"Accuracy = {accuracy_data['mean_accuracy']:.1f}% ± {accuracy_data['std_error']:.1f}%"
            if not samples_info:
                samples_info = f"{accuracy_data['num_samples']} samples"
        
        # Combine the information
        metrics = [info for info in [brier_info, accuracy_info, binary_info] if info]
        metrics_str = ", ".join(metrics)
        
        print(f"  {model_key}: {samples_info}, {metrics_str}")

if __name__ == "__main__":
    main() 