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
# Make everything larger and more readable
mpl.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 34,
    'axes.labelsize': 30,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
})

# Custom display names for models
print_names = {
    'Qwen3-4B': 'Qwen3-4B',
    'Qwen3-8B': 'Qwen3-8B', 
    'Qwen3-4B-sft-rl': '\\texttt{OpenForecaster}-4B',
    'Qwen3-8B-sft-rl': '\\texttt{OpenForecaster}-8B',
}

# Dataset configurations
DATASETS = {
    'SimpleQA': {
        'path': '/fast/nchandak/forecasting/evals/freeform/SimpleQA/simpleqa-iclr',
        'type': 'judge',
        'judge_field': 'score_Llama_4_Scout',
        'marker': 'o',  # circle
        'color': '#1f77b4'  # blue
    },
    'GPQA': {
        'path': '/fast/nchandak/forecasting/evals/gpqa/gpqa_diamond', 
        'type': 'mcq',
        'judge_field': None,
        'marker': 's',  # square
        'color': '#ff7f0e'  # orange
    },
    'MMLU Pro': {
        'path': '/fast/nchandak/forecasting/evals/mmlu_pro/mmlu_pro',
        'type': 'mcq',
        'judge_field': None,
        'marker': '^',  # triangle
        'color': '#2ca02c'  # green
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Scatter plot of change in accuracy vs change in brier across benchmarks")
    parser.add_argument("--output-dir", type=str, default="plots/across_benchmarks",
                       help="Output directory for plots")
    parser.add_argument("--judge", type=str, default="Llama_4_Scout",
                       help="Judge model name for SimpleQA evaluation")
    parser.add_argument("--model-size", type=str, choices=['4B', '8B', 'both'], default='8B',
                       help="Which model size to plot (4B, 8B, or both)")
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

def extract_model_info_from_filename(filename, dataset_type='mcq'):
    """Extract model name and number of generations from filename."""
    name_without_ext = filename.replace('.jsonl', '')
    
    if dataset_type == 'mcq':
        # For MMLU Pro and GPQA: ModelName_split_size_N_generations_M.jsonl
        model_match = re.match(r'([^_]+(?:-[^_]*)*?)(?:_(?:train|test))', name_without_ext)
        if model_match:
            model_name = model_match.group(1)
        else:
            parts = name_without_ext.split('_')
            model_name = parts[0]
            for i, part in enumerate(parts[1:], 1):
                if part in ['train', 'test', 'size'] or part.isdigit():
                    break
                model_name += f"_{part}"
    else:
        # For SimpleQA and other freeform: ModelName_test_size_N_generations_M.jsonl
        model_match = re.match(r'([^_]+(?:-[^_]*)*?)(?:_(?:test|eval))', name_without_ext)
        if model_match:
            model_name = model_match.group(1)
        else:
            parts = name_without_ext.split('_')
            model_name = parts[0]
            for i, part in enumerate(parts[1:], 1):
                if part in ['train', 'test', 'eval', 'size'] or part.isdigit():
                    break
                model_name += f"_{part}"
    
    # Extract number of generations
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    return model_name, num_generations

def calculate_brier_score(probability: float, is_correct: bool) -> float:
    """Calculate Brier score."""
    if is_correct:
        return -((1 - probability) ** 2)
    else:
        return -(probability ** 2)

def calculate_mcq_generation_scores(data: List[Dict[str, Any]], generation_idx: int) -> Tuple[List[float], float]:
    """Calculate Brier scores and accuracy for MCQ data (MMLU Pro, GPQA)."""
    brier_scores = []
    correct_count = 0
    total_count = 0
    
    for item in data:
        if "extracted_answer" not in item or "answer" not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        correct_answer = item.get("answer", "")
        
        if generation_idx >= len(extracted_answers):
            continue
            
        generation_answer = extracted_answers[generation_idx]
        
        if isinstance(generation_answer, dict):
            brier_score = 0
            any_correct = False
            
            for answer_option, probability in generation_answer.items():
                if answer_option and probability is not None:
                    if not isinstance(probability, float) or probability < 0 or probability > 1:
                        continue
                    is_correct = (answer_option == correct_answer)
                    if is_correct:
                        any_correct = True
                    brier_score += calculate_brier_score(probability, is_correct)
            
            if not any_correct:
                brier_score -= 1  # Penalize for not having correct answer
            
            brier_scores.append(brier_score)
            
            # Check if any answer matches correct answer for accuracy
            if any_correct:
                correct_count += 1
            total_count += 1
        
        elif isinstance(generation_answer, str):
            is_correct = (generation_answer == correct_answer)
            if is_correct:
                brier_score = 0.0
                correct_count += 1
            else:
                brier_score = -2.0
            brier_scores.append(brier_score)
            total_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return brier_scores, accuracy

def calculate_judge_generation_scores(data: List[Dict[str, Any]], generation_idx: int, judge_field: str) -> Tuple[List[float], float]:
    """Calculate Brier scores and accuracy for judge-scored data (SimpleQA)."""
    brier_scores = []
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
            brier_score = 0
            
            for answer_option, probability in generation_answer.items():
                if not answer_option or probability is None:
                    continue
                    
                if answer_option in generation_scores:
                    is_correct = (int(generation_scores[answer_option]) == 1)
                    if is_correct:
                        any_correct = True
                    brier_score += calculate_brier_score(probability, is_correct)
            
            if not any_correct:
                brier_score -= 1
            
            brier_scores.append(brier_score)
            
            if any_correct:
                correct_count += 1
            total_count += 1
        
        elif isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
            is_correct = (int(generation_scores) == 1)
            if is_correct:
                brier_score = 0
                correct_count += 1
            else:
                brier_score = -2
            brier_scores.append(brier_score)
            total_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return brier_scores, accuracy

def calculate_model_statistics(data: List[Dict[str, Any]], num_generations: int, 
                             dataset_type: str, judge_field: str = None) -> Tuple[float, float, float, float]:
    """Calculate mean Brier score and accuracy with standard errors."""
    all_brier_means = []
    all_accuracies = []
    
    for gen_idx in range(num_generations):
        if dataset_type == 'mcq':
            brier_scores, accuracy = calculate_mcq_generation_scores(data, gen_idx)
        else:  # judge
            brier_scores, accuracy = calculate_judge_generation_scores(data, gen_idx, judge_field)
        
        if brier_scores:
            # Shift scores to positive range
            brier_scores = [score + 1 for score in brier_scores]
            generation_brier_mean = np.mean(brier_scores)
            all_brier_means.append(generation_brier_mean)
        
        all_accuracies.append(accuracy * 100.0)  # Convert to percentage
    
    # Calculate means and standard errors
    mean_brier = np.mean(all_brier_means) if all_brier_means else 0.0
    brier_se = np.std(all_brier_means, ddof=1) / np.sqrt(len(all_brier_means)) if len(all_brier_means) > 1 else 0.0
    
    mean_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
    accuracy_se = np.std(all_accuracies, ddof=1) / np.sqrt(len(all_accuracies)) if len(all_accuracies) > 1 else 0.0
    
    return mean_brier, brier_se, mean_accuracy, accuracy_se

def process_dataset(dataset_name: str, config: dict, judge: str) -> Dict[str, Dict[str, float]]:
    """Process a single dataset and return model statistics."""
    print(f"\nProcessing {dataset_name}...")
    
    dataset_path = config['path']
    dataset_type = config['type']
    judge_field = f"score_{judge}" if config['judge_field'] else None
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return {}
    
    # Get all JSONL files
    jsonl_files = [f for f in os.listdir(dataset_path) 
                   if f.endswith('.jsonl') and any(model in f for model in ['Qwen3-4B', 'Qwen3-8B'])]
    
    print(f"Found {len(jsonl_files)} relevant files")
    
    model_stats = {}
    
    for filename in jsonl_files:
        file_path = os.path.join(dataset_path, filename)
        print(f"  Processing: {filename}")
        
        # Extract model info
        model_name, num_generations = extract_model_info_from_filename(filename, dataset_type)
        
        # Only include our target models
        if not any(target in model_name for target in ['Qwen3-4B', 'Qwen3-8B']):
            continue
        
        # Load data
        data = load_jsonl_file(file_path)
        print(f"    Loaded {len(data)} samples for {model_name}")
        
        # Calculate statistics
        if dataset_type == 'mcq':
            mean_brier, brier_se, mean_accuracy, accuracy_se = calculate_model_statistics(
                data, num_generations, dataset_type)
        else:
            # Check if judge field exists
            if not any(judge_field in item for item in data):
                print(f"    Warning: {judge_field} not found, skipping")
                continue
            mean_brier, brier_se, mean_accuracy, accuracy_se = calculate_model_statistics(
                data, num_generations, dataset_type, judge_field)
        
        model_stats[model_name] = {
            'mean_brier': mean_brier,
            'brier_se': brier_se,
            'mean_accuracy': mean_accuracy,
            'accuracy_se': accuracy_se,
            'num_samples': len(data)
        }
        
        print(f"    {model_name}: Brier = {mean_brier:.4f} ± {brier_se:.4f}, Accuracy = {mean_accuracy:.2f}% ± {accuracy_se:.2f}%")
    
    return model_stats

def calculate_differences(all_stats: Dict[str, Dict[str, Dict[str, float]]], model_size: str) -> Dict[str, Dict[str, float]]:
    """Calculate differences (sft-rl - base) for each dataset."""
    differences = {}
    
    base_model = f'Qwen3-{model_size}'
    sft_model = f'Qwen3-{model_size}-sft-rl'
    
    for dataset_name in DATASETS.keys():
        if dataset_name not in all_stats:
            continue
            
        stats = all_stats[dataset_name]
        
        if base_model not in stats or sft_model not in stats:
            print(f"Warning: Missing models for {dataset_name} ({model_size})")
            continue
        
        base_stats = stats[base_model]
        sft_stats = stats[sft_model]
        
        # Calculate differences: sft-rl - base
        diff_accuracy = sft_stats['mean_accuracy'] - base_stats['mean_accuracy']
        diff_brier = sft_stats['mean_brier'] - base_stats['mean_brier']
        
        # Calculate standard errors for differences (assuming independence)
        diff_accuracy_se = np.sqrt(sft_stats['accuracy_se']**2 + base_stats['accuracy_se']**2)
        diff_brier_se = np.sqrt(sft_stats['brier_se']**2 + base_stats['brier_se']**2)
        
        differences[dataset_name] = {
            'diff_accuracy': diff_accuracy,
            'diff_accuracy_se': diff_accuracy_se,
            'diff_brier': diff_brier,
            'diff_brier_se': diff_brier_se,
        }
        
        print(f"{dataset_name} ({model_size}): ΔAcc = {diff_accuracy:.2f}% ± {diff_accuracy_se:.2f}%, "
              f"ΔBrier = {diff_brier:.4f} ± {diff_brier_se:.4f}")
    
    return differences

def create_scatter_plot(differences_4b: Dict[str, Dict[str, float]], 
                       differences_8b: Dict[str, Dict[str, float]],
                       output_dir: str, model_size: str):
    """Create scatter plot of change in accuracy vs change in brier."""
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Determine which differences to plot
    if model_size == '4B':
        all_differences = {'4B': differences_4b}
    elif model_size == '8B':
        all_differences = {'8B': differences_8b}
    else:  # both
        all_differences = {'4B': differences_4b, '8B': differences_8b}
    
    # Plot each dataset
    for size_key, differences in all_differences.items():
        for dataset_name, diff_data in differences.items():
            config = DATASETS[dataset_name]
            x = diff_data['diff_brier']
            y = diff_data['diff_accuracy']
            xerr = diff_data['diff_brier_se']
            yerr = diff_data['diff_accuracy_se']
            
            # Use different markers for each benchmark
            marker = config['marker']
            color = config['color']
            
            # Adjust marker style for different model sizes if plotting both
            if model_size == 'both':
                if size_key == '4B':
                    markerfacecolor = color
                    markeredgecolor = 'white'
                    markersize = 300
                    alpha = 0.8
                    label = f"{dataset_name} (4B)"
                else:  # 8B
                    markerfacecolor = 'none'
                    markeredgecolor = color
                    markersize = 350
                    alpha = 0.9
                    label = f"{dataset_name} (8B)"
            else:
                markerfacecolor = color
                markeredgecolor = 'white'
                markersize = 50
                alpha = 1 # 0.8
                label = dataset_name
            
            # Plot with error bars
            ax.errorbar(
                x, y,
                xerr=xerr, yerr=yerr,
                fmt=marker,
                markersize=markersize,
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor,
                markeredgewidth=3,
                ecolor=color,
                elinewidth=2.5,
                capsize=8,
                alpha=alpha,
                label=label
            )
            
            # Annotate with dataset name
            ax.annotate(
                dataset_name,
                (x, y),
                textcoords="offset points",
                xytext=(50, -50),
                ha='center', va='bottom', 
                fontsize=24, fontweight='bold', color=color,
                # bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, lw=2, alpha=0.8)
            )
    
    # Add reference lines at zero
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set labels
    ax.set_xlabel('Change in Brier Score', fontsize=28, fontweight='bold')
    ax.set_ylabel('Change in Accuracy (%)', fontsize=28, fontweight='bold', labelpad=14)
    
    # Dynamic limits with padding
    all_xs = []
    all_ys = []
    for differences in all_differences.values():
        for diff_data in differences.values():
            brier_se = diff_data['diff_brier_se']
            accuracy_se = diff_data['diff_accuracy_se']
            
            all_xs.append(diff_data['diff_brier'] + brier_se)
            all_ys.append(diff_data['diff_accuracy'] + accuracy_se)
            all_xs.append(diff_data['diff_brier'] - brier_se)
            all_ys.append(diff_data['diff_accuracy'] - accuracy_se)
    
    if all_xs and all_ys:
        minx, maxx = float(min(all_xs)), float(max(all_xs))
        miny, maxy = float(min(all_ys)), float(max(all_ys))
        
        if np.isclose(maxx - minx, 0.0):
            minx, maxx = minx - 0.01, maxx + 0.01
        if np.isclose(maxy - miny, 0.0):
            miny, maxy = miny - 1.0, maxy + 1.0
        
        xpad = 0.15 * (maxx - minx) if maxx != minx else 0.01
        ypad = 0.15 * (maxy - miny) if maxy != miny else 1.0
        
        ax.set_xlim(minx - xpad, maxx + xpad)
        ax.set_ylim(miny - ypad, maxy + ypad)
        
        # hardcoded limits
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-4, 4)
    
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.tick_params(axis='both', labelsize=30, length=6, width=1.2)
    
    # Add legend if plotting both model sizes
    if model_size == 'both':
        ax.legend(fontsize=20, loc='best', frameon=True, fancybox=True, ncol=2)
    
    plt.tight_layout()
    
    # Save plots
    suffix = model_size.lower() if model_size != 'both' else 'both'
    output_path = os.path.join(output_dir, f'scatter_change_{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {output_path}")
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Processing datasets for cross-benchmark comparison...")
    
    # Process all datasets
    all_stats = {}
    for dataset_name, config in DATASETS.items():
        stats = process_dataset(dataset_name, config, args.judge)
        all_stats[dataset_name] = stats
    
    # Calculate differences for both model sizes
    print(f"\n{'='*60}")
    print("CALCULATING DIFFERENCES (sft-rl - base)")
    print(f"{'='*60}")
    
    differences_4b = calculate_differences(all_stats, '4B')
    differences_8b = calculate_differences(all_stats, '8B')
    
    # Create scatter plot
    print(f"\nCreating scatter plot...")
    create_scatter_plot(differences_4b, differences_8b, args.output_dir, args.model_size)
    
    print(f"\nPlots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

