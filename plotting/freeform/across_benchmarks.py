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
        'judge_field': 'score_Llama_4_Scout'
    },
    'GPQA': {
        'path': '/fast/nchandak/forecasting/evals/gpqa/gpqa_diamond', 
        'type': 'mcq',
        'judge_field': None
    },
    'MMLU Pro': {
        'path': '/fast/nchandak/forecasting/evals/mmlu_pro/mmlu_pro',
        'type': 'mcq',
        'judge_field': None
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot model performance across multiple benchmarks")
    parser.add_argument("--output-dir", type=str, default="plots/across_benchmarks",
                       help="Output directory for plots")
    parser.add_argument("--judge", type=str, default="Llama_4_Scout",
                       help="Judge model name for SimpleQA evaluation")
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

def family_color_map() -> Dict[str, str]:
    # Vibrant palette
    return {
        'Qwen': '#1f77b4',     # blue
        'Meta': '#ff7f0e',    # orange
        'DeepSeek': '#2ca02c', # green
        'Claude': '#9467bd',   # purple
        'OpenAI': '#d62728',   # red
        'Grok': '#17becf',     # cyan
        'Kimi': '#e377c2',     # pink
        'Gemini': '#bcbd22',   # olive
        'Other': '#7f7f7f',    # gray
        'Grok-3-Mini Distill': '#7f7f7f',    # gray
        'Ours': '#9467bd',   # purple
    }
    
    
def get_model_color(model_name):
    """Get color for model based on type."""
    if 'sft-rl' in model_name.lower():
        return '#9467bd'  # purple for OpenForecaster
    else:
        return '#1f77b4'  # blue for base Qwen
    
    
    if 'sft-rl' in model_name.lower():
        return '#9467bd'  # purple for 'Ours'/OpenForecaster
    else:
        return '#1f77b4'  # blue for base Qwen

def get_model_pattern(model_name):
    """Get hatching pattern based on model size."""
    if '4b' in model_name.lower():
        return '///'  # diagonal stripes for 4B
    else:
        return None  # solid for 8B

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
        print(f"    DEBUG: Added {model_name} to {dataset_name} stats")
    
    return model_stats

def create_comparison_plots(all_stats: Dict[str, Dict[str, Dict[str, float]]], output_dir: str):
    """Create comparison plots across datasets."""
    
    # Define model order and styling
    target_models = ['Qwen3-4B', 'Qwen3-4B-sft-rl', 'Qwen3-8B', 'Qwen3-8B-sft-rl']
    target_models = ['Qwen3-8B', 'Qwen3-8B-sft-rl']
    datasets = list(all_stats.keys())
    
    # Create accuracy plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    x = np.arange(len(datasets))
    width = 0.3
    
    # Store all model data for improvement calculations
    all_model_data = {}
    for i, model in enumerate(target_models):
        values = []
        errors = []
        
        for dataset in datasets:
            if model in all_stats[dataset]:
                values.append(all_stats[dataset][model]['mean_accuracy'])
                errors.append(all_stats[dataset][model]['accuracy_se'])
            else:
                values.append(0)
                errors.append(0)
        
        all_model_data[model] = {'values': values, 'errors': errors}
        
        color = get_model_color(model)
        pattern = get_model_pattern(model)
        label = print_names.get(model, model)
        
        bars = ax.bar(x + i * width, values, width, yerr=errors, 
                     label=label, color=color, alpha=0.8, 
                     hatch=pattern, edgecolor='black', linewidth=1.5,
                     capsize=5)
    
    # Add improvement labels for sft-rl models
    for i, dataset in enumerate(datasets):
        print(f"DEBUG: Processing dataset {dataset} (index {i})")
        print(f"  Available models: {list(all_model_data.keys())}")
        
        # Calculate improvements for 4B sft-rl vs 4B base
        if ('Qwen3-4B' in all_model_data and 'Qwen3-4B-sft-rl' in all_model_data):
            base_val = all_model_data['Qwen3-4B']['values'][i]
            sft_val = all_model_data['Qwen3-4B-sft-rl']['values'][i]
            print(f"  4B: base={base_val:.2f}, sft={sft_val:.2f}")
            
            if base_val > 0 and sft_val > 0:
                improvement = ((sft_val - base_val) / base_val) * 100
                sft_bar_x = i + 1 * width  # 4B-sft-rl is index 1
                sft_bar_height = sft_val + all_model_data['Qwen3-4B-sft-rl']['errors'][i]
                
                print(f"  4B improvement: {improvement:.2f}%")
                if abs(improvement) > 0.1:  # Only show if improvement is meaningful
                    ax.text(sft_bar_x, max(0, sft_bar_height) + max(all_model_data['Qwen3-4B-sft-rl']['values']) * 0.02,
                           f'{improvement:+.1f}%', ha='center', va='bottom', 
                           fontsize=22, fontweight='bold', color='red' if improvement < 0 else 'green')
        
        # Calculate improvements for 8B sft-rl vs 8B base  
        if ('Qwen3-8B' in all_model_data and 'Qwen3-8B-sft-rl' in all_model_data):
            base_val = all_model_data['Qwen3-8B']['values'][i]
            sft_val = all_model_data['Qwen3-8B-sft-rl']['values'][i]
            print(f"  8B: base={base_val:.2f}, sft={sft_val:.2f}")
            
            # if base_val > 0 and sft_val > 0:
            if True:
                improvement = ((sft_val - base_val) / abs(base_val)) * 100
                sft_bar_x = i + 1 * width  # 8B-sft-rl is index 3
                sft_bar_height = sft_val + all_model_data['Qwen3-8B-sft-rl']['errors'][i]
                
                print(f"  8B improvement: {improvement:.2f}%")
                if abs(improvement) > 0.1:  # Only show if improvement is meaningful
                    text = f'+ {improvement:.1f}\%' if improvement > 0 else f'- {abs(improvement):.1f}\%'
                    
                    y_pos = max(0, sft_bar_height) + max(all_model_data['Qwen3-8B-sft-rl']['values']) * 0.01
                    import matplotlib.patheffects as patheffects
                    ax.text(
                        sft_bar_x,
                        y_pos,
                        text,
                        ha='center',
                        va='bottom',
                        fontsize=24,
                        fontweight='bold',
                        color='red' if improvement < 0 else 'green',
                        path_effects=[patheffects.withStroke(linewidth=1.5)]
                    )
                    
                    # ax.text(sft_bar_x, max(0, sft_bar_height) + max(all_model_data['Qwen3-8B-sft-rl']['values']) * 0.02,
                    #        f'{improvement:+.1f}%', ha='center', va='bottom',
                    #        fontsize=22, fontweight='bold', color='red' if improvement < 0 else 'green')
    
    # ax.set_xlabel('Dataset', fontsize=32, fontweight='bold', labelpad=15)
    ax.set_ylabel('Accuracy (\%)', fontsize=32, fontweight='bold', labelpad=15)
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels(datasets, fontsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.legend(fontsize=30, loc='best', frameon=True, fancybox=True, ncol=1,   borderaxespad=1)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.pdf'), dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison plot saved to {output_dir}")
    plt.close()
    
    # Create Brier score plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Store all model data for improvement calculations
    all_brier_data = {}
    for i, model in enumerate(target_models):
        values = []
        errors = []
        
        for dataset in datasets:
            if model in all_stats[dataset]:
                values.append(all_stats[dataset][model]['mean_brier'])
                errors.append(all_stats[dataset][model]['brier_se'])
            else:
                values.append(0)
                errors.append(0)
        
        all_brier_data[model] = {'values': values, 'errors': errors}
        
        color = get_model_color(model)
        pattern = get_model_pattern(model)
        label = print_names.get(model, model)
        
        bars = ax.bar(x + i * width, values, width, yerr=errors, 
                     label=label, color=color, alpha=0.8,
                     hatch=pattern, edgecolor='black', linewidth=1.5,
                     capsize=5)
    
    # Add improvement labels for sft-rl models (note: higher Brier is better)
    for i, dataset in enumerate(datasets):
        print(f"DEBUG BRIER: Processing dataset {dataset} (index {i})")
        print(f"  Available models: {list(all_brier_data.keys())}")
        
        # Calculate improvements for 4B sft-rl vs 4B base
        if ('Qwen3-4B' in all_brier_data and 'Qwen3-4B-sft-rl' in all_brier_data):
            base_val = all_brier_data['Qwen3-4B']['values'][i]
            sft_val = all_brier_data['Qwen3-4B-sft-rl']['values'][i]
            print(f"  4B Brier: base={base_val:.3f}, sft={sft_val:.3f}")
            
            # if base_val > 0 and sft_val > 0:
            if True:
                improvement = abs(((sft_val - base_val) / base_val) * 100)
                sft_bar_x = i + 1 * width  # 4B-sft-rl is index 1
                sft_bar_height = sft_val + all_brier_data['Qwen3-4B-sft-rl']['errors'][i]
                
                print(f"  4B Brier improvement: {improvement:.2f}%")
                if abs(improvement) > 0.5:  # Only show if improvement is meaningful
                    ax.text(sft_bar_x, max(0, sft_bar_height) + max(all_brier_data['Qwen3-4B-sft-rl']['values']) * 0.02,
                           f'{improvement:+.1f}%', ha='center', va='bottom', 
                           fontsize=22, fontweight='bold', color='red' if improvement < 0 else 'green')
        
        # Calculate improvements for 8B sft-rl vs 8B base  
        if ('Qwen3-8B' in all_brier_data and 'Qwen3-8B-sft-rl' in all_brier_data):
            base_val = all_brier_data['Qwen3-8B']['values'][i]
            sft_val = all_brier_data['Qwen3-8B-sft-rl']['values'][i]
            print(f"  8B Brier: base={base_val:.3f}, sft={sft_val:.3f}")
            
            # if base_val > 0 and sft_val > 0:
            if True:
                improvement = ((sft_val - base_val) / abs(base_val)) * 100
                sft_bar_x = i + 1 * width  # 8B-sft-rl is index 3
                sft_bar_height = sft_val + all_brier_data['Qwen3-8B-sft-rl']['errors'][i]
                
                print(f"  8B Brier improvement: {improvement:.2f}%")
                if abs(improvement) > 0.5:  # Only show if improvement is meaningful
                    y_pos = max(0, sft_bar_height) + max(all_brier_data['Qwen3-8B-sft-rl']['values']) * 0.02
                    # if sft_val < 0:
                    #     y_pos = sft_bar_height - max(all_brier_data['Qwen3-8B-sft-rl']['values']) * 0.02
                    #     print(sft_val, base_val, improvement)
                        
                    import matplotlib.patheffects as patheffects
                    ax.text(
                        sft_bar_x,
                        y_pos,
                        f'+ {improvement:.1f}\%',
                        ha='center',
                        va='bottom',
                        fontsize=24,
                        fontweight='bold',
                        color='red' if improvement < 0 else 'green',
                        path_effects=[patheffects.withStroke(linewidth=1.5)]
                    )
                    
    # ax.set_xlabel('Dataset', fontsize=32, fontweight='bold', labelpad=15)
    ax.set_ylabel('Brier Score \ (higher is better)', fontsize=32, fontweight='bold', labelpad=2)
    ax.set_xticks(x + width * 0.3)
    ax.set_xticklabels(datasets, fontsize=28)
    ax.tick_params(axis='y', labelsize=28)
    # ax.legend(fontsize=28, loc='upper right', frameon=True, fancybox=True, ncol=2, bbox_to_anchor=(1.05, 1), borderaxespad=1)
    # ax.legend(fontsize=24, loc='upper right', frameon=True, fancybox=True, ncol=2, borderaxespad=1)
    ax.legend(fontsize=30, loc='best', frameon=True, fancybox=True, ncol=1,   borderaxespad=1)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brier_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'brier_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig("poster_plot2.png", dpi=1200, transparent=True, bbox_inches="tight")
    print(f"Brier score comparison plot saved to {output_dir}")
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
    
    # Create comparison plots
    print(f"\nCreating comparison plots...")
    create_comparison_plots(all_stats, args.output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CROSS-BENCHMARK PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    target_models = ['Qwen3-4B', 'Qwen3-8B', 'Qwen3-4B-sft-rl', 'Qwen3-8B-sft-rl']
    
    for model in target_models:
        print(f"\n{print_names.get(model, model)}:")
        print("-" * 40)
        for dataset_name in DATASETS.keys():
            if model in all_stats[dataset_name]:
                stats = all_stats[dataset_name][model]
                print(f"  {dataset_name:12}: Acc = {stats['mean_accuracy']:5.1f}% ± {stats['accuracy_se']:4.1f}%, "
                      f"Brier = {stats['mean_brier']:5.3f} ± {stats['brier_se']:5.3f}")
            else:
                print(f"  {dataset_name:12}: No data available")
    
    print(f"\nPlots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
