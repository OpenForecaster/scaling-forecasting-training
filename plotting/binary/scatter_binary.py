#!/usr/bin/env python3
import os
import json
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

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
    'Qwen3-4B-sft-rl': '\\texttt{OpenForecaster}-4B',
    'Qwen3-8B-sft-rl': '\\texttt{OpenForecaster}-8B',
    'Qwen3-4B-RL': '\\texttt{OpenForecaster}-4B',
    'Qwen3-8B-RL': '\\texttt{OpenForecaster}-8B',
    'gemini-2.5-pro-preview': 'Gemini 2.5 Pro',
    'gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
    'qwen3-235b-a22b': 'Qwen3-235B-A22B',
    'qwen3-235b-a22b-07-25': 'Qwen3-235B-A22B',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Scatter plot of Brier Score (y) vs Accuracy (x) for binary forecasting")
    parser.add_argument("--input_dir", type=str,
                        default="/fast/nchandak/forecasting/evals/binary/with_retrieval/metaculus_metaculusFromMayTillNov_30.jsonl",
                        help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/arxiv",
                        help="Output directory for the scatter plot")
    return parser.parse_args()


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    try:
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


def extract_model_info_from_filename(filename: str) -> Tuple[str, int]:
    """
    Extract model name and number of generations from filename.
    Returns: (model_name, num_generations)
    """
    name_without_ext = filename.replace('.jsonl', '')

    model_match = re.match(r'([^_]+(?:_[^_]*?)?(?:-\d+\.?\d*[bB])?)', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = name_without_ext.split('_')[0]

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
        Brier score (range: [0, 1], lower is better, but we negate for consistency)
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
        if "extracted_answer" not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        responses = item.get("response", [])
        resolution = item.get("resolution", None)
        
        if (not resolution or resolution < 0 or resolution > 1) and item.get("answer", None):
            if "yes" in item.get("answer", "").lower():
                resolution = 1
            elif "no" in item.get("answer", "").lower():
                resolution = 0
            else:
                resolution = None
        
        # Skip if generation_idx is out of bounds or resolution is missing
        if generation_idx >= len(extracted_answers) or resolution is None:
            continue
            
        generation_answer = extracted_answers[generation_idx]
        response = responses[generation_idx] if generation_idx < len(responses) else ""
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
                    if model_prob < 0.5:
                        model_prob = 1 - model_prob
                    
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
            # Convert to negative Brier (higher is better) for consistency with plotting
            generation_brier_mean = -np.mean(generation_brier_scores)
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


def determine_family(model_name: str) -> str:
    name = model_name.lower()
    if 'qwen' in name:
        if re.match(r'^[Qq]wen3-1\.7[bB]-', model_name) and 'rl' in model_name.lower():
            return 'Ours'
        if re.match(r'^[Qq]wen3-4[bB]-', model_name) and 'rl' in model_name.lower():
            return 'Ours'
        if re.match(r'^[Qq]wen3-8[bB]-', model_name) and 'rl' in model_name.lower():
            return 'Ours'
        
        if 'sft' in model_name.lower():
            return 'Grok-3-Mini Distill'
        
        return 'Qwen'
    
    if 'llama' in name:
        return 'Llama'
    if 'deepseek' in name:
        return 'DeepSeek'
    if 'claude' in name:
        return 'Claude'
    if 'gpt' in name or name.startswith('o4') or name.startswith('o3'):
        return 'OpenAI'
    if 'grok' in name:
        return 'Grok'
    if 'kimi' in name:
        return 'Kimi'
    if 'gemini' in name:
        return 'Gemini'
    return 'Other'


def family_color_map() -> Dict[str, str]:
    # Vibrant palette
    return {
        'Qwen': '#1f77b4',     # blue
        'Llama': '#ff7f0e',    # orange
        'DeepSeek': '#2ca02c', # green
        'Claude': '#9467bd',   # purple
        'OpenAI': '#d62728',   # red
        'Grok': '#17becf',     # cyan
        'Kimi': '#e377c2',     # pink
        'Gemini': '#bcbd22',   # olive
        'Other': '#7f7f7f',    # gray  
        'Ours': '#9467bd',     # purple
    }


def qwen_trained_marker(model_name: str) -> str:
    # Use special shapes for trained Qwen3-XB- models
    if re.match(r'^[Qq]wen3-1\.7[bB]-', model_name):
        return '*'
    if re.match(r'^[Qq]wen3-4[bB]-', model_name):
        return '*'
    if re.match(r'^[Qq]wen3-8[bB]-', model_name):
        return '*'
    return 'o'


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def compute_model_metrics(input_dir: str) -> Dict[str, Dict[str, Any]]:
    model_metrics: Dict[str, Dict[str, Any]] = {}

    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.jsonl')]

    print(f"Found {len(jsonl_files)} JSONL files")

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        model_name, num_generations = extract_model_info_from_filename(filename)
        model_key = f"{model_name}"

        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples")

        mean_brier, brier_se, mean_acc, acc_se = calculate_model_binary_statistics(data, num_generations)

        model_metrics[model_key] = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'brier_se': brier_se,
            'mean_accuracy': mean_acc,
            'acc_se': acc_se,
            'num_samples': len(data),
            'num_generations': num_generations,
        }
        
        print(f"  Mean Brier = {mean_brier:.4f} ± {brier_se:.4f}, Mean Accuracy = {mean_acc:.2f}% ± {acc_se:.2f}%")

    return model_metrics


def plot_scatter(metrics: Dict[str, Dict[str, Any]], output_path: str, title: str) -> None:
    if not metrics:
        print("No metrics to plot")
        return

    families = family_color_map()

    # Prepare plot (bigger canvas)
    fig, ax = plt.subplots(figsize=(12, 12))

    handles = {}
    
    # Plot each model
    for model_key, info in metrics.items():
        x = info['mean_accuracy']
        y = info['mean_brier']
        xerr = info.get('acc_se', 0)
        yerr = info.get('brier_se', 0)
        print(f"{model_key}: x={x:.2f}, y={y:.4f}")
        print(f"  errors: xerr={xerr:.2f}, yerr={yerr:.4f}")
        print(f"--------------------------------")
        fam = determine_family(model_key)
        color = families.get(fam, families['Other'])
        marker = qwen_trained_marker(model_key)
        label = print_names.get(model_key, model_key)

        marker_size = 26
        if "rl" in model_key.lower():
            marker_size = 36
            if "8b" in model_key.lower():
                marker_size = 40

        # Plot with error bars
        sc = ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt=marker,
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=2,
            ecolor=color,
            elinewidth=2,
            capsize=8,
            alpha=0.95,
            label=None
        )

        # Annotate above the point with dynamic xytext based on label length
        label_len = len(label)
        x_offset = -5 - (label_len * 1.5)
        y_offset = 30
        
        if model_key.lower() == "qwen3-8b" or model_key.lower() == "qwen3-4b" or model_key.lower() == "qwen3-1.7b":
            x_offset = - 70 - label_len * 1.5
            y_offset = 1
            
            if "8b" in model_key.lower():
                x_offset += 160
                y_offset = -40
                
            if "4b" in model_key.lower():
                x_offset += 160
                y_offset -= 50
            
        elif "rl" in model_key.lower():
            x_offset = 10
            y_offset = -50
            if "4b" in model_key.lower():
                y_offset -= 40
                x_offset -= 20
            if "8b" in model_key.lower():
                x_offset = 80
                y_offset = 30
            
        else:
            x_offset = -20 - (label_len * 7)
            y_offset = 1
        
        if "gpt-oss-120b" in model_key.lower():
            y_offset = -50
            
        if "gpt-oss-20b" in model_key.lower():
            y_offset = -50
            x_offset = -90
            
        if "maverick" in model_key.lower():
            y_offset -= 60
            x_offset += 260
            
        if "grok-3-mini" in model_key.lower():
            y_offset = -40
            x_offset = -100
            
        if "v3" in model_key.lower():
            # y_offset = -45
            # x_offset = -90
            y_offset = 15
            x_offset = 100
            
        if "r1" in model_key.lower():
            y_offset = -70
            x_offset = 0
            
        if "235" in model_key.lower():
            y_offset = 30
            x_offset = -100
        
        ax.annotate(label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(x_offset, y_offset),
                    ha='center', va='bottom', fontsize=30, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))

        # For legend by family, store one handle per family
        if fam not in handles:
            handles[fam] = sc[0] if isinstance(sc, tuple) else sc

    ax.set_xlabel('Accuracy (\%) ($\\uparrow$)', fontsize=28, fontweight='bold')
    ax.set_ylabel('Binary Brier Score ($\\uparrow$)', fontsize=28, fontweight='bold', labelpad=14)

    # half the number of xticks (keep only the even ones)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    
    # Dynamic limits with small padding
    xs = [info['mean_accuracy'] for info in metrics.values()]
    ys = [info['mean_brier'] for info in metrics.values()]
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))
    
    if np.isclose(maxx - minx, 0.0):
        minx, maxx = minx - 0.05, maxx + 0.05
    if np.isclose(maxy - miny, 0.0):
        miny, maxy = miny - 0.5, maxy + 0.5
    xpad = 0.1 * (maxx - minx)
    ypad = 0.1 * (maxy - miny)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad * 1.5)
    

    ax.grid(True, alpha=0.35, linestyle='--')
    ax.tick_params(axis='both', labelsize=30, length=6, width=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to {output_path}")
    # also save as pdf
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')

    plt.close(fig)


def main():
    args = parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    output_path = os.path.join(args.output_dir, f"scatter_binary_brier_accuracy_{dataset_suffix}.png")

    metrics = compute_model_metrics(args.input_dir)

    title = f"Binary Forecasting: Accuracy vs Brier Score | Dataset: {dataset_suffix}"
    plot_scatter(metrics, output_path, title)


if __name__ == "__main__":
    main()

