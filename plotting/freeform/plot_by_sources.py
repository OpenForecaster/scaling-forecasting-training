#!/usr/bin/env python3
import os
import json
import argparse
import re
from typing import List, Dict, Any, Tuple, Set
from urllib.parse import urlparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

mpl.style.use(['science'])

# Custom display names for models (reused)
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


# Vibrant, contrasting color palette (cycled as needed)
VIBRANT_COLORS = [
    # '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#17becf',
    '#66c2a5', '#8da0cb',  '#fc8d62', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Brier and Accuracy per news source (domain) for freeform forecasting evaluation")
    parser.add_argument("--input_dir", type=str,
                        default="/fast/nchandak/forecasting/evals/freeform/manual/news5-retrieval_1000/news5-monthly",
                        help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/by_news_source",
                        help="Output directory for per-source plots")
    parser.add_argument("--judge", type=str, default="Qwen3_4B",
                        help="Judge model name for score field, e.g., Qwen3_4B")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum number of samples for a domain to be plotted")
    parser.add_argument("--original-questions", type=str, default=None,
                        help="Optional path to original questions JSONL file to filter by qid")
    return parser.parse_args()


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def load_qids_from_original_questions(file_path: str) -> Set[str]:
    """
    Load qids from the original questions file.
    
    Args:
        file_path: Path to the original questions JSONL file
        
    Returns:
        Set of qids from the original questions file
    """
    qids = set()
    data = load_jsonl_file(file_path)
    
    for item in data:
        if "human_filter" not in item or item.get("human_filter") == 0:
            continue 
        
        if "qid" in item: 
            qids.add(str(item["qid"]))
    
    print(f"Loaded {len(qids)} unique qids from {file_path}")
    return qids


def filter_data_by_qids(data: List[Dict[str, Any]], qids: Set[str]) -> List[Dict[str, Any]]:
    """
    Filter evaluation data to only include rows where qid matches or idx == qid - 1.
    
    Args:
        data: List of evaluation entries
        qids: Set of qids to filter by
        
    Returns:
        Filtered list of evaluation entries
    """
    filtered_data = []
    
    for item in data:
        # Check if qid matches
        if "qid" in item and str(item["qid"]) in qids:
            filtered_data.append(item)
            continue
        
        # Check if idx == qid - 1 for any qid in the set
        if "idx" in item:
            idx = item["idx"]
            # idx should equal qid - 1, so qid should equal idx + 1
            req = str(idx + 1)
            if req in qids:
                filtered_data.append(item)
                continue
    
    return filtered_data


def extract_model_info_from_filename(filename: str) -> Tuple[str, int]:
    """
    Extract model name and number of generations from filename.
    Expected format: ModelName_eval_size_N_generations_M.jsonl
    Returns: (model_name, num_generations)
    """
    name_without_ext = filename.replace('.jsonl', '')

    multiple = "_list" if "multiple" in filename else ""

    model_match = re.match(r'([^_]+(?:_[^_]*?)?(?:-\d+\.?\d*[bB])?)', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = name_without_ext.split('_')[0]

    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1

    if model_name.endswith('_'):
        model_name = model_name[:-1]

    model_name = f"{model_name}{multiple}"

    return model_name, num_generations


def extract_domain(url: str) -> str:
    if not url:
        return "unknown"
    parsed = urlparse(url if (url.startswith('http://') or url.startswith('https://')) else f"http://{url}")
    host = parsed.netloc
    if not host:
        return "unknown"
    host = host.split(':')[0]
    if host.startswith('www.'):
        host = host[4:]
    return host.lower()


def calculate_brier_score(probability: float, is_correct: bool) -> float:
    if is_correct:
        return -((1 - probability) ** 2)
    else:
        return -(probability ** 2)


def calculate_generation_brier_scores(data: List[Dict[str, Any]], generation_idx: int, judge_field: str) -> List[float]:
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
                if probability is None:
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
            brier_score = 0 if is_correct else -2
            brier_scores.append(brier_score)
    return brier_scores


def calculate_model_brier_statistics(data: List[Dict[str, Any]], num_generations: int, judge_field: str) -> Tuple[float, float]:
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
    all_generation_accuracies = []
    for gen_idx in range(num_generations):
        generation_accuracy = calculate_generation_accuracy(data, gen_idx, judge_field) * 100.0
        all_generation_accuracies.append(generation_accuracy)
    if not all_generation_accuracies:
        return 0.0, 0.0
    mean_accuracy = np.mean(all_generation_accuracies)
    std_error = np.std(all_generation_accuracies, ddof=1) / np.sqrt(len(all_generation_accuracies)) if len(all_generation_accuracies) > 1 else 0.0
    return mean_accuracy, std_error


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def plot_combined_brier_and_accuracy(
    brier_data: Dict[str, Dict[str, Any]],
    accuracy_data: Dict[str, Dict[str, Any]],
    output_path: str,
    title_prefix: str
) -> None:
    # Use union of models appearing in either metric
    all_model_names = sorted(set(list(brier_data.keys()) + list(accuracy_data.keys())))
    if not all_model_names:
        print("No valid model data found for plotting")
        return

    # Prepare values in consistent order
    brier_scores = [brier_data.get(m, {}).get('mean_brier', 0.0) for m in all_model_names]
    brier_errors = [brier_data.get(m, {}).get('std_error', 0.0) for m in all_model_names]
    acc_scores = [accuracy_data.get(m, {}).get('mean_accuracy', 0.0) for m in all_model_names]
    acc_errors = [accuracy_data.get(m, {}).get('std_error', 0.0) for m in all_model_names]

    # Colors per model
    def model_color(name: str) -> str:
        name_lower = name.lower()
        if 'filtered' in name_lower or 'freeform' in name_lower:
            return 'red'
        if 'checkpoint' in name_lower or 'rl' in name_lower:
            return '#ff7f0e'
        return '#2ca02c'

    bar_colors = [model_color(m) for m in all_model_names]

    # Layout
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Accuracy subplot
    x_positions = np.arange(len(all_model_names))
    lowest = min([s - e for s, e in zip(acc_scores, acc_errors)] + [0])
    highest = max([s + e for s, e in zip(acc_scores, acc_errors)] + [0])
    b0 = axes[0].bar(x_positions, acc_scores, yerr=acc_errors, capsize=5, alpha=0.85, color=bar_colors)
    
    axes[0].set_ylabel('Accuracy (\%)', fontsize=24, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    axes[0].set_ylim(lowest, highest + 5)
    axes[0].tick_params(axis='y', labelsize=22)
    
    for bar, value, error in zip(b0, acc_scores, acc_errors):
        if value > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., value + error + 0.5, f'{value:.1f}',
                         ha='center', va='bottom', fontsize=20, fontweight='bold')

    # axes[0].set_xticks(x_positions)
    # axes[0].set_xticklabels([print_names.get(name, name) for name in all_model_names], rotation=40, ha='right', fontsize=16)
    # increase fontsize of yticks
    # axes[1].tick_params(axis='y', labelsize=22)

    # Brier subplot
    lowest = min([s - e for s, e in zip(brier_scores, brier_errors)] + [0])
    highest = max([s + e for s, e in zip(brier_scores, brier_errors)] + [0])
    b1 = axes[1].bar(x_positions, brier_scores, yerr=brier_errors, capsize=5, alpha=0.85, color=bar_colors)
    axes[1].set_ylabel('Brier Score', fontsize=24, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1].set_ylim(lowest - 0.05, highest + 0.05)
    axes[1].tick_params(axis='y', labelsize=22)

    for bar, value, error in zip(b1, brier_scores, brier_errors):
        if value != 0:
            axes[1].text(bar.get_x() + bar.get_width()/2.,
                         value + (error + 0.01 if value > 0 else -error - 0.05),
                         f'{value:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')

    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([print_names.get(name, name) for name in all_model_names], rotation=40, ha='right', fontsize=22)

    fig.suptitle(f'{title_prefix}', fontsize=24, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")
    plt.close(fig)


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_root = os.path.abspath(args.output_dir)
    judge = args.judge
    min_samples = args.min_samples

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # Load qids from original questions file if provided
    qids = None
    if args.original_questions:
        if not os.path.exists(args.original_questions):
            print(f"Error: Original questions file {args.original_questions} does not exist")
            return
        qids = load_qids_from_original_questions(args.original_questions)

    dataset_suffix = os.path.basename(input_dir.rstrip('/'))
    output_dir = os.path.join(output_root, dataset_suffix)
    os.makedirs(output_dir, exist_ok=True)

    # Collect all domains present in the dataset
    print(f"Scanning JSONL files in: {input_dir}")
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.jsonl')]
    print(f"Found {len(jsonl_files)} JSONL files")

    domains: Set[str] = set()
    file_cache: Dict[str, List[Dict[str, Any]]] = {}

    for file_path in jsonl_files:
        data = load_jsonl_file(file_path)
        # Filter data by qids if provided
        if qids is not None:
            original_len = len(data)
            data = filter_data_by_qids(data, qids)
            print(f"  Filtered {os.path.basename(file_path)}: {original_len} -> {len(data)} rows")
        file_cache[file_path] = data
        for item in data:
            domain = extract_domain(item.get('article_url', ''))
            domains.add(domain)

    if not domains:
        print("No domains found in the provided dataset")
        return

    print(f"Discovered {len(domains)} domains")

    judge_field_template = "score_{}"

    for domain in sorted(domains):
        domain_safe = sanitize_filename(domain)
        print(f"\nProcessing domain: {domain}")
        brier_model_data: Dict[str, Dict[str, Any]] = {}
        accuracy_model_data: Dict[str, Dict[str, Any]] = {}
        total_domain_samples = 0

        for file_path in jsonl_files:
            filename = os.path.basename(file_path)
            model_name, num_generations = extract_model_info_from_filename(filename)
            if "withbinary" in model_name:
                parts = model_name.split("-")
                model_name = "-".join([part for part in parts if "with" not in part and "binary" not in part])
            model_key = f"{model_name}"

            data = file_cache[file_path]
            filtered_data = [item for item in data if extract_domain(item.get('article_url', '')) == domain]
            if not filtered_data:
                continue

            total_domain_samples += len(filtered_data)

            judge_field = judge_field_template.format(judge)
            has_judge_field = any(judge_field in item for item in filtered_data)
            if not has_judge_field:
                continue

            mean_brier, brier_se = calculate_model_brier_statistics(filtered_data, num_generations, judge_field)
            mean_acc, acc_se = calculate_model_accuracy_statistics(filtered_data, num_generations, judge_field)

            brier_model_data[model_key] = {
                'model_name': model_name,
                'mean_brier': mean_brier,
                'std_error': brier_se,
                'num_samples': len(filtered_data),
                'num_generations': num_generations,
            }
            accuracy_model_data[model_key] = {
                'model_name': model_name,
                'mean_accuracy': mean_acc,
                'std_error': acc_se,
                'num_samples': len(filtered_data),
                'num_generations': num_generations,
            }

        if total_domain_samples < min_samples:
            print(f"Skipping {domain} due to insufficient samples ({total_domain_samples} < {min_samples})")
            continue

        if not brier_model_data and not accuracy_model_data:
            print(f"No valid model data for domain: {domain}")
            continue

        combined_output_path = os.path.join(output_dir, f"combined_{domain_safe}_{dataset_suffix}_{judge}.png")
        # title_prefix = f"Source: {domain} | Dataset: {dataset_suffix} | Judge: {judge}"
        title_prefix = f"Source: www.{domain}"
        plot_combined_brier_and_accuracy(brier_model_data, accuracy_model_data, combined_output_path, title_prefix)

    print("\nDone generating per-source plots.")


if __name__ == "__main__":
    main()
