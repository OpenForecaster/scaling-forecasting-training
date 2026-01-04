#!/usr/bin/env python3
import os
import json
import argparse
import re
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
from datetime import datetime, date

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
    'Qwen3-4B-sft-rl': '\\texttt{OpenForecaster}-4B',
    'Qwen3-8B-sft-rl': '\\texttt{OpenForecaster}-8B',
    'Qwen3-4B-RL': '\\texttt{OpenForecaster}-4B',
    'Qwen3-8B-RL': '\\texttt{OpenForecaster}-8B',
    'gemini-2.5-pro-preview': 'Gemini 2.5 Pro',
    'gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
}


NUM_MONTHS = 4
MONTH_LABELS = ["May 2025", "June 2025", "July 2025", "Aug 2025", "Sep 2025", "Oct 2025", "Nov 2025", "Dec 2025"]
MONTH_STARTS = [date(2025, 5, 1), date(2025, 6, 1), date(2025, 7, 1), date(2025, 8, 1), date(2025, 9, 1), date(2025, 10, 1), date(2025, 11, 1), date(2025, 12, 1)]
MONTH_ENDS = [date(2025, 5, 31), date(2025, 6, 30), date(2025, 7, 31), date(2025, 8, 31), date(2025, 9, 30), date(2025, 10, 31), date(2025, 11, 30), date(2025, 12, 31)]

MONTH_LABELS = MONTH_LABELS[:NUM_MONTHS]
MONTH_STARTS = MONTH_STARTS[:NUM_MONTHS]
MONTH_ENDS = MONTH_ENDS[:NUM_MONTHS]

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Brier and Accuracy by resolution month for freeform forecasting evaluation")
    parser.add_argument("--input_dir", type=str,
                        default="/fast/nchandak/forecasting/evals/freeform/manual/news5-retrieval_1000/news5-monthly",
                        help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/by_time",
                        help="Output directory for time-based plots")
    parser.add_argument("--judge", type=str, default="Qwen3_4B",
                        help="Judge model name for score field, e.g., Qwen3_4B")
    parser.add_argument("--min-samples", type=int, default=1,
                        help="Minimum number of samples per model-month to include that data point")
    return parser.parse_args()


# Vibrant, contrasting color palette (cycled as needed)
VIBRANT_COLORS = [
    # '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#17becf',
    '#66c2a5', '#8da0cb',  '#fc8d62', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'
]


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


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


def parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    # Numeric epoch seconds or milliseconds
    if isinstance(value, (int, float)):
        try:
            # Heuristic: if > 10^11 treat as ms
            ts = float(value)
            if ts > 1e11:
                ts /= 1000.0
            return datetime.utcfromtimestamp(ts).date()
        except Exception:
            return None
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        # If looks like ISO, take date part before 'T'
        if 'T' in v and len(v) >= 10:
            v = v.split('T', 1)[0]
        # If like YYYY/MM/DD -> convert to dashes
        if re.match(r'^\d{4}/\d{2}/\d{2}$', v):
            v = v.replace('/', '-')
        # Extract YYYY-MM-DD prefix if string is longer
        if len(v) > 10 and re.match(r'^\d{4}-\d{2}-\d{2}', v):
            v = v[:10]
        # Epoch string
        if re.match(r'^\d{10}(\.\d+)?$', v):
            try:
                ts = float(v)
                return datetime.utcfromtimestamp(ts).date()
            except Exception:
                return None
        try:
            return datetime.strptime(v, '%Y-%m-%d').date()
        except Exception:
            return None
    return None


def get_resolution_date(item: Dict[str, Any]) -> Optional[date]:
    candidate_keys = [
        'resolution_date', 'date_resolve_at', 'resolve_date', 'resolutionDate',
        'resolveAt', 'resolve_time', 'date_resolved', 'resolved_at', 'end_time'
    ]
    for key in candidate_keys:
        if key in item:
            d = parse_date(item.get(key))
            if d is not None:
                return d
    # Some items may have nested metadata
    meta = item.get('metadata') or item.get('meta')
    if isinstance(meta, dict):
        for key in candidate_keys:
            if key in meta:
                d = parse_date(meta.get(key))
                if d is not None:
                    return d
    return None


def month_index_for_date(d: date) -> Optional[int]:
    for idx, (start_d, end_d) in enumerate(zip(MONTH_STARTS, MONTH_ENDS)):
        if start_d <= d <= end_d:
            return idx
    return None


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


def model_color(name: str) -> str:
    name_lower = name.lower()
    if 'filtered' in name_lower or 'freeform' in name_lower:
        return 'red'
    if 'checkpoint' in name_lower or 'rl' in name_lower:
        return '#ff7f0e'
    return '#2ca02c'


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def build_style_maps(model_names: List[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    # Use a vibrant, contrasting palette and distinct markers
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '>', '<']
    color_map: Dict[str, Any] = {}
    marker_map: Dict[str, str] = {}
    for idx, model in enumerate(model_names):
        color_map[model] = VIBRANT_COLORS[idx % len(VIBRANT_COLORS)]
        marker_map[model] = markers[idx % len(markers)]
    return color_map, marker_map


def compute_axis_limits(series: List[Tuple[List[float], List[float]]]) -> Tuple[float, float]:
    values = []
    for means, errs in series:
        for m, e in zip(means, errs):
            if m is None or (isinstance(m, float) and np.isnan(m)):
                continue
            e_val = 0.0 if e is None else e
            values.append(m - e_val)
            values.append(m + e_val)
    if not values:
        return 0.0, 1.0
    vmin, vmax = np.min(values), np.max(values)
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    pad = 0.08 * (vmax - vmin)
    return vmin - pad, vmax + pad


def plot_accuracy_only(
    monthly_accuracy: Dict[str, Dict[int, Tuple[float, float, int]]],
    output_path: str,
    title_prefix: str
) -> None:
    all_models = sorted(monthly_accuracy.keys())
    if not all_models:
        print("No accuracy data found for plotting by month")
        return

    x_positions = np.arange(NUM_MONTHS)
    colors, markers = build_style_maps(all_models)

    fig, ax = plt.subplots(figsize=(10, 6))

    handles = []
    labels = []
    all_series = []

    for m in all_models:
        means: List[float] = []
        errs: List[float] = []
        for idx in range(NUM_MONTHS):
            if idx in monthly_accuracy.get(m, {}):
                mean, se, _ = monthly_accuracy[m][idx]
                means.append(mean)
                errs.append(se)
            else:
                means.append(np.nan)
                errs.append(0.0)
        eb = ax.errorbar(
            x_positions,
            means,
            yerr=errs,
            marker=markers[m],
            markersize=10,
            linewidth=3.5,
            capsize=5,
            color=colors[m],
            label=print_names.get(m, m),
            alpha=0.95,
        )
        handles.append(eb.lines[0] if hasattr(eb, 'lines') else eb)
        labels.append(print_names.get(m, m))
        all_series.append((means, errs))

    ymin, ymax = compute_axis_limits(all_series)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Accuracy (\%)', fontsize=22, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(MONTH_LABELS, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.35, linestyle='--', axis='y')

    # Legend on top in a row, larger font
    fig.legend(handles=handles, labels=labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy plot to {output_path}")
    # also save the plot as a pdf
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_brier_only(
    monthly_brier: Dict[str, Dict[int, Tuple[float, float, int]]],
    output_path: str,
    title_prefix: str
) -> None:
    all_models = sorted(monthly_brier.keys())
    if not all_models:
        print("No nBrier data found for plotting by month")
        return

    x_positions = np.arange(NUM_MONTHS)
    colors, markers = build_style_maps(all_models)

    fig, ax = plt.subplots(figsize=(10, 6))

    handles = []
    labels = []
    all_series = []

    for m in all_models:
        means: List[float] = []
        errs: List[float] = []
        for idx in range(NUM_MONTHS):
            if idx in monthly_brier.get(m, {}):
                mean, se, _ = monthly_brier[m][idx]
                means.append(mean)
                errs.append(se)
            else:
                means.append(np.nan)
                errs.append(0.0)
        eb = ax.errorbar(
            x_positions,
            means,
            yerr=errs,
            marker=markers[m],
            markersize=10,
            linewidth=3.5,
            capsize=5,
            color=colors[m],
            label=print_names.get(m, m),
            alpha=0.95,
        )
        handles.append(eb.lines[0] if hasattr(eb, 'lines') else eb)
        labels.append(print_names.get(m, m))
        all_series.append((means, errs))

    ymin, ymax = compute_axis_limits(all_series)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Brier Score \ (higher is better)', fontsize=22, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(MONTH_LABELS, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.35, linestyle='--', axis='y')

    # Legend on top in a row, larger font
    fig.legend(handles=handles, labels=labels, 
               loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Brier plot to {output_path}")
    # also save the plot as a pdf
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_by_resolution_month(
    monthly_brier: Dict[str, Dict[int, Tuple[float, float, int]]],
    monthly_accuracy: Dict[str, Dict[int, Tuple[float, float, int]]],
    output_path: str,
    title_prefix: str
) -> None:
    # Union of models
    all_models = sorted(set(list(monthly_brier.keys()) + list(monthly_accuracy.keys())))
    if not all_models:
        print("No valid model data found for plotting by month")
        return

    x_positions = np.arange(NUM_MONTHS)
    colors, markers = build_style_maps(all_models)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # Accuracy subplot (top)
    acc_handles = []
    acc_labels = []
    acc_series = []
    for m in all_models:
        means = []
        errs = []
        for idx in range(NUM_MONTHS):
            if idx in monthly_accuracy.get(m, {}):
                mean, se, n = monthly_accuracy[m][idx]
                means.append(mean)
                errs.append(se)
            else:
                means.append(np.nan)
                errs.append(0.0)
        h = axes[0].errorbar(x_positions, means, yerr=errs, marker=markers[m], markersize=7,
                              linewidth=2.5, capsize=4, color=colors[m], label=print_names.get(m, m), alpha=0.95)
        acc_handles.append(h.lines[0] if hasattr(h, 'lines') else h)
        acc_labels.append(print_names.get(m, m))
        acc_series.append((means, errs))

    ymin_acc, ymax_acc = compute_axis_limits(acc_series)
    axes[0].set_ylim(ymin_acc, ymax_acc)
    axes[0].set_ylabel('Mean Accuracy', fontsize=22, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')

    # Brier subplot (bottom)
    brier_series = []
    for m in all_models:
        means = []
        errs = []
        for idx in range(NUM_MONTHS):
            if idx in monthly_brier.get(m, {}):
                mean, se, n = monthly_brier[m][idx]
                means.append(mean)
                errs.append(se)
            else:
                means.append(np.nan)
                errs.append(0.0)
        axes[1].errorbar(x_positions, means, yerr=errs, marker=markers[m], markersize=7,
                         linewidth=2.5, capsize=4, color=colors[m], label=print_names.get(m, m), alpha=0.95)
        brier_series.append((means, errs))

    ymin_b, ymax_b = compute_axis_limits(brier_series)
    axes[1].set_ylim(ymin_b, ymax_b)
    axes[1].set_ylabel('Mean Brier', fontsize=22, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')

    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(MONTH_LABELS, rotation=0, ha='center', fontsize=16)
    axes[0].tick_params(axis='y', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)

    # fig.suptitle(f'{title_prefix}', fontsize=24, fontweight='bold', y=0.98)
    fig.legend(handles=acc_handles, labels=acc_labels, title='Model', title_fontsize=14,
               loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=min(6, max(1, len(all_models))),
               frameon=False, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined time-based plot to {output_path}")
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

    dataset_suffix = os.path.basename(input_dir.rstrip('/'))
    output_dir = os.path.join(output_root, dataset_suffix)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning JSONL files in: {input_dir}")
    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.jsonl')]
    print(f"Found {len(jsonl_files)} JSONL files")

    file_cache: Dict[str, List[Dict[str, Any]]] = {}

    for file_path in jsonl_files:
        data = load_jsonl_file(file_path)
        # Attach resolution month index if available
        for item in data:
            d = get_resolution_date(item)
            item['_resolution_month_idx'] = month_index_for_date(d) if d else None
        file_cache[file_path] = data

    judge_field_template = "score_{}"

    # Aggregation: per model -> per month idx -> list of items
    for_plot_brier: Dict[str, Dict[int, Tuple[float, float, int]]] = {}
    for_plot_acc: Dict[str, Dict[int, Tuple[float, float, int]]] = {}

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        model_name, num_generations = extract_model_info_from_filename(filename)
        if "withbinary" in model_name:
            parts = model_name.split("-")
            model_name = "-".join([part for part in parts if "with" not in part and "binary" not in part])
        model_key = f"{model_name}"

        data = file_cache[file_path]

        judge_field = judge_field_template.format(judge)
        has_judge_field = any(judge_field in item for item in data)
        if not has_judge_field:
            continue

        # For each month, filter items
        for idx in range(NUM_MONTHS):
            month_items = [it for it in data if it.get('_resolution_month_idx') == idx]
            if len(month_items) < min_samples:
                continue
            
            print(f"Model: {model_key}, Month: {idx}, Number of items: {len(month_items)}")
            mean_brier, brier_se = calculate_model_brier_statistics(month_items, num_generations, judge_field)
            mean_acc, acc_se = calculate_model_accuracy_statistics(month_items, num_generations, judge_field)

            for_plot_brier.setdefault(model_key, {})[idx] = (mean_brier, brier_se, len(month_items))
            for_plot_acc.setdefault(model_key, {})[idx] = (mean_acc, acc_se, len(month_items))

    if not for_plot_brier and not for_plot_acc:
        print("No valid model-month data found for plotting")
        return

    # Commented out combined plot to focus on separate, cleaner figures
    # output_path = os.path.join(output_dir, f"combined_by_month_{dataset_suffix}_{judge}.png")
    # title_prefix = f"By Resolution Month (May–Aug 2025) | Dataset: {dataset_suffix} | Judge: {judge}"
    # plot_by_resolution_month(for_plot_brier, for_plot_acc, output_path, title_prefix)

    # Save separate figures for Accuracy and Brier
    title_prefix = f"By Resolution Month (May–Aug 2025) | Dataset: {dataset_suffix} | Judge: {judge}"
    acc_output_path = os.path.join(output_dir, f"accuracy_by_month_{dataset_suffix}_{judge}.png")
    brier_output_path = os.path.join(output_dir, f"brier_by_month_{dataset_suffix}_{judge}.png")
    plot_accuracy_only(for_plot_acc, acc_output_path, title_prefix)
    plot_brier_only(for_plot_brier, brier_output_path, title_prefix)


if __name__ == "__main__":
    main() 