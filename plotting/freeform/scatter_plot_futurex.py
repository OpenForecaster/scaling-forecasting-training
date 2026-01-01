#!/usr/bin/env python3
import os
import json
import re
import ast
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
    'deepseek-chat-v3-0324': 'deepseek-v3',
    'DeepSeek-V3-0324': 'V3',
    'deepseek-r1-0528': 'R1',
    'llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'llama-4-maverick': 'llama-4-maverick',
    'llama-4-scout': 'Scout',
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
    parser = argparse.ArgumentParser(description="Scatter plot of Accuracy (y) vs nBrier (x) for FutureX evals")
    parser.add_argument("--input_dir", type=str,
                        default="/fast/nchandak/forecasting/evals/futurex-past86-retrieval/",
                        help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/futurex/scatter",
                        help="Output directory for the scatter plot")
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


def extract_model_info_from_filename(filename: str) -> Tuple[str, int]:
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


def calculate_brier_score(probability: float, is_correct: bool) -> float:
    if is_correct:
        return -((1 - probability) ** 2)
    else:
        return -(probability ** 2)
 
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
    all_generation_means = []
    for gen_idx in range(num_generations):
        generation_brier_scores = calculate_generation_brier_scores_futurex(data, gen_idx)
        if generation_brier_scores:
            generation_mean = np.mean(generation_brier_scores)
            all_generation_means.append(generation_mean)
    if not all_generation_means:
        return 0.0, 0.0
    mean_brier = np.mean(all_generation_means)
    std_error = np.std(all_generation_means, ddof=1) / np.sqrt(len(all_generation_means)) if len(all_generation_means) > 1 else 0.0
    return mean_brier, std_error


def calculate_generation_accuracy_futurex(data: List[Dict[str, Any]], generation_idx: int) -> float:
    correct_count = 0
    total_count = 0

    for item in data:
        if "extracted_answer" not in item or "answer" not in item:
            continue

        extracted_answers = item.get("extracted_answer", [])
        ground_truth_raw = item.get("answer", "")
        level = int(item.get("level", 0))
        is_binary_list = item.get("is_binary", [])
        if len(is_binary_list) == 0:
            is_binary = "no" in ground_truth_raw.lower() or "yes" in ground_truth_raw.lower()
            is_binary_list = [is_binary] * len(extracted_answers)

        if generation_idx >= len(extracted_answers):
            continue

        generation_answer = extracted_answers[generation_idx]

        try:
            if isinstance(ground_truth_raw, str):
                if ground_truth_raw.startswith('[') and ground_truth_raw.endswith(']'):
                    ground_truth_list = ast.literal_eval(ground_truth_raw)
                    ground_truth = ground_truth_list[0].lower() if ground_truth_list else ""
                else:
                    ground_truth = ground_truth_raw.lower()
            else:
                ground_truth = str(ground_truth_raw).lower()
        except Exception:
            continue

        if level <= 1:
            if isinstance(generation_answer, dict) and len(generation_answer) > 0:
                predicted = list(generation_answer.keys())[0].lower() if generation_answer else None
                if predicted:
                    is_correct = False
                    if ground_truth in ["yes", "y", "true", "1"]:
                        is_correct = predicted in ["yes", "y", "true", "1"]
                    elif ground_truth in ["no", "n", "false", "0"]:
                        is_correct = predicted in ["no", "n", "false", "0"]
                    else:
                        is_correct = predicted == ground_truth or predicted.strip() in ground_truth
                    if is_correct:
                        correct_count += 1
                    total_count += 1
        elif level == 4:
            if isinstance(generation_answer, dict) and len(generation_answer) > 0:
                predicted = list(generation_answer.keys())[0].lower() if generation_answer else None
                if predicted:
                    try:
                        estimation = float(predicted)
                        ground_truth_val = float(ground_truth)
                        relative_error = abs(estimation - ground_truth_val) / ground_truth_val
                        score = 1 - relative_error
                    except Exception:
                        continue
                    correct_count += max(0, score)
                    total_count += 1
        else:
            continue

    return correct_count / total_count if total_count > 0 else 0.0


def calculate_model_accuracy_statistics_futurex(data: List[Dict[str, Any]], num_generations: int) -> Tuple[float, float]:
    all_generation_accuracies = []
    for gen_idx in range(num_generations):
        generation_accuracy = calculate_generation_accuracy_futurex(data, gen_idx) * 100.0
        all_generation_accuracies.append(generation_accuracy)
    if not all_generation_accuracies:
        return 0.0, 0.0
    mean_accuracy = np.mean(all_generation_accuracies)
    std_error = np.std(all_generation_accuracies, ddof=1) / np.sqrt(len(all_generation_accuracies)) if len(all_generation_accuracies) > 1 else 0.0
    return mean_accuracy, std_error


def determine_family(model_name: str) -> str:
    name = model_name.lower()
    # if 'qwen' in name:
    #     if re.match(r'^[Qq]wen3-1\.7[bB]-', model_name):
    #         return 'Trained on \\texttt{OpenForesight}'
    #     if re.match(r'^[Qq]wen3-4[bB]-', model_name):
    #         return 'Trained on \\texttt{OpenForesight}'
    #     if re.match(r'^[Qq]wen3-8[bB]-', model_name):
    #         return 'Trained on \\texttt{OpenForesight}'
        
    #     return 'Qwen'
    
    
    
    if 'qwen' in name:
        if re.match(r'^[Qq]wen3-1\.7[bB]-', model_name) and 'rl' in model_name.lower():
            return 'Ours'
            return 'Trained on \\texttt{OpenForesight}'
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
        'Ours': '#9467bd',   # purple
    }


def qwen_trained_marker(model_name: str) -> str:
    # Use special shapes for trained Qwen3-XB- models
    if re.match(r'^[Qq]wen3-1\.7[bB]-', model_name):
        return '*'
        return '^'
    if re.match(r'^[Qq]wen3-4[bB]-', model_name):
        return '*'
        return 's'
    if re.match(r'^[Qq]wen3-8[bB]-', model_name):
        return '*'
    return 'o'


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def compute_model_metrics(input_dir: str) -> Dict[str, Dict[str, Any]]:
    model_metrics: Dict[str, Dict[str, Any]] = {}

    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.jsonl')]

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        model_name, num_generations = extract_model_info_from_filename(filename)
        model_key = f"{model_name}"

        data = load_jsonl_file(file_path)

        mean_brier, brier_se = calculate_model_brier_statistics_futurex(data, num_generations)
        mean_acc, acc_se = calculate_model_accuracy_statistics_futurex(data, num_generations)

        model_metrics[model_key] = {
            'model_name': model_name,
            'mean_brier': mean_brier,
            'brier_se': brier_se,
            'mean_accuracy': mean_acc,
            'acc_se': acc_se,
            'num_samples': len(data),
            'num_generations': num_generations,
        }

    return model_metrics


def plot_scatter(metrics: Dict[str, Dict[str, Any]], output_path: str, title: str) -> None:
    if not metrics:
        print("No metrics to plot")
        return

    families = family_color_map()

    # Prepare plot (bigger canvas)
    fig, ax = plt.subplots(figsize=(12, 12))

    handles = {}
    
    x_max = -1 
    y_max = -1 
    # Plot each model
    for model_key, info in metrics.items():
        x = info['mean_accuracy']
        y = info['mean_brier']
        xerr = info.get('acc_se', 0)
        yerr = info.get('brier_se', 0)
        print(model_key, x, y)
        print(model_key, xerr, yerr)
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
        # Shift left more for longer labels, and up a bit more for longer labels
        x_offset = -5 - (label_len * 1.5)
        y_offset = 15 #- (label_len)
        y_offset = 30 #- (label_len)
        
        if model_key.lower() == "qwen3-8b" or model_key.lower() == "qwen3-4b" or model_key.lower() == "qwen3-1.7b":
            x_offset = - 70 - label_len * 1.5
            y_offset = 1 #- (label_len)
            
            if "8b" in model_key.lower():
                x_offset += 160
                y_offset = -40
                
            if "4b" in model_key.lower():
                x_offset += 160
                y_offset -= 50
            
            
        # elif "sft" in model_key.lower() and "rl" in model_key.lower():
        elif "rl" in model_key.lower():
            x_offset = 10 # 10 + (label_len * 3)
            y_offset = -50 #- (label_len)
            if "4b" in model_key.lower():
                y_offset -= 40
                x_offset -= 20
            if "8b" in model_key.lower():
                # x_offset = -155
                # y_offset = -10
                y_offset = -60
                x_offset = -50
            
        else :
            x_offset = -20 - (label_len * 7)
            y_offset = 1 #- (label_len)
        
        if "gpt-oss-120b" in model_key.lower():
            # y_offset = -50
            y_offset = 20
            x_offset = 130
            
            # x_offset = 20
        
        if "gpt-oss-20b" in model_key.lower():
            y_offset = -70
            x_offset = 115
            
        if "maverick" in model_key.lower():
            # y_offset -= 60 #- (label_len)
            y_offset = 40
            x_offset += 250
            
        if "grok-3-mini" in model_key.lower():
            # y_offset = -105
            # x_offset = 0
            y_offset = 10
            x_offset = -100
            
        if "v3" in model_key.lower():
            y_offset = 23
            # x_offset = -80
            x_offset = 115
            
        if "r1" in model_key.lower():
            y_offset = -50
            x_offset = 80
            # x_offset = 20
            
        if "235" in model_key.lower():
            y_offset = -100
            x_offset = 40
            
        # x_max = max(x_max, x + x_offset)
        # y_max = max(y_max, y + y_offset)
        
        ax.annotate(label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(x_offset, y_offset),
                    ha='center', va='bottom', fontsize=30, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))

        # For legend by family, store one handle per family
        if fam not in handles:
            # Use the Line2D object for the marker from errorbar for legend
            handles[fam] = sc[0] if isinstance(sc, tuple) else sc

        # For legend by family, store one handle per family
        if fam not in handles:
            handles[fam] = sc

    ax.set_xlabel('Accuracy (\%) ($\\uparrow$)', fontsize=28, fontweight='bold')
    ax.set_ylabel('Brier Score ($\\uparrow$)', fontsize=28, fontweight='bold', labelpad=14)

    # half the number of xticks (keep only the even ones)
    xticks = ax.get_xticks()
    # xticks = [x for i,x in enumerate(xticks) if i % 2 == 0]
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
    xpad = 0.2 * (maxx - minx)
    ypad = 0.1 * (maxy - miny)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad * 2 , maxy + ypad * 1.5)
    

    ax.grid(True, alpha=0.35, linestyle='--')
    ax.tick_params(axis='both', labelsize=30, length=6, width=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to {output_path}")
    # also save as pdf
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig("poster_plot_futurex.png", dpi=1200, transparent=True, bbox_inches="tight")

    plt.close(fig)


def main():
    args = parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    output_path = os.path.join(args.output_dir, f"scatter_brier_accuracy_{dataset_suffix}.png")

    metrics = compute_model_metrics(args.input_dir)

    title = f"Accuracy vs nBrier | Dataset: {dataset_suffix}"
    plot_scatter(metrics, output_path, title)


if __name__ == "__main__":
    main() 