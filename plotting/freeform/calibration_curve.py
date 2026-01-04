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
from sklearn.calibration import calibration_curve

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
    'grok-3-mini': 'Grok 3 Mini',
    'grok-4': 'Grok 4',
    'kimi-k2': 'Kimi K2',
    'gemini-2.5-pro-preview': 'Gemini 2.5 Pro',
    'gemini-2.5-flash-preview': 'Gemini 2.5 Flash',
    'grok-4-fast': 'Grok 4 Fast',
    'grok-4-fast:free': 'Grok 4 Fast',
    # 'Qwen3-4B-sft-rl': '\\texttt{OpenForecaster}-4B',
    # 'Qwen3-8B-sft-rl': '\\texttt{OpenForecaster}-8B',
    # 'Qwen3-8B-RL': '\\texttt{OpenForecaster}-8B',
    'Qwen3-8B-RL': 'Qwen3-8B-rl (\\texttt{OpenForecaster})',
    'Qwen3-1.7B-sft-rl': '\\texttt{OpenForecaster}-1.7B',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot calibration curves for models")
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/test5news_302/",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/calibration",
                       help="Output directory for plots")
    parser.add_argument("--judge", type=str, default="Llama_4_Scout",
                       help="Judge model name for score field")
    parser.add_argument("--n_bins", type=int, default=10,
                       help="Number of bins for calibration curve")
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

def extract_probabilities_and_labels(data: List[Dict[str, Any]], num_generations: int, judge_field: str) -> Tuple[List[float], List[int]]:
    """
    Extract all probabilities and their corresponding binary labels (correct/incorrect) for calibration analysis.
    
    Args:
        data: List of evaluation entries
        num_generations: Number of generations to evaluate
        judge_field: Field name for judge scores
        
    Returns:
        Tuple of (probabilities, binary_labels)
    """
    all_probabilities = []
    all_labels = []
    
    for gen_idx in range(num_generations):
        for item in data:
            # Skip items without necessary fields
            if "extracted_answer" not in item or judge_field not in item:
                continue
                
            extracted_answers = item.get("extracted_answer", [])
            judge_scores = item.get(judge_field, [])
            
            # Skip if generation_idx is out of bounds
            if gen_idx >= len(extracted_answers) or gen_idx >= len(judge_scores):
                continue
                
            generation_answer = extracted_answers[gen_idx]
            generation_scores = judge_scores[gen_idx]
            
            # Handle dictionary format (new probabilistic format)
            if isinstance(generation_answer, dict) and isinstance(generation_scores, dict):
                # For each answer option in this generation
                for answer_option, probability in generation_answer.items():
                    if not answer_option or probability is None:
                        continue 
                    
                    if answer_option in generation_scores:
                        is_correct = (int(generation_scores[answer_option]) == 1)
                        all_probabilities.append(float(probability))
                        all_labels.append(int(is_correct))
            
            # Handle string format (old format) - less common for probabilistic forecasting
            elif isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
                # For string format, assume probability of 1.0 for the given answer
                is_correct = (int(generation_scores) == 1)
                all_probabilities.append(1.0)
                all_labels.append(int(is_correct))
    
    return all_probabilities, all_labels

def find_qwen3_models(input_dir: str) -> List[str]:
    """
    Find all Qwen3 model files in the directory (only 4B and 8B models).
    
    Args:
        input_dir: Directory containing evaluation JSONL files
        
    Returns:
        List of file paths for Qwen3 models
    """
    # Get all JSONL files in the directory
    jsonl_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")]
    
    # Filter for Qwen3 models (only 4B and 8B)
    qwen3_files = []
    for file_path in jsonl_files:
        model_name, _ = extract_model_info_from_filename(file_path)
        
        # Only include files that match Qwen3-4B, Qwen3-8B, or llama-3.1 patterns
        if re.match(r'^Qwen3-(4|8)[bB]', model_name, re.IGNORECASE) or re.match(r'^llama-3\.1', model_name, re.IGNORECASE):
            qwen3_files.append(file_path)
            print(f"Found Qwen3/llama-3.1 model: {model_name} ({file_path})")
        elif re.match(r'^Qwen3-\d+\.?\d*[bB]', model_name, re.IGNORECASE) or re.match(r'^llama-3\.\d+', model_name, re.IGNORECASE):
            print(f"Skipping {model_name} (not 4B, 8B, or llama-3.1)")
    
    return qwen3_files

def find_model_pairs(input_dir: str) -> List[Tuple[str, str]]:
    """
    Find pairs of Qwen models where one is the base model and the other is the trained version (with sft-rl suffix).
    Only includes Qwen3 models that are either base (no suffix) or have "sft-rl" suffix.
    Excludes models with only "sft" suffix.
    
    Args:
        input_dir: Directory containing evaluation JSONL files
        
    Returns:
        List of tuples (base_model_file, trained_model_file)
    """
    # Get all JSONL files in the directory
    jsonl_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".jsonl")]
    
    # Extract model names from filenames and filter for Qwen3 models
    model_files = {}
    for file_path in jsonl_files:
        model_name, _ = extract_model_info_from_filename(file_path)
        
        # Only include Qwen3 models
        if not model_name.lower().startswith('qwen3-'): # and not model_name.lower().startswith('llama-'):
            continue
            
        # Exclude models that have "sft" but not "sft-rl" (i.e., only sft models)
        # if 'sft' in model_name.lower() and 'sft-rl' not in model_name.lower():
            # print(f"Excluding SFT-only model: {model_name}")
            # continue
            
        model_files[model_name] = file_path
    
    # Find pairs
    pairs = []
    base_models = set()
    
    # First, identify all base Qwen3 models (those without sft-rl)
    for model_name in model_files.keys():
        if "sft-rl" not in model_name.lower():
            # This should be a base model like Qwen3-4B, Qwen3-8B, etc.
            if re.match(r'^Qwen3-\d+\.?\d*[bB]$', model_name, re.IGNORECASE):
                base_models.add(model_name)
                print(f"Found base model: {model_name}")
    
    # Then find matching trained models
    for base_model in base_models:
        # Look for the corresponding sft-rl version
        for model_name in model_files.keys():
            if "sft-rl" in model_name.lower() or "rl" in model_name.lower():
                # Check if this is the trained version of the base model
                # Expected pattern: Qwen3-4B-sft-rl for base Qwen3-4B
                expected_trained = f"{base_model}-sft-rl" if "sft-rl" in model_name.lower() else f"{base_model}-rl"
                
                if model_name.lower() == expected_trained.lower():
                    pairs.append((model_files[base_model], model_files[model_name]))
                    print(f"Found pair: {base_model} -> {model_name}")
                    break
    
    return pairs

def plot_single_calibration_curve(probs, labels, model_name, output_path, n_bins=10):
    """
    Plot calibration curve for a single model with error bars.
    
    Args:
        probs: Probabilities from the model
        labels: True labels for predictions
        model_name: Name of the model
        output_path: Path to save the plot
        n_bins: Number of bins for calibration curve
    """
    plt.figure(figsize=(10, 8))
    
    # Helper functions for consistent styling
    def get_model_color(model_name):
        if 'qwen' in model_name.lower():
            if 'rl' in model_name.lower():
                return '#9467bd'  # purple for 'Ours'/OpenForecaster
            else:
                return '#1f77b4'  # blue for base Qwen
        return '#1f77b4'  # default blue
    
    def get_line_style(model_name):
        """Return line style based on model size: dashed for 4B, solid for 8B"""
        if '4b' in model_name.lower():
            return '--'
        elif '8b' in model_name.lower():
            return '-'
        else:
            return '-'  # default solid
    
    # Calculate calibration curve with error bars (min 5 samples per bucket)
    if len(probs) > 0:
        display_name = print_names.get(model_name, model_name)
        mean_pred, fraction_pos, errors = calculate_calibration_with_errors(
            labels, probs, n_bins=n_bins, min_samples=5, model_name=display_name, debug=False)
        
        if len(mean_pred) == 0:
            print(f"  Warning: No buckets with sufficient samples (min 5) for {display_name}")
            return
        
        color = get_model_color(model_name)
        line_style = get_line_style(model_name)
        
        # Plot calibration curve with line and markers
        plt.plot(mean_pred, fraction_pos,
                marker='o', linestyle=line_style,
                label=f'{display_name}', 
                linewidth=2.5, markersize=8, color=color, alpha=0.7)
        
        # Add shaded region for uncertainty
        plt.fill_between(mean_pred, 
                        fraction_pos - errors, 
                        fraction_pos + errors,
                        color=color, alpha=0.15)
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration', linewidth=2)
    
    # Customize plot
    plt.xlabel('Confidence', fontsize=20, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=20, fontweight='bold')
    display_name = print_names.get(model_name, model_name)
    plt.title(f'Calibration Curve: {display_name}', fontsize=22, fontweight='bold', pad=20)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=16, loc='best', frameon=False)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add statistics text
    if len(probs) > 0:
        brier = np.mean((np.array(probs) - np.array(labels))**2)
        stats_text = f"Brier Score: {brier:.4f}\nSamples: {len(probs)}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Calibration curve saved to {output_path}")
    
    # Also save as PDF
    pdf_output_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
    print(f"Calibration curve saved to {pdf_output_path}")
    
    plt.close()

def plot_calibration_curve(base_probs, base_labels, trained_probs, trained_labels, 
                          base_model_name, trained_model_name, output_path, n_bins=10):
    """
    Plot calibration curves for base and trained models.
    
    Args:
        base_probs: Probabilities from base model
        base_labels: True labels for base model predictions
        trained_probs: Probabilities from trained model  
        trained_labels: True labels for trained model predictions
        base_model_name: Name of base model
        trained_model_name: Name of trained model
        output_path: Path to save the plot
        n_bins: Number of bins for calibration curve
    """
    plt.figure(figsize=(10, 8))
    
    # Helper functions for consistent styling
    def get_model_color(model_name):
        if 'qwen' in model_name.lower():
            if 'sft-rl' in model_name.lower():
                return '#9467bd'  # purple for 'Ours'/OpenForecaster
            else:
                return '#1f77b4'  # blue for base Qwen
        return '#1f77b4'  # default blue
    
    def get_line_style(model_name):
        """Return line style based on model size: dashed for 4B, solid for 8B"""
        if '4b' in model_name.lower():
            return '--'
        elif '8b' in model_name.lower():
            return '-'
        else:
            return '-'  # default solid
    
    # Calculate calibration curves
    if len(base_probs) > 0:
        base_fraction_pos, base_mean_pred = calibration_curve(base_labels, base_probs, n_bins=n_bins)
        base_color = get_model_color(base_model_name)
        base_line_style = get_line_style(base_model_name)
        plt.plot(base_mean_pred, base_fraction_pos, 'o' + base_line_style, 
                label=f'{print_names.get(base_model_name, base_model_name)} (Base)', 
                linewidth=2, markersize=8, color=base_color)
    
    if len(trained_probs) > 0:
        trained_fraction_pos, trained_mean_pred = calibration_curve(trained_labels, trained_probs, n_bins=n_bins)
        trained_display_name = print_names.get(trained_model_name, trained_model_name)
        trained_color = get_model_color(trained_model_name)
        trained_line_style = get_line_style(trained_model_name)
        plt.plot(trained_mean_pred, trained_fraction_pos, 's' + trained_line_style, 
                label=f'{trained_display_name}', 
                linewidth=2, markersize=8, color=trained_color)
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration', linewidth=2)
    
    # Customize plot
    plt.xlabel('Confidence', fontsize=20, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=20, fontweight='bold')
    plt.title('Calibration Curve: Base vs OpenForecaster Models', fontsize=22, fontweight='bold', pad=20)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=16, loc='best', frameon=True, fancybox=True)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add statistics text
    stats_text = ""
    if len(base_probs) > 0:
        base_brier = np.mean((np.array(base_probs) - np.array(base_labels))**2)
        stats_text += f"Base Brier Score: {base_brier:.4f}\n"
    if len(trained_probs) > 0:
        trained_brier = np.mean((np.array(trained_probs) - np.array(trained_labels))**2)
        stats_text += f"Trained Brier Score: {trained_brier:.4f}"
    
    if stats_text:
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Calibration curve saved to {output_path}")
    
    # Also save as PDF
    pdf_output_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
    print(f"Calibration curve saved to {pdf_output_path}")
    
    plt.close()

def plot_all_calibration_curves(model_pairs, input_dir, judge, output_dir, n_bins=10):
    """
    Plot calibration curves for all model pairs in a single plot.
    """
    plt.figure(figsize=(10, 10))
    
    # Use the same color scheme as scatter_plot.py
    def get_model_color(model_name):
        if 'qwen' in model_name.lower():
            if 'rl' in model_name.lower():
                return '#9467bd'  # purple for 'Ours'/OpenForecaster
            else:
                return '#1f77b4'  # blue for base Qwen
        return '#1f77b4'  # default blue
    
    def get_line_style(model_name):
        """Return line style based on model size: dashed for 4B, solid for 8B"""
        if '4b' in model_name.lower():
            return '--'
        elif '8b' in model_name.lower():
            return '-'
        else:
            return '-'  # default solid
    
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    judge_field = f"score_{judge}"
    pair_count = 0
    
    for i, (base_file, trained_file) in enumerate(model_pairs):
        base_model_name, _ = extract_model_info_from_filename(base_file)
        trained_model_name, _ = extract_model_info_from_filename(trained_file)
        
        # Load data for both models
        base_data = load_jsonl_file(os.path.join(input_dir, base_file))
        trained_data = load_jsonl_file(os.path.join(input_dir, trained_file))
        
        if not base_data or not trained_data:
            continue
            
        # Check if judge field exists
        has_judge_field = any(judge_field in item for item in base_data + trained_data)
        if not has_judge_field:
            print(f"Warning: {judge_field} not found in data for {base_model_name}")
            continue
        
        # Extract probabilities and labels
        base_num_gen = extract_model_info_from_filename(base_file)[1]
        trained_num_gen = extract_model_info_from_filename(trained_file)[1]
        
        base_probs, base_labels = extract_probabilities_and_labels(base_data, base_num_gen, judge_field)
        trained_probs, trained_labels = extract_probabilities_and_labels(trained_data, trained_num_gen, judge_field)
        
        if len(base_probs) == 0 or len(trained_probs) == 0:
            print(f"No valid data found for pair {base_model_name} - {trained_model_name}")
            continue
        
        marker_base = markers[pair_count % len(markers)]
        
        # Calculate calibration curves
        base_fraction_pos, base_mean_pred = calibration_curve(base_labels, base_probs, n_bins=n_bins)
        trained_fraction_pos, trained_mean_pred = calibration_curve(trained_labels, trained_probs, n_bins=n_bins)
        
        # Get colors and line styles for each model
        base_color = get_model_color(base_model_name)
        trained_color = get_model_color(trained_model_name)
        base_line_style = get_line_style(base_model_name)
        trained_line_style = get_line_style(trained_model_name)
        
        # Plot base model
        plt.plot(base_mean_pred, base_fraction_pos, marker_base + base_line_style, 
                label=f'{print_names.get(base_model_name, base_model_name)}', 
                linewidth=4, markersize=10, color=base_color, alpha=0.7)
        
        # Plot trained model with same marker but different color
        trained_display_name = print_names.get(trained_model_name, trained_model_name)
        plt.plot(trained_mean_pred, trained_fraction_pos, marker_base + trained_line_style, 
                label=f'{trained_display_name}', 
                linewidth=4, markersize=10, color=trained_color, alpha=0.7)
        
        pair_count += 1
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration', linewidth=2)
    
    # Customize plot
    plt.xlabel('Confidence', fontsize=32, fontweight='bold', labelpad=12)
    plt.ylabel('Accuracy', fontsize=32, fontweight='bold', labelpad=12)
    # plt.title('Calibration Curves: Base vs OpenForecaster Models', fontsize=22, fontweight='bold', pad=20)
    
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=26, loc='best', frameon=True, fancybox=True, ncol=1)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_output_path = os.path.join(output_dir, "calibration_curves_all_pairs.png")
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    plt.savefig("poster_plot3.png", dpi=1200, transparent=True, bbox_inches="tight")
    print(f"Combined calibration curves saved to {combined_output_path}")
    
    # Also save as PDF
    pdf_output_path = combined_output_path.replace('.png', '.pdf')
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
    print(f"Combined calibration curves saved to {pdf_output_path}")
    
    plt.close()

def calculate_calibration_with_errors(y_true, y_prob, n_bins=10, min_samples=5, model_name=None, debug=False):
    """
    Calculate calibration curve with error bars (standard error) for each bin.
    Only include bins with at least min_samples samples.
    
    Returns:
        mean_predicted_probs: Mean predicted probability in each bin
        fraction_positives: Fraction of positive outcomes in each bin
        errors: Standard error for each bin
    """
    # Calculate which bin each prediction belongs to
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    mean_preds = []
    frac_pos = []
    errors = []
    
    if debug and model_name:
        print(f"    Debug info for {model_name}:")
    
    for i in range(n_bins):
        # Get all samples in this bin
        bin_mask = (binids == i)
        bin_probs = np.array(y_prob)[bin_mask]
        bin_labels = np.array(y_true)[bin_mask]
        
        if debug and model_name:
            bin_range = f"[{bins[i]:.2f}, {bins[i+1]:.2f})"
            print(f"      Bin {i} {bin_range}: {len(bin_labels)} samples")
        
        # Only include bins with sufficient samples
        if len(bin_labels) >= min_samples:
            # Calculate mean predicted probability in this bin
            mean_pred = np.mean(bin_probs)
            
            # Calculate fraction of positives
            p = np.mean(bin_labels)
            
            # Calculate standard error using binomial distribution
            n = len(bin_labels)
            se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
            
            mean_preds.append(mean_pred)
            frac_pos.append(p)
            errors.append(se)
    
    return np.array(mean_preds), np.array(frac_pos), np.array(errors)

def plot_all_qwen3_models(qwen3_files, input_dir, judge, output_dir, n_bins=10):
    """
    Plot calibration curves for all Qwen3 models in a single plot with shaded uncertainty.
    """
    plt.figure(figsize=(10, 10))
    
    # Define colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Use same marker and linestyle for all models
    marker = 'o'
    linestyle = '-'
    linewidth = 2.5
    line_alpha = 0.7
    
    judge_field = f"score_{judge}"
    model_count = 0
    
    for i, model_file in enumerate(qwen3_files):
        model_name, num_gen = extract_model_info_from_filename(model_file)
        
        # Load data
        data = load_jsonl_file(os.path.join(input_dir, model_file))
        
        if not data:
            print(f"  Skipping {model_name}: No data found")
            continue
            
        # Check if judge field exists
        has_judge_field = any(judge_field in item for item in data)
        if not has_judge_field:
            print(f"  Warning: {judge_field} not found in data for {model_name}")
            continue
        
        # Extract probabilities and labels
        probs, labels = extract_probabilities_and_labels(data, num_gen, judge_field)
        
        if len(probs) == 0:
            print(f"  Skipping {model_name}: No valid data found")
            continue
        
        # Get display name
        display_name = print_names.get(model_name, model_name)
        
        # Calculate calibration curve with error bars (min 5 samples per bucket)
        mean_pred, fraction_pos, errors = calculate_calibration_with_errors(
            labels, probs, n_bins=n_bins, min_samples=5, model_name=display_name, debug=True)
        
        if len(mean_pred) == 0:
            print(f"  Skipping {display_name}: No buckets with sufficient samples (min 5)")
            continue
        color = colors[model_count % len(colors)]
        
        # Plot calibration curve with line and markers
        plt.plot(mean_pred, fraction_pos, 
                marker=marker, linestyle=linestyle,
                label=f'{display_name}', 
                linewidth=linewidth, markersize=8, color=color, alpha=line_alpha)
        
        # Add shaded region for uncertainty
        plt.fill_between(mean_pred, 
                        fraction_pos - errors, 
                        fraction_pos + errors,
                        color=color, alpha=0.15)
        
        # Calculate and print Brier score
        brier = np.mean((np.array(probs) - np.array(labels))**2)
        print(f"  {display_name}: Brier={brier:.4f}, Samples={len(probs)}, Valid buckets={len(mean_pred)}")
        
        model_count += 1
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration', linewidth=2)
    
    # Customize plot
    plt.xlabel('Confidence', fontsize=32, fontweight='bold', labelpad=12)
    plt.ylabel('Accuracy', fontsize=32, fontweight='bold', labelpad=12)
    
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=26, loc='best', frameon=False, fancybox=False, ncol=1)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_output_path = os.path.join(output_dir, "calibration_all_qwen3_models.png")
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined calibration curves saved to {combined_output_path}")
    
    # Also save as PDF
    pdf_output_path = combined_output_path.replace('.png', '.pdf')
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
    print(f"Combined calibration curves saved to {pdf_output_path}")
    
    plt.close()

def main():
    args = parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    print(f"Processing directory: {args.input_dir}")
    print(f"Judge: {args.judge}")
    
    # Find all Qwen3 models
    print("\nFinding Qwen3 models...")
    qwen3_files = find_qwen3_models(args.input_dir)
    
    if not qwen3_files:
        print("No Qwen3 models found")
        return
    
    print(f"Found {len(qwen3_files)} Qwen3 models")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot all models in a single calibration curve
    print("\nCreating combined calibration curve for all Qwen3 models...")
    plot_all_qwen3_models(qwen3_files, args.input_dir, args.judge, args.output_dir, args.n_bins)
    
    print(f"\nCalibration analysis complete. Plot saved to {args.output_dir}")

if __name__ == "__main__":
    main()
