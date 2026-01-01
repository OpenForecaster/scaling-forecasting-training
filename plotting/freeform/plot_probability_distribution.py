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
    parser = argparse.ArgumentParser(description="Plot probability distribution histograms for two model files")
    parser.add_argument("--model1_file", type=str, default="/fast/nchandak/forecasting/evals/freeform/manual/theguardian_207/Qwen3-8B_eval_size_207_generations_8_no_article.jsonl",
                       help="Path to first model's JSONL evaluation file")
    parser.add_argument("--model2_file", type=str, default="/fast/nchandak/forecasting/evals/freeform/manual/theguardian_207/Qwen3-8B-2048-10240-datamix70k-checkpoint400_eval_size_207_generations_8_no_article.jsonl",
                       help="Path to second model's JSONL evaluation file")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Output directory for plots")
    parser.add_argument("--bins", type=int, default=20,
                       help="Number of bins for histogram (default: 20)")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Transparency for histogram bars (default: 0.7)")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for output image (default: 300)")
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

def extract_model_name_from_filename(filename):
    """
    Extract model name from filename.
    
    Expected format: ModelName_eval_size_N_generations_M.jsonl
    Returns: model_name
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract model name (everything before _eval)
    model_match = re.match(r'([^_]+(?:_[^_]*?)?(?:-\d+\.?\d*[bB])?)', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = name_without_ext.split('_')[0]
    
    if model_name.endswith('_'):
        model_name = model_name[:-1]
    
    return model_name

def extract_probabilities_from_data(data: List[Dict[str, Any]]) -> List[float]:
    """
    Extract all probability values from the evaluation data.
    
    Args:
        data: List of evaluation entries
        
    Returns:
        List of probability values (0-1)
    """
    probabilities = []
    
    for item in data:
        if "extracted_answer" not in item:
            continue
            
        extracted_answers = item.get("extracted_answer", [])
        
        for generation_answer in extracted_answers:
            # Handle dictionary format (new probabilistic format)
            if isinstance(generation_answer, dict):
                for answer_option, probability in generation_answer.items():
                    if probability is not None and isinstance(probability, (int, float)):
                        # Ensure probability is between 0 and 1
                        if 0 <= probability <= 1:
                            probabilities.append(probability)
            
            # Handle string format (old format) - less common for probabilistic forecasting
            elif isinstance(generation_answer, str):
                # Try to extract probability from the string
                prob_match = re.search(r'<probability>(.*?)</probability>', generation_answer, re.DOTALL)
                if prob_match:
                    try:
                        probability = float(prob_match.group(1).strip())
                        if 0 <= probability <= 1:
                            probabilities.append(probability)
                    except (ValueError, TypeError):
                        continue
    
    return probabilities

def calculate_probability_statistics(probabilities: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for probability distribution.
    
    Args:
        probabilities: List of probability values
        
    Returns:
        Dictionary with statistics
    """
    if not probabilities:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        }
    
    return {
        'mean': np.mean(probabilities),
        'median': np.median(probabilities),
        'std': np.std(probabilities),
        'min': np.min(probabilities),
        'max': np.max(probabilities),
        'count': len(probabilities)
    }

def plot_probability_distribution(model1_data: List[Dict[str, Any]], 
                                model2_data: List[Dict[str, Any]],
                                model1_name: str,
                                model2_name: str,
                                output_path: str,
                                bins: int = 20,
                                alpha: float = 0.7,
                                dpi: int = 300):
    """
    Plot probability distribution histograms for two models.
    
    Args:
        model1_data: Evaluation data for first model
        model2_data: Evaluation data for second model
        model1_name: Name of first model
        model2_name: Name of second model
        output_path: Path to save the plot
        bins: Number of histogram bins
        alpha: Transparency for bars
        dpi: DPI for output image
    """
    
    # Extract probabilities from both models
    model1_probs = extract_probabilities_from_data(model1_data)
    model2_probs = extract_probabilities_from_data(model2_data)
    
    print(f"Model 1 ({model1_name}): {len(model1_probs)} probability values")
    print(f"Model 2 ({model2_name}): {len(model2_probs)} probability values")
    
    if not model1_probs and not model2_probs:
        print("No probability data found in either model file")
        return
    
    # Calculate statistics
    model1_stats = calculate_probability_statistics(model1_probs)
    model2_stats = calculate_probability_statistics(model2_probs)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for the histograms
    color1 = '#1f77b4'  # blue
    color2 = '#ff7f0e'  # orange
    
    # Plot histogram for model 1
    if model1_probs:
        ax1.hist(model1_probs, bins=bins, alpha=alpha, color=color1, 
                edgecolor='black', linewidth=0.5, density=True)
        ax1.axvline(model1_stats['mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {model1_stats['mean']:.3f}")
        ax1.axvline(model1_stats['median'], color='green', linestyle='--', 
                   linewidth=2, label=f"Median: {model1_stats['median']:.3f}")
        ax1.set_title(f'{print_names.get(model1_name, model1_name)} Probability Distribution', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Probability', fontsize=14)
        ax1.set_ylabel('Density', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Count: {model1_stats['count']}\n"
        stats_text += f"Mean: {model1_stats['mean']:.3f}\n"
        stats_text += f"Median: {model1_stats['median']:.3f}\n"
        stats_text += f"Std: {model1_stats['std']:.3f}\n"
        stats_text += f"Range: [{model1_stats['min']:.3f}, {model1_stats['max']:.3f}]"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
        ax1.set_title(f'{print_names.get(model1_name, model1_name)} Probability Distribution', 
                     fontsize=16, fontweight='bold')
    
    # Plot histogram for model 2
    if model2_probs:
        ax2.hist(model2_probs, bins=bins, alpha=alpha, color=color2, 
                edgecolor='black', linewidth=0.5, density=True)
        ax2.axvline(model2_stats['mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {model2_stats['mean']:.3f}")
        ax2.axvline(model2_stats['median'], color='green', linestyle='--', 
                   linewidth=2, label=f"Median: {model2_stats['median']:.3f}")
        ax2.set_title(f'{print_names.get(model2_name, model2_name)} Probability Distribution', 
                     fontsize=16, fontweight='bold')
        ax2.set_xlabel('Probability', fontsize=14)
        ax2.set_ylabel('Density', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Count: {model2_stats['count']}\n"
        stats_text += f"Mean: {model2_stats['mean']:.3f}\n"
        stats_text += f"Median: {model2_stats['median']:.3f}\n"
        stats_text += f"Std: {model2_stats['std']:.3f}\n"
        stats_text += f"Range: [{model2_stats['min']:.3f}, {model2_stats['max']:.3f}]"
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title(f'{print_names.get(model2_name, model2_name)} Probability Distribution', 
                     fontsize=16, fontweight='bold')
    
    # Set consistent x-axis limits for both plots
    if model1_probs and model2_probs:
        all_probs = model1_probs + model2_probs
        x_min, x_max = 0, 1
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Probability distribution plot saved to {output_path}")
    plt.close()

def plot_combined_probability_distribution(model1_data: List[Dict[str, Any]], 
                                         model2_data: List[Dict[str, Any]],
                                         model1_name: str,
                                         model2_name: str,
                                         output_path: str,
                                         bins: int = 20,
                                         alpha: float = 0.7,
                                         dpi: int = 300):
    """
    Plot combined probability distribution histogram for both models on the same plot.
    
    Args:
        model1_data: Evaluation data for first model
        model2_data: Evaluation data for second model
        model1_name: Name of first model
        model2_name: Name of second model
        output_path: Path to save the plot
        bins: Number of histogram bins
        alpha: Transparency for bars
        dpi: DPI for output image
    """
    
    # Extract probabilities from both models
    model1_probs = extract_probabilities_from_data(model1_data)
    model2_probs = extract_probabilities_from_data(model2_data)
    
    if not model1_probs and not model2_probs:
        print("No probability data found in either model file")
        return
    
    # Calculate statistics
    model1_stats = calculate_probability_statistics(model1_probs)
    model2_stats = calculate_probability_statistics(model2_probs)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Colors for the histograms
    color1 = '#1f77b4'  # blue
    color2 = '#ff7f0e'  # orange
    
    # Plot histograms
    if model1_probs:
        plt.hist(model1_probs, bins=bins, alpha=alpha, color=color1, 
                edgecolor='black', linewidth=0.5, density=True, 
                label=f"{print_names.get(model1_name, model1_name)} (n={len(model1_probs)})")
        plt.axvline(model1_stats['mean'], color=color1, linestyle='--', 
                   linewidth=2, alpha=0.8, label=f"{print_names.get(model1_name, model1_name)} Mean: {model1_stats['mean']:.3f}")
    
    if model2_probs:
        plt.hist(model2_probs, bins=bins, alpha=alpha, color=color2, 
                edgecolor='black', linewidth=0.5, density=True, 
                label=f"{print_names.get(model2_name, model2_name)} (n={len(model2_probs)})")
        plt.axvline(model2_stats['mean'], color=color2, linestyle='--', 
                   linewidth=2, alpha=0.8, label=f"{print_names.get(model2_name, model2_name)} Mean: {model2_stats['mean']:.3f}")
    
    plt.xlabel('Probability', fontsize=16, fontweight='bold')
    plt.ylabel('Density', fontsize=16, fontweight='bold')
    plt.title('Probability Distribution Comparison', fontsize=18, fontweight='bold', pad=20)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Add statistics text
    stats_text = ""
    if model1_probs:
        stats_text += f"{print_names.get(model1_name, model1_name)}:\n"
        stats_text += f"  Mean: {model1_stats['mean']:.3f}, Median: {model1_stats['median']:.3f}\n"
        stats_text += f"  Std: {model1_stats['std']:.3f}, Range: [{model1_stats['min']:.3f}, {model1_stats['max']:.3f}]\n\n"
    
    if model2_probs:
        stats_text += f"{print_names.get(model2_name, model2_name)}:\n"
        stats_text += f"  Mean: {model2_stats['mean']:.3f}, Median: {model2_stats['median']:.3f}\n"
        stats_text += f"  Std: {model2_stats['std']:.3f}, Range: [{model2_stats['min']:.3f}, {model2_stats['max']:.3f}]"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Combined probability distribution plot saved to {output_path}")
    plt.close()

def main():
    args = parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.model1_file):
        print(f"Error: Model 1 file {args.model1_file} does not exist")
        return
    
    if not os.path.exists(args.model2_file):
        print(f"Error: Model 2 file {args.model2_file} does not exist")
        return
    
    # Extract model names from filenames
    model1_name = extract_model_name_from_filename(os.path.basename(args.model1_file))
    model2_name = extract_model_name_from_filename(os.path.basename(args.model2_file))
    
    print(f"Model 1: {model1_name} ({args.model1_file})")
    print(f"Model 2: {model2_name} ({args.model2_file})")
    
    # Load data from both files
    print(f"\nLoading data from {args.model1_file}")
    model1_data = load_jsonl_file(args.model1_file)
    print(f"Loaded {len(model1_data)} samples for {model1_name}")
    
    print(f"\nLoading data from {args.model2_file}")
    model2_data = load_jsonl_file(args.model2_file)
    print(f"Loaded {len(model2_data)} samples for {model2_name}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    base_filename = f"probability_distribution_{model1_name}_vs_{model2_name}"
    side_by_side_path = os.path.join(args.output_dir, f"{base_filename}_side_by_side.png")
    combined_path = os.path.join(args.output_dir, f"{base_filename}_combined.png")
    
    # Plot side-by-side histograms
    print(f"\nCreating side-by-side probability distribution plot...")
    plot_probability_distribution(
        model1_data, model2_data, 
        model1_name, model2_name, 
        side_by_side_path,
        bins=args.bins,
        alpha=args.alpha,
        dpi=args.dpi
    )
    
    # Plot combined histogram
    print(f"\nCreating combined probability distribution plot...")
    plot_combined_probability_distribution(
        model1_data, model2_data, 
        model1_name, model2_name, 
        combined_path,
        bins=args.bins,
        alpha=args.alpha,
        dpi=args.dpi
    )
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Model 1: {print_names.get(model1_name, model1_name)}")
    print(f"Model 2: {print_names.get(model2_name, model2_name)}")
    
    # Extract and print statistics
    model1_probs = extract_probabilities_from_data(model1_data)
    model2_probs = extract_probabilities_from_data(model2_data)
    
    if model1_probs:
        model1_stats = calculate_probability_statistics(model1_probs)
        print(f"\n{print_names.get(model1_name, model1_name)}:")
        print(f"  Count: {model1_stats['count']}")
        print(f"  Mean: {model1_stats['mean']:.4f}")
        print(f"  Median: {model1_stats['median']:.4f}")
        print(f"  Std: {model1_stats['std']:.4f}")
        print(f"  Range: [{model1_stats['min']:.4f}, {model1_stats['max']:.4f}]")
    
    if model2_probs:
        model2_stats = calculate_probability_statistics(model2_probs)
        print(f"\n{print_names.get(model2_name, model2_name)}:")
        print(f"  Count: {model2_stats['count']}")
        print(f"  Mean: {model2_stats['mean']:.4f}")
        print(f"  Median: {model2_stats['median']:.4f}")
        print(f"  Std: {model2_stats['std']:.4f}")
        print(f"  Range: [{model2_stats['min']:.4f}, {model2_stats['max']:.4f}]")
    
    print(f"\nPlots saved to:")
    print(f"  Side-by-side: {side_by_side_path}")
    print(f"  Combined: {combined_path}")

if __name__ == "__main__":
    main() 