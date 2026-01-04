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
    parser = argparse.ArgumentParser(description="Plot probability changes between two models using Brier scores")
    parser.add_argument("--original_file", type=str, required=True,
                       help="Path to the original model evaluation JSONL file")
    parser.add_argument("--checkpoint_file", type=str, required=True,
                       help="Path to the checkpoint model evaluation JSONL file")
    parser.add_argument("--output-dir", type=str, default="plots/brier_change",
                       help="Output directory for plots")
    parser.add_argument("--judge", type=str, default="Qwen3_4B",
                       help="Judge model name for score field")
    parser.add_argument("--bins", type=int, default=10,
                       help="Number of bins for the grid (default: 10)")
    parser.add_argument("--colormap", type=str, default="Pastel2",
                       choices=['turbo', 'plasma', 'Pastel1', 'hsv', 'magma', 'inferno', 'hot', 'coolwarm', 'viridis'],
                       help="Colormap for the heatmap (default: turbo)")
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
        return - (probability ** 2)

def calculate_sample_brier_score(item: Dict[str, Any], judge_field: str) -> float:
    """
    Calculate mean Brier score for a single sample across all generations.
    
    Args:
        item: Single evaluation entry
        judge_field: Field name for judge scores
        
    Returns:
        Mean Brier score for this sample
    """
    if "extracted_answer" not in item or judge_field not in item:
        return None
        
    extracted_answers = item.get("extracted_answer", [])
    judge_scores = item.get(judge_field, [])
    
    if not extracted_answers or not judge_scores:
        return None
    
    generation_brier_scores = []
    
    for gen_idx in range(len(extracted_answers)):
        if gen_idx >= len(judge_scores):
            continue
            
        generation_answer = extracted_answers[gen_idx]
        generation_scores = judge_scores[gen_idx]
        
        # Handle dictionary format (new probabilistic format)
        if isinstance(generation_answer, dict) and isinstance(generation_scores, dict):
            # For each answer option in this generation
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
                brier_score -= 1 # Penalize for not having any correct answer so its probability is taken as 0
            
            generation_brier_scores.append(brier_score)
        
        # Handle string format (old format)
        elif isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
            is_correct = (int(generation_scores) == 1)
            if not is_correct:
                brier_score = -2
            else:
                brier_score = 0
            generation_brier_scores.append(brier_score)
    
    if not generation_brier_scores:
        return None
    
    # Return mean Brier score across all generations
    return np.mean(generation_brier_scores) + 1

def extract_model_name_from_filename(filename):
    """Extract model name from filename."""
    # Remove .jsonl extension
    name_without_ext = os.path.basename(filename).replace('.jsonl', '')
    
    # Extract model name (everything before _eval)
    model_match = re.match(r'([^_]+(?:_[^_]*?)?(?:-\d+\.?\d*[bB])?)', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = name_without_ext.split('_')[0]
    
    if model_name.endswith('_'):
        model_name = model_name[:-1]
    
    parts = model_name.split('-')
    if len(parts) > 3:
        model_name = '-'.join(parts[:3] + parts[-1:])
    
    return model_name

def get_sample_brier_scores(file_path: str, judge: str) -> Dict[str, float]:
    """
    Get Brier scores for all samples in a file, indexed by URL.
    
    Args:
        file_path: Path to the evaluation JSONL file
        judge: Judge model name
        
    Returns:
        Dictionary mapping URL to Brier score
    """
    data = load_jsonl_file(file_path)
    judge_field = f"score_{judge}"
    
    sample_scores = {}
    
    for item in data:
        url = item.get("url", item.get("idx", None))
        if url is None:
            continue
        
        brier_score = calculate_sample_brier_score(item, judge_field)
        
        if brier_score is not None:
            sample_scores[url] = brier_score
    
    print(f"Loaded {len(sample_scores)} samples with valid Brier scores from {file_path}")
    return sample_scores

def create_brier_change_grid(original_scores: Dict[str, float], 
                           checkpoint_scores: Dict[str, float], 
                           bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid showing how Brier scores changed between models.
    
    Args:
        original_scores: Dictionary mapping URL to original Brier score
        checkpoint_scores: Dictionary mapping URL to checkpoint Brier score
        bins: Number of bins for the grid
        
    Returns:
        Tuple of (grid, x_edges, y_edges)
    """
    # Find common samples
    common_urls = set(original_scores.keys()) & set(checkpoint_scores.keys())
    
    if not common_urls:
        print("No common samples found between the two files")
        return None, None, None
    
    print(f"Found {len(common_urls)} common samples")
    
    # Extract scores for common samples
    original_values = [original_scores[url] for url in common_urls]
    checkpoint_values = [checkpoint_scores[url] for url in common_urls]
    
    # Convert to numpy arrays
    original_array = np.array(original_values)
    checkpoint_array = np.array(checkpoint_values)
    
    # Create 2D histogram
    grid, x_edges, y_edges = np.histogram2d(
        original_array, checkpoint_array, 
        bins=bins, range=[[-1, 1], [-1, 1]]
    )
    
    # Convert to percentages
    total_samples = len(common_urls)
    grid_percentage = (grid / total_samples) * 100
    
    return grid_percentage, x_edges, y_edges

def plot_probability_change_grid(grid: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                               original_model: str, checkpoint_model: str,
                               judge: str, output_path: str, colormap: str = 'turbo'):
    """
    Plot the probability change grid.

    Args:
        grid: 2D array of percentages
        x_edges: X-axis bin edges (original Brier scores)
        y_edges: Y-axis bin edges (checkpoint Brier scores)
        original_model: Name of original model
        checkpoint_model: Name of checkpoint model
        judge: Judge model name
        output_path: Output file path
    """
    # Create figure with white background
    plt.figure(figsize=(12, 10), facecolor='white')

    # Create the heatmap with vibrant colors
    im = plt.imshow(grid.T, origin='lower', cmap=colormap, aspect='auto',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

    # Add colorbar
    cbar = plt.colorbar(im, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Percentage of Samples (\%)', fontsize=16, fontweight='bold')

    # Set labels
    plt.xlabel(f'{print_names.get(original_model, original_model)} Brier Score', 
              fontsize=18, fontweight='bold')
    plt.ylabel(f'{print_names.get(checkpoint_model, checkpoint_model)} Brier Score', 
              fontsize=18, fontweight='bold')

    # Set title
    plt.title(f'Brier Score Change: {print_names.get(original_model, original_model)} → {print_names.get(checkpoint_model, checkpoint_model)}\n({judge} Judge)\n', 
              fontsize=20, fontweight='bold', pad=20)

    # Add grid lines at bin edges to border each cell
    plt.grid(True, alpha=0.5, linestyle='-', color='darkgrey', linewidth=1)

    # Set custom grid ticks to match bin edges
    plt.xticks(x_edges, fontsize=14)
    plt.yticks(y_edges, fontsize=14)

    # Add text annotations to cells
    # Create a meshgrid for proper cell center positioning
    x_centers = [(x_edges[i] + x_edges[i+1]) / 2 for i in range(len(x_edges)-1)]
    y_centers = [(y_edges[j] + y_edges[j+1]) / 2 for j in range(len(y_edges)-1)]

    for i in range(grid.shape[0]):  # x-axis bins (original scores)
        for j in range(grid.shape[1]):  # y-axis bins (checkpoint scores)
            percentage = grid[i, j]
            if percentage > 0.1:  # Only show text for cells with >0.1% to avoid clutter
                # Get cell center coordinates
                x_pos = x_centers[i]
                y_pos = y_centers[j]

                # Use black text with white background for maximum contrast
                plt.text(x_pos, y_pos, f'{percentage:.1f}%', 
                        ha='center', va='center', fontsize=20, fontweight='bold',
                        color='black', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

    # Set axis limits first (keep original grid area)
    plt.xlim(x_edges[0], x_edges[-1])
    plt.ylim(y_edges[0], y_edges[-1])

    # Add row sums (sum across columns for each row) - positioned outside the grid
    row_sums = np.sum(grid, axis=0)  # Sum across x-axis (columns) for each y-bin (row)
    for j in range(len(row_sums)):
        if row_sums[j] > 0:
            y_pos = y_centers[j]
            plt.text(1.01, (y_pos - y_edges[0]) / (y_edges[-1] - y_edges[0]), f'{row_sums[j]:.1f}', 
                    ha='left', va='center', fontsize=16, fontweight='bold',
                    color='black', transform=plt.gca().transAxes)

    # Add column sums (sum across rows for each column) - positioned outside the grid
    col_sums = np.sum(grid, axis=1)  # Sum across y-axis (rows) for each x-bin (column)
    for i in range(len(col_sums)):
        if col_sums[i] > 0:
            x_pos = x_centers[i]
            plt.text((x_pos - x_edges[0]) / (x_edges[-1] - x_edges[0]), 1.01, f'{col_sums[i]:.1f}', 
                    ha='center', va='bottom', fontsize=16, fontweight='bold',
                    color='black', transform=plt.gca().transAxes)

    # Add reference lines at 0 for both axes
    plt.axhline(y=0, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.7, linewidth=2)

    # Add diagonal line for reference (no change)
    min_val = min(x_edges[0], y_edges[0])
    max_val = max(x_edges[-1], y_edges[-1])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, 
             label='No Change')
    plt.legend(fontsize=14)

    # ---- Print sum of probabilities in each quadrant ----
    # Quadrants: 
    # Q1: x in [-1,0), y in [-1,0)
    # Q2: x in [0,1], y in [-1,0)
    # Q3: x in [-1,0), y in [0,1]
    # Q4: x in [0,1], y in [0,1]
    # Find bin indices for 0 in x and y
    x_zero_idx = np.searchsorted(x_edges, 0, side='right') - 1
    y_zero_idx = np.searchsorted(y_edges, 0, side='right') - 1

    # grid[i, j]: i = x bin, j = y bin
    # Q1: x: 0..x_zero_idx-1, y: 0..y_zero_idx-1
    # Q2: x: x_zero_idx.., y: 0..y_zero_idx-1
    # Q3: x: 0..x_zero_idx-1, y: y_zero_idx..
    # Q4: x: x_zero_idx.., y: y_zero_idx..

    # But since grid.shape[0] = len(x_edges)-1, grid.shape[1] = len(y_edges)-1
    # So for y, j in 0..y_zero_idx-1 is y < 0, j in y_zero_idx.. is y >= 0

    # Slicing: [start:stop] includes start, excludes stop
    # So for x: 0:x_zero_idx is x < 0, x_zero_idx: is x >= 0
    # For y: 0:y_zero_idx is y < 0, y_zero_idx: is y >= 0

    Q1 = grid[0:x_zero_idx, 0:y_zero_idx]
    Q2 = grid[x_zero_idx:, 0:y_zero_idx]
    Q3 = grid[0:x_zero_idx, y_zero_idx:]
    Q4 = grid[x_zero_idx:, y_zero_idx:]

    Q1_sum = Q1.sum() if Q1.size > 0 else 0.0
    Q2_sum = Q2.sum() if Q2.size > 0 else 0.0
    Q3_sum = Q3.sum() if Q3.size > 0 else 0.0
    Q4_sum = Q4.sum() if Q4.size > 0 else 0.0

    print("Quadrant sums (percent of samples):")
    print(f"  Q1 (x<0, y<0): {Q1_sum:.2f}%")
    print(f"  Q2 (x>=0, y<0): {Q2_sum:.2f}%")
    print(f"  Q3 (x<0, y>=0): {Q3_sum:.2f}%")
    print(f"  Q4 (x>=0, y>=0): {Q4_sum:.2f}%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Probability change plot saved to {output_path}")
    plt.close()

def main():
    args = parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.original_file):
        print(f"Error: Original file {args.original_file} does not exist")
        return
    
    if not os.path.exists(args.checkpoint_file):
        print(f"Error: Checkpoint file {args.checkpoint_file} does not exist")
        return
    
    # Extract model names
    original_model = extract_model_name_from_filename(args.original_file)
    checkpoint_model = extract_model_name_from_filename(args.checkpoint_file)
    
    print(f"Original model: {original_model}")
    print(f"Checkpoint model: {checkpoint_model}")
    print(f"Judge: {args.judge}")
    
    # Get Brier scores for both models
    print(f"\nLoading original model scores from {args.original_file}")
    original_scores = get_sample_brier_scores(args.original_file, args.judge)
    
    print(f"\nLoading checkpoint model scores from {args.checkpoint_file}")
    checkpoint_scores = get_sample_brier_scores(args.checkpoint_file, args.judge)
    
    if not original_scores or not checkpoint_scores:
        print("No valid data found in one or both files")
        return
    
    # Create the change grid
    print(f"\nCreating probability change grid with {args.bins} bins")
    grid, x_edges, y_edges = create_brier_change_grid(
        original_scores, checkpoint_scores, args.bins
    )
    
    if grid is None:
        print("Failed to create grid")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    output_filename = f"brier_change_{original_model}_to_{checkpoint_model}_{args.judge}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Plot the grid
    plot_probability_change_grid(grid, x_edges, y_edges, 
                               original_model, checkpoint_model,
                               args.judge, output_path, args.colormap)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Original model: {original_model}")
    print(f"Checkpoint model: {checkpoint_model}")
    print(f"Judge: {args.judge}")
    print(f"Common samples: {len(set(original_scores.keys()) & set(checkpoint_scores.keys()))}")
    
    # Calculate some additional statistics
    common_urls = set(original_scores.keys()) & set(checkpoint_scores.keys())
    if common_urls:
        original_values = [original_scores[url] for url in common_urls]
        checkpoint_values = [checkpoint_scores[url] for url in common_urls]
        
        original_mean = np.mean(original_values)
        checkpoint_mean = np.mean(checkpoint_values)
        improvement = checkpoint_mean - original_mean
        
        print(f"Original mean Brier: {original_mean:.4f}")
        print(f"Checkpoint mean Brier: {checkpoint_mean:.4f}")
        print(f"Improvement: {improvement:.4f} ({'better' if improvement > 0 else 'worse'})")
        
        # Count improvements
        improvements = sum(1 for i, url in enumerate(common_urls) 
                          if checkpoint_scores[url] > original_scores[url])
        improvement_pct = (improvements / len(common_urls)) * 100
        
        print(f"Samples improved: {improvements}/{len(common_urls)} ({improvement_pct:.1f}%)")

if __name__ == "__main__":
    main()
