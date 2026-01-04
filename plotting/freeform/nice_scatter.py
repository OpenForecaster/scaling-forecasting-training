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
from matplotlib.transforms import Bbox

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
    'deepseek-chat-v3-0324': 'V3',
    'DeepSeek-V3-0324': 'V3',
    'deepseek-r1-0528': 'R1',
    'llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'llama-4-maverick': 'Maverick',
    'llama-4-scout': 'Scout',
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
    parser = argparse.ArgumentParser(description="Scatter plot of Accuracy (y) vs nBrier (x) per model")
    parser.add_argument("--input_dir", type=str,
                        default="/fast/nchandak/forecasting/evals/freeform/manual/fivenewscombined-retrieval_1000/fivenews-test",
                        help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/iclr/test",
                        help="Output directory for the scatter plot")
    parser.add_argument("--judge", type=str, default="Qwen3_4B",
                        help="Judge model name for score field, e.g., Qwen3_4B")
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


def determine_family(model_name: str) -> str:
    name = model_name.lower()
    if 'qwen' in name:
        if re.match(r'^[Qq]wen3-1\.7[bB]-', model_name):
            return 'Trained on \\texttt{OpenForesight}'
        if re.match(r'^[Qq]wen3-4[bB]-', model_name):
            return 'Trained on \\texttt{OpenForesight}'
        if re.match(r'^[Qq]wen3-8[bB]-', model_name):
            return 'Trained on \\texttt{OpenForesight}'
        
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
    }


def qwen_trained_marker(model_name: str) -> str:
    # Use special shapes for trained Qwen3-XB- models (use a unified diamond marker for trained variants)
    if re.match(r'^[Qq]wen3-1\.7[bB]-', model_name):
        return 'D'
    if re.match(r'^[Qq]wen3-4[bB]-', model_name):
        return 'D'
    if re.match(r'^[Qq]wen3-8[bB]-', model_name):
        return 'D'
    return 'o'


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def compute_model_metrics(input_dir: str, judge: str) -> Dict[str, Dict[str, Any]]:
    model_metrics: Dict[str, Dict[str, Any]] = {}

    jsonl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.jsonl')]

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        model_name, num_generations = extract_model_info_from_filename(filename)
        if "withbinary" in model_name:
            parts = model_name.split("-")
            model_name = "-".join([part for part in parts if "with" not in part and "binary" not in part])
        model_key = f"{model_name}"

        data = load_jsonl_file(file_path)

        judge_field = f"score_{judge}"
        has_judge_field = any(judge_field in item for item in data)
        if not has_judge_field:
            print(f"No judge field found in {file_path}")
            continue

        mean_brier, brier_se = calculate_model_brier_statistics(data, num_generations, judge_field)
        mean_acc, acc_se = calculate_model_accuracy_statistics(data, num_generations, judge_field)

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


def _place_label_non_overlapping(ax, fig, x: float, y: float, text: str, color: str, fontsize: int = 30,
                                 marker_size_pts: int = 26,
                                 marker_bboxes: List[Bbox] = None) -> Tuple[Any, int, int]:
    """
    Place a text label near (x, y) in data coords with an offset chosen to avoid
    overlapping existing labels and marker glyphs. Uses an expanding ring of
    8-direction candidate offsets and rejects placements whose screen-space
    bounding boxes intersect any existing label bbox or any marker bbox.

    Returns: (annotation, dx_points, dy_points)
    """
    if marker_bboxes is None:
        marker_bboxes = []

    # Initialize store for placed label bboxes on the axes
    if not hasattr(ax, '_placed_label_bboxes'):
        ax._placed_label_bboxes = []

    renderer = fig.canvas.get_renderer()

    base = max(18, int(marker_size_pts * 0.9))
    radii = [base, base + 8, base + 16, base + 24, base + 32, base + 44, base + 58, base + 74, base + 92]

    def candidate_offsets(r: int) -> List[Tuple[int, int]]:
        d = int(max(10, r * 0.6))
        return [
            (0, r),    # N
            (r, d),    # NE
            (r, 0),    # E
            (r, -d),   # SE
            (0, -r),   # S
            (-r, -d),  # SW
            (-r, 0),   # W
            (-r, d),   # NW
        ]

    # Try candidates in increasing radius
    for radius in radii:
        for dx, dy in candidate_offsets(radius):
            ann = ax.annotate(
                text,
                (x, y),
                textcoords='offset points',
                xytext=(dx, dy),
                ha='center', va='bottom', fontsize=fontsize, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none')
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.08, 1.20)
            conflict_labels = any(bbox.overlaps(b) for b in ax._placed_label_bboxes)
            conflict_markers = any(bbox.overlaps(m) for m in marker_bboxes)
            if not (conflict_labels or conflict_markers):
                ax._placed_label_bboxes.append(bbox)
                return ann, dx, dy
            ann.remove()

    # Fallback: place directly above with a larger offset
    dx, dy = 0, base + 24
    ann = ax.annotate(
        text,
        (x, y),
        textcoords='offset points',
        xytext=(dx, dy),
        ha='center', va='bottom', fontsize=fontsize, fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none')
    )
    fig.canvas.draw()
    bbox = ann.get_window_extent(renderer=renderer).expanded(1.08, 1.20)
    ax._placed_label_bboxes.append(bbox)
    return ann, dx, dy


def _relax_label_collisions(ax, fig, placed: List[Tuple[Any, int, int]], marker_bboxes: List[Bbox],
                            max_iter: int = 100, step_px: float = 6.0) -> None:
    """
    Iteratively adjust label offsets to remove any remaining overlaps between labels,
    and between labels and markers. Offsets are in points; we move them by step_px in
    display pixels converted to points each iteration.
    """
    if not placed:
        return
    px_per_point = fig.dpi / 72.0

    renderer = fig.canvas.get_renderer()

    for _ in range(max_iter):
        moved_any = False
        # Recompute bboxes and centers
        bboxes = []
        centers = []
        for ann, dx, dy in placed:
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.08, 1.20)
            bboxes.append(bbox)
            c = bbox.get_points().mean(axis=0)
            centers.append(c)

        # Label-label collisions
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                if not bboxes[i].overlaps(bboxes[j]):
                    continue
                # Push apart along the line connecting centers
                direction = centers[i] - centers[j]
                norm = np.linalg.norm(direction) or 1.0
                ux, uy = direction / norm
                # Move each label half a step in opposite directions
                delta_pts = (step_px / px_per_point) * 0.7
                di = (ux * delta_pts, uy * delta_pts)
                dj = (-ux * delta_pts, -uy * delta_pts)
                # Update offsets
                ann_i, dxi, dyi = placed[i]
                ann_j, dxj, dyj = placed[j]
                dxi += di[0]
                dyi += di[1]
                dxj += dj[0]
                dyj += dj[1]
                ann_i.set_position((dxi, dyi))
                ann_j.set_position((dxj, dyj))
                placed[i] = (ann_i, dxi, dyi)
                placed[j] = (ann_j, dxj, dyj)
                moved_any = True

        # Label-marker collisions
        if marker_bboxes:
            # Refresh after possible moves
            fig.canvas.draw()
            bboxes = [ann.get_window_extent(renderer=renderer).expanded(1.08, 1.20) for ann, _, _ in placed]
            centers = [bbox.get_points().mean(axis=0) for bbox in bboxes]
            for i, (ann, dx, dy) in enumerate(placed):
                collided = False
                for mbox in marker_bboxes:
                    if not bboxes[i].overlaps(mbox):
                        continue
                    collided = True
                    # Move label away from marker center
                    mc = mbox.get_points().mean(axis=0)
                    direction = centers[i] - mc
                    norm = np.linalg.norm(direction) or 1.0
                    ux, uy = direction / norm
                    delta_pts = (step_px / px_per_point)
                    dx += ux * delta_pts
                    dy += uy * delta_pts
                    ann.set_position((dx, dy))
                if collided:
                    placed[i] = (ann, dx, dy)
                    moved_any = True

        fig.canvas.draw()
        # If nothing moved, we are done
        if not moved_any:
            break


def plot_scatter(metrics: Dict[str, Dict[str, Any]], output_path: str, title: str) -> None:
    if not metrics:
        print("No metrics to plot")
        return

    families = family_color_map()

    # Prepare plot (bigger canvas)
    fig, ax = plt.subplots(figsize=(12, 12))

    handles = {}
    
    # Compute axis limits up-front for stable transforms
    xs = [info['mean_brier'] for info in metrics.values()]
    ys = [info['mean_accuracy'] for info in metrics.values()]
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))
    if np.isclose(maxx - minx, 0.0):
        minx, maxx = minx - 0.05, maxx + 0.05
    if np.isclose(maxy - miny, 0.0):
        miny, maxy = miny - 0.5, maxy + 0.5
    xpad = 0.1 * (maxx - minx)
    ypad = 0.1 * (maxy - miny)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)

    # Ensure renderer is initialized for bbox calculations
    fig.canvas.draw()

    # Build marker bounding boxes in display coords to avoid label overlap with points
    marker_size_pts = 26
    size_pix = marker_size_pts * fig.dpi / 72.0
    half = (size_pix * 0.9) / 2.0  # a bit tighter than full size
    marker_bboxes: List[Bbox] = []
    for x, y in zip(xs, ys):
        px, py = ax.transData.transform((x, y))
        marker_bboxes.append(Bbox.from_bounds(px - half, py - half, half * 2, half * 2))

    # Reset previously placed label bboxes
    if hasattr(ax, '_placed_label_bboxes'):
        ax._placed_label_bboxes = []

    placed_labels: List[Tuple[Any, int, int]] = []

    # Plot each model
    for model_key, info in metrics.items():
        x = info['mean_brier']
        y = info['mean_accuracy']
        fam = determine_family(model_key)
        color = families.get(fam, families['Other'])
        marker = qwen_trained_marker(model_key)
        label = print_names.get(model_key, model_key)

        # Plot point (marker-only aesthetics)
        sc = ax.errorbar(
            x, y,
            fmt=marker,
            markersize=26,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=2,
            ecolor=color,
            elinewidth=0,
            alpha=0.95,
            label=None
        )

        # Place a non-overlapping label near the point (avoid markers and prior labels)
        ann, dx, dy = _place_label_non_overlapping(ax, fig, x, y, label, color, fontsize=30, marker_size_pts=26, marker_bboxes=marker_bboxes)
        placed_labels.append((ann, dx, dy))

        # For legend by family, store one handle per family (use the marker from errorbar)
        if fam not in handles:
            handles[fam] = sc[0] if isinstance(sc, tuple) else sc

    # Final relaxation pass to resolve any residual overlaps in crowded regions
    _relax_label_collisions(ax, fig, placed_labels, marker_bboxes, max_iter=120, step_px=8.0)

    ax.set_xlabel('Brier Score (higher is better)', fontsize=28, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=28, fontweight='bold')

    # Keep default xticks but ensure they render after limits
    ax.set_xticks(ax.get_xticks())
    
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.tick_params(axis='both', labelsize=30, length=6, width=1.2)

    # No title as requested
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
    output_path = os.path.join(args.output_dir, f"scatter_brier_accuracy_{dataset_suffix}_{args.judge}.png")

    metrics = compute_model_metrics(args.input_dir, args.judge)

    title = f"Accuracy vs nBrier | Dataset: {dataset_suffix} | Judge: {args.judge}"
    plot_scatter(metrics, output_path, title)


if __name__ == "__main__":
    main() 