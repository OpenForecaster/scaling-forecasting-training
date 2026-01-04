#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import glob
from collections import defaultdict
import re

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
    'Qwen3_'
    'deepseek-chat-v3-0324': 'DeepSeek V3',
    'DeepSeek-V3-0324': 'DeepSeek V3',
    'llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'claude-3.5-haiku': 'Claude 3.5 Haiku',
    'gpt-4o': 'GPT 4o',
    'gpt-4o-mini': 'GPT 4o Mini',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot pass@k curves for freeform forecasting evaluation")
    parser.add_argument("--input-dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/dw_19754/",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Output directory for plots")
    parser.add_argument("--max-k", type=int, default=16,
                       help="Maximum value of k to plot")
    parser.add_argument("--judge", type=str, default="Qwen3_4B",
                       help="Judge model name (fixed for all models)")
    parser.add_argument("--majority", action="store_true",
                       help="Plot majority@k for with_article files as dashed horizontal lines")
    parser.add_argument("--continuous", action="store_true",
                       help="Use continuous_score fields instead of score_ fields")
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
    Extract model name and context type from filename.
    
    Expected format: ModelName_evals_size_N_generations_M_context_type.jsonl
    Returns: (model_name, context_type, num_generations)
    """
    # Remove .jsonl extension
    name_without_ext = filename.replace('.jsonl', '')
    
    # Extract context type (with_article or no_article)
    if 'with_article' in name_without_ext:
        context_type = 'with_article'
    elif 'no_article' in name_without_ext:
        context_type = 'no_article'
    else:
        context_type = 'unknown'
    
    # Extract model name (everything before _evals or _eval)
    model_match = re.match(r'([^_]+(?:_[^_]*?)?(?:-\d+\.?\d*[bB])?)', name_without_ext)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = name_without_ext.split('_')[0]
    
    # Extract number of generations
    gen_match = re.search(r'generations_(\d+)', name_without_ext)
    num_generations = int(gen_match.group(1)) if gen_match else 1
    
    if model_name[-1] == '_':
        model_name = model_name[:-1]
    
    return model_name, context_type, num_generations

def calculate_pass_at_k_improved(data, k, judge_field):
    """Calculate pass@k accuracy with proper question grouping."""
    # Group data by question idx/id
    question_data = defaultdict(list)
    
    for item in data:
        question_id = item.get("idx", item.get("question_id"))
        if question_id is not None and judge_field in item:
            score_value = item[judge_field]
            if isinstance(score_value, list):
                question_data[question_id].extend(score_value)
            elif isinstance(score_value, (int, float)):
                question_data[question_id].append(score_value)
            elif isinstance(score_value, str):
                try:
                    question_data[question_id].append(int(score_value))
                except ValueError:
                    continue
    
    # Calculate pass@k for each question
    pass_count = 0
    total_questions = len(question_data)
    
    for question_id, scores in question_data.items():
        # Take the first k attempts for this question
        first_k_attempts = list(map(float, scores[:k]))
        
        # Check if any of the first k attempts were correct (score = 1)
        # if any(attempt == 1 for attempt in first_k_attempts):
        #     pass_count += 1
        
        pass_count += max(first_k_attempts)
    
    return pass_count / total_questions if total_questions > 0 else 0.0

def calculate_majority_accuracy(data, judge_field):
    """Calculate majority vote accuracy using the judge scores at the majority answer indices."""
    from collections import Counter
    
    # Group data by question idx/id with their judge scores
    question_data = defaultdict(lambda: {'answers': [], 'scores': [], 'item': None})
    
    for item in data:
        question_id = item.get("idx", item.get("question_id"))
        if question_id is not None and "extracted_answer" in item and judge_field in item:
            extracted_answers = item.get("extracted_answer", [])
            judge_scores = item.get(judge_field, [])
            
            if isinstance(extracted_answers, list) and isinstance(judge_scores, list):
                question_data[question_id]['answers'] = extracted_answers
                question_data[question_id]['scores'] = judge_scores
                question_data[question_id]['item'] = item
    
    # Calculate majority vote accuracy for each question
    total_score = 0
    total_questions = len(question_data)
    
    for question_id, data_dict in question_data.items():
        answers = data_dict['answers']
        scores = data_dict['scores']
        
        if not answers or not scores:
            continue
            
        # Convert all answers to lowercase and count occurrences
        lowercase_answers = [str(ans).lower().strip() for ans in answers if ans is not None]
        
        if not lowercase_answers:
            continue
            
        # Find the most common answer
        answer_counts = Counter(lowercase_answers)
        majority_answer = answer_counts.most_common(1)[0][0]
        
        # Find any index where the majority answer occurs
        majority_index = None
        for i, ans in enumerate(answers):
            if ans is not None and str(ans).lower().strip() == majority_answer:
                majority_index = i
                break
        
        # Take the judge score at that index
        if majority_index is not None and majority_index < len(scores):
            total_score += scores[majority_index]
    
    return total_score / total_questions if total_questions > 0 else 0.0

def get_model_data(input_dir, judge, max_k=16, calculate_majority=False, use_continuous=False):
    """Get pass@k data for all models in the input directory."""
    model_data = {}
    
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        
        # Extract model info from filename
        model_name, context_type, num_generations = extract_model_info_from_filename(filename)
        
        if '8b' in filename.lower() and context_type == 'with_article':
            continue
        
        # Create a unique key for model+context combination
        model_key = f"{model_name}_{context_type}"
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name} ({context_type})")
        
        # Check if judge field exists
        if use_continuous:
            judge_field = f"continuous_score_{judge}"
        else:
            judge_field = f"score_{judge}"
        available_fields = set()
        for item in data:
            for key in item.keys():
                if key.startswith("score_") or key.startswith("continuous_score_"):
                    available_fields.add(key)
        
        if judge_field not in available_fields:
            print(f"  Warning: {judge_field} not found, available fields: {available_fields}")
            # If the requested field is not found, try the other one as fallback
            if use_continuous:
                fallback_field = f"score_{judge}"
            else:
                fallback_field = f"continuous_score_{judge}"
            if fallback_field in available_fields:
                print(f"  Using fallback field: {fallback_field}")
                judge_field = fallback_field
            # else: keep judge_field as is, will likely result in 0s
        
        # Calculate maximum possible k from the data
        max_possible_k = min(max_k, num_generations)
        
        # Calculate pass@k for different k values (powers of 2)
        k_values = [2 ** i for i in range(int(np.log2(max_possible_k)) + 1) if 2 ** i <= max_possible_k]
        if 1 not in k_values:
            k_values = [1] + k_values
        k_values = sorted(set(k_values))
        
        scores = []
        
        for k in k_values:
            score = calculate_pass_at_k_improved(data, k, judge_field)
            scores.append(score)
        
        # Calculate majority accuracy once if requested and this is a with_article file
        majority_accuracy = None
        if calculate_majority and context_type == 'with_article':
            majority_accuracy = calculate_majority_accuracy(data, judge_field)
        
        model_data[model_key] = {
            'model_name': model_name,
            'context_type': context_type,
            'k_values': k_values,
            'scores': scores,
            'majority_accuracy': majority_accuracy,
            'num_samples': len(data),
            'num_generations': num_generations
        }
        
        if calculate_majority and context_type == 'with_article' and majority_accuracy is not None:
            print(f"  {model_name} ({context_type}): {len(data)} samples, pass@k: {[f'{s:.3f}' for s in scores]}, majority: {majority_accuracy:.3f}")
        else:
            print(f"  {model_name} ({context_type}): {len(data)} samples, scores: {[f'{s:.3f}' for s in scores]}")
    
    return model_data

def extract_dataset_name(input_dir):
    """Extract and format dataset name from input directory path."""
    # Extract the last part of the path (e.g., dw_21317, dw_30)
    dataset_part = os.path.basename(input_dir.rstrip('/'))
    
    # Format it nicely
    if dataset_part.startswith('dw_'):
        return f"DW 2024-25 ({dataset_part.split('_')[1]} Questions)"
    else:
        return f"{dataset_part} Forecasting"

def plot_pass_at_k_curves_multi_models(model_data, judge, output_path, dataset_name=None, show_majority=False, use_continuous=False):
    """Plot pass@k curves for multiple models with different contexts."""
    
    plt.figure(figsize=(12, 10))
    
    # Define colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Group by model name to assign consistent colors
    models_by_name = defaultdict(list)
    for model_key, data in model_data.items():
        models_by_name[data['model_name']].append((model_key, data))
    
    # Plot each model and context combination
    color_idx = 0
    
    for model_name, model_variants in models_by_name.items():
        base_color = colors[color_idx % len(colors)]
        color_idx += 1
        
        for model_key, data in model_variants:
            # Format model name for display
            display_name = print_names.get(model_name, model_name)
            
            # Add context information to legend
            if data['context_type'] == 'with_article':
                legend_label = f"{display_name} (with article)"
                if show_majority:
                    legend_label = f"{display_name} (with article) (majority)"
                linestyle = '-'
                alpha = 0.8
            elif data['context_type'] == 'no_article':
                legend_label = f"{display_name} (no article)"
                linestyle = '--'
                alpha = 0.6
            else:
                legend_label = f"{display_name} ({data['context_type']})"
                linestyle = '-'
                alpha = 0.8
            
            # Use different markers for different contexts
            if data['context_type'] == 'with_article':
                marker = 'o'
            else:
                marker = 's'
            
            # Determine what to plot based on majority flag and context type
            if show_majority:
                # When majority flag is on:
                # - Plot pass@k only for non-"with_article" files
                # - Plot majority only for "with_article" files
                if data['context_type'] != 'with_article':
                    # Plot pass@k curve for non-with_article files
                    plt.plot(data['k_values'], data['scores'], 
                            linestyle=linestyle, marker=marker,
                            linewidth=3.5, markersize=12, 
                            label=legend_label, 
                            color=base_color, alpha=alpha)
                
                elif data['context_type'] == 'with_article' and data.get('majority_accuracy') is not None:
                    # Plot majority accuracy as horizontal line for with_article files
                    majority_acc = data['majority_accuracy']
                    
                    # Plot horizontal line across the entire k range (solid and thick)
                    plt.axhline(y=majority_acc, color=base_color, linestyle='-', 
                               linewidth=4.0, alpha=0.8, 
                               xmin=0, xmax=1)  # xmin=0, xmax=1 means full width
                    
                    # Add a legend entry for majority accuracy (only once per model)
                    if model_variants.index((model_key, data)) == 0:  # Only for the first variant of this model
                        majority_legend_label = f"{display_name} (with article) (majority)"
                        plt.plot([], [], color=base_color, linestyle='-', linewidth=4.0, alpha=0.8,
                               label=majority_legend_label)
            else:
                # When majority flag is off, plot pass@k curves for all files
                plt.plot(data['k_values'], data['scores'], 
                        linestyle=linestyle, marker=marker,
                        linewidth=3.5, markersize=12, 
                        label=legend_label, 
                        color=base_color, alpha=alpha)
    
    plt.xlabel('Number of Attempts (k)', fontsize=24, fontweight='bold')
    plt.ylabel('Pass@k Accuracy', fontsize=24, fontweight='bold')
    
    # Set title with dataset name and score type
    score_type_str = "Continuous" if use_continuous else "Binary"
    if dataset_name:
        plt.title(f'{dataset_name} - Pass@k Performance ({score_type_str} Score)', fontsize=28, fontweight='bold', pad=30)
    else:
        plt.title(f'Pass@k Curves - {judge} Judge ({score_type_str} Score)', fontsize=28, fontweight='bold', pad=30)
    
    plt.legend(fontsize=20, frameon=True, fancybox=True, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis to log scale with powers of 2
    if model_data:
        all_k_values = set()
        for data in model_data.values():
            all_k_values.update(data['k_values'])
        all_k_values = sorted(all_k_values)
        
        plt.xscale('log', base=2)
        plt.xticks(all_k_values, [str(k) for k in all_k_values], fontsize=20)
    
    plt.ylim(0, 1.0)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.gca().tick_params(axis='both', which='major', labelsize=20)
    
    # Add some padding around the plot
    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Pass@k plot saved to {output_path}")
    plt.close()

def main():
    args = parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Extract dataset name from input directory
    dataset_name = extract_dataset_name(args.input_dir)
    
    print(f"Processing directory: {args.input_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Judge: {args.judge}")
    print(f"Max k: {args.max_k}")
    print(f"Using {'continuous_score_' if args.continuous else 'score_'} fields for evaluation")
    
    # Get data for all models
    print(f"\nLoading data for all models in {args.input_dir}")
    model_data = get_model_data(
        args.input_dir,
        args.judge,
        args.max_k,
        calculate_majority=args.majority,
        use_continuous=args.continuous
    )
    
    if not model_data:
        print("No valid model data found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    dataset_suffix = os.path.basename(args.input_dir.rstrip('/'))
    majority_suffix = "_with_majority" if args.majority else ""
    continuous_suffix = "_continuous" if args.continuous else ""
    output_filename = f"passk_{dataset_suffix}_{args.judge}{majority_suffix}{continuous_suffix}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Plot the pass@k curves
    plot_pass_at_k_curves_multi_models(
        model_data,
        args.judge,
        output_path,
        dataset_name,
        show_majority=args.majority,
        use_continuous=args.continuous
    )
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Dataset: {dataset_name}")
    print(f"Judge: {args.judge}")
    print(f"Total model variants: {len(model_data)}")
    
    for model_key, data in model_data.items():
        context_info = f" ({data['context_type']})" if data['context_type'] != 'unknown' else ""
        print(f"  {data['model_name']}{context_info}: {data['num_samples']} samples, {data['num_generations']} generations")

if __name__ == "__main__":
    main()
