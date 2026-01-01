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
    parser.add_argument("--input_dir", type=str, 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/dw_19805/",
                       help="Directory containing evaluation JSONL files")
    parser.add_argument("--output-dir", type=str, default="plots/passk",
                       help="Output directory for plots")
    parser.add_argument("--max-k", type=int, default=16,
                       help="Maximum value of k to plot")
    parser.add_argument("--judge", type=str, default="Qwen3_4B",
                       help="Judge model name (fixed for all models)")
    parser.add_argument("--majority", action="store_true",
                       help="Plot majority@k for with_article files as dashed horizontal lines")
    parser.add_argument("--continuous", action="store_true",
                       help="Use continuous_score fields instead of score_ fields")
    parser.add_argument("--source-dir", type=str,
                       default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/recent/qgen/",
                       help="Directory containing source JSONL files with final_question_valid field")
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
    
    context_type = ''
    # Extract context type (with_article or no_article)
    if 'with_article' in name_without_ext:
        context_type = 'with_article'
    elif 'no_article' in name_without_ext:
        context_type = 'no_article'
    # else:
    #     context_type = 'unknown'
    
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
        
    # parts = model_name.split('-')
    # relevant_parts = []
    # for part in parts:
    #     # only append if it has a digit and a character
    #     if not part.isdigit() and not part.isalpha():
    #         relevant_parts.append(part)
            
    # model_name = '-'.join(relevant_parts)
    model_name = model_name[:60]
    
    return model_name, context_type, num_generations

def find_corresponding_source_file(eval_filename, source_dir):
    """
    Find the corresponding source file for an evaluation file.
    
    Args:
        eval_filename: e.g., "deepseek-chat-v3-0324_evals_size_N_generations_M_context_type.jsonl"
        source_dir: Directory containing source files
        
    Returns:
        Path to the corresponding source file, or None if not found
    """
    # Extract model name and dataset from eval filename
    model_name, _, _ = extract_model_info_from_filename(eval_filename)
    
    # Extract dataset identifier (like dw_19805)
    dataset_match = re.search(r'(_\d+_)', eval_filename)
    if not dataset_match:
        print(f"Warning: Could not extract dataset from {eval_filename}")
        return None
    
    dataset_id = dataset_match.group(1)[1:-1]
    
    # Look for source files with pattern: model_name_dataset_free_N.jsonl
    pattern = f"{model_name}_{dataset_id}_free_*.jsonl"
    source_files = glob.glob(os.path.join(source_dir, pattern))
    
    if not source_files:
        print(f"Warning: No source file found for pattern {pattern}")
        return None
    
    # If multiple files found, take the first one (or could be more sophisticated)
    if len(source_files) > 1:
        print(f"Warning: Multiple source files found for {pattern}, using {source_files[0]}")
    
    return source_files[0]

def load_validity_data(source_file_path):
    """
    Load validity data from source file.
    
    Args:
        source_file_path: Path to source JSONL file
        
    Returns:
        Dictionary mapping article URLs to validity status
    """
    validity_data = {}
    
    if not source_file_path or not os.path.exists(source_file_path):
        return validity_data
    
    try:
        with open(source_file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    url = item.get('url', '')
                    final_question_valid = item.get('final_question_valid', 1)  # Default to valid if not present
                    
                    if url:
                        validity_data[url] = final_question_valid
                        
                except json.JSONDecodeError:
                    continue
                    
        print(f"  Loaded validity data for {len(validity_data)} articles from {os.path.basename(source_file_path)}")
        
    except Exception as e:
        print(f"  Error loading validity data from {source_file_path}: {e}")
    
    return validity_data


# --- Begin: Guardian URL Filtering ---

def load_guardian_url_set(guardian_jsonl_path):
    """Load theguardian cleaned file and return a set of URLs."""
    url_set = set()
    if not os.path.exists(guardian_jsonl_path):
        print(f"Guardian file {guardian_jsonl_path} not found!")
        return url_set
    with open(guardian_jsonl_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                url = obj.get("url", None)
                if url:
                    url_set.add(url)
            except Exception:
                continue
    print(f"Loaded {len(url_set)} URLs from {guardian_jsonl_path}")
    return url_set

GUARDIAN_JSONL_PATH = "/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian_207_free_3_cleaned.jsonl"
GUARDIAN_URL_SET = load_guardian_url_set(GUARDIAN_JSONL_PATH)

def filter_data_by_guardian_urls(data, url_set):
    """Filter data to only keep entries whose article_url is in the guardian url set."""
    filtered = []
    for item in data:
        url = item.get("article_url", None)
        if url and url in url_set:
            filtered.append(item)
    print(f"  Filtered by Guardian URLs: kept {len(filtered)}/{len(data)} items")
    return filtered

# --- End: Guardian URL Filtering ---

def filter_data_by_validity(data, validity_data):
    """
    Filter evaluation data to keep only articles where final_question_valid == 1.
    
    Args:
        data: List of evaluation items
        validity_data: Dictionary mapping URLs to validity status
        
    Returns:
        Filtered list of evaluation items
    """
    if not validity_data:
        print("  No validity data available, keeping all items")
        return data
    
    filtered_data = []
    total_items = len(data)
    
    for item in data:
        article_url = item.get('article_url', '')
        
        # Default to valid if URL not found in validity data
        is_valid = validity_data.get(article_url, 1)
        
        if is_valid == 1:
            filtered_data.append(item)
    
    filtered_count = len(filtered_data)
    excluded_count = total_items - filtered_count
    
    print(f"  Filtered: kept {filtered_count}/{total_items} items ({excluded_count} excluded due to invalid questions)")
    
    return filtered_data

def calculate_pass_at_k_improved(data, k, judge_field, use_continuous):
    """Calculate pass@k accuracy with proper question grouping."""
    # Group data by question idx/id
    question_data = defaultdict(list)
    
    for item in data:
        # answer_type = item.get("answer_type", "multiple_choice")
        
        # item_url = item.get("article_url", None)
        # if not item_url:
        #     item_url = item.get("url", None)
            
        # if item_url not in GUARDIAN_URL_SET:
        #     continue
        
        # if "numeric" in answer_type.lower() or "date" in answer_type.lower():
        #     continue
        
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
    
    print(f"Calculating pass@k for {len(question_data)} questions")
    # Calculate pass@k for each question
    pass_count = 0
    total_questions = len(question_data)
    
    for question_id, scores in question_data.items():
        # Take the first k attempts for this question
        scores_k = scores[:k]
        scores_k = [list(x.values())[0] for x in scores_k if isinstance(x, dict)]
        first_k_attempts = list(map(float, scores_k))
        
        # Check if any of the first k attempts were correct (score = 1)
        # if any(attempt == 1 for attempt in first_k_attempts):
        #     pass_count += 1
        
        mx = max(first_k_attempts)
        if use_continuous:
            pass_count += mx
        else:
            pass_count += 1 if mx >= 0.9999 else 0
    
    return pass_count / total_questions if total_questions > 0 else 0.0

def calculate_majority_accuracy(data, judge_field):
    """Calculate majority vote accuracy using the judge scores at the majority answer indices."""
    from collections import Counter
    
    # Group data by question idx/id with their judge scores
    question_data = defaultdict(lambda: {'answers': [], 'scores': [], 'item': None})
    
    for item in data:
        # item_url = item.get("article_url", None)
        # if not item_url:
        #     item_url = item.get("url", None)
            
        # if item_url not in GUARDIAN_URL_SET:
        #     continue
        
        question_id = item.get("idx", item.get("question_id"))
        answer_type = item.get("answer_type", "multiple_choice")
        
        # if "numeric" in answer_type.lower() or "date" in answer_type.lower():
        #     continue
        
        if question_id is not None and "extracted_answer" in item and judge_field in item:
            extracted_answers = item.get("extracted_answer", [])
            judge_scores = item.get(judge_field, [])
            
            if isinstance(extracted_answers, list) and isinstance(judge_scores, list):
                question_data[question_id]['answers'] = extracted_answers
                question_data[question_id]['scores'] = judge_scores
                question_data[question_id]['item'] = item
    
    print(f"Calculating majority accuracy for {len(question_data)} questions")
    
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

def get_model_data(input_dir, judge, source_dir=None, max_k=16, calculate_majority=False, use_continuous=False):
    """Get pass@k data for all models in the input directory."""
    model_data = {}
    
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Load validity data once for all files if source directory is provided
    validity_data = {}
    if source_dir and jsonl_files:
        # Extract dataset ID from any file in the directory
        first_filename = os.path.basename(jsonl_files[0])
        dataset_match = re.search(r'(_\d+_)', first_filename)
        if dataset_match:
            dataset_id = dataset_match.group(1)[1:-1]  # Remove surrounding underscores
            print(f"Extracted dataset ID: {dataset_id}")
            
            # Find a source file for this dataset (use any model that has this dataset)
            source_pattern = f"*_{dataset_id}_free_*.jsonl"
            source_files = glob.glob(os.path.join(source_dir, source_pattern))
            
            if source_files:
                source_file_path = source_files[0]  # Use the first matching source file
                print(f"Using source file for all evaluations: {os.path.basename(source_file_path)}")
                validity_data = load_validity_data(source_file_path)
            else:
                print(f"Warning: No source file found for dataset {dataset_id} in {source_dir}")
        else:
            print(f"Warning: Could not extract dataset ID from {first_filename}")
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        
        # Extract model info from filename
        model_name, context_type, num_generations = extract_model_info_from_filename(filename)
        
        if '8b' in filename.lower() and context_type == 'with_article':
            continue
        if '32b' in filename.lower():
            continue
        
        # if '-4b' not in filename.lower():
        #     continue
        
        # if 'thinking' in filename.lower():
        #     continue
        
        if 'instruct' in filename.lower():
            continue
        
        # if '4b' not in filename.lower():
        #     continue
        
        # Create a unique key for model+context combination
        if context_type:
            model_key = f"{model_name}_{context_type}"
        else:
            model_key = model_name
        
        # Load data for this file
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples for {model_name} ({context_type})")
        
        # Apply validity filtering if we have validity data
        if validity_data:
            data = filter_data_by_validity(data, validity_data)
        elif source_dir:
            print(f"  No validity data available, keeping all items")
        else:
            print(f"  No source directory provided, skipping validity filtering")
        
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
            score = calculate_pass_at_k_improved(data, k, judge_field, use_continuous)
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
            'num_generations': num_generations,
            'filtering_applied': bool(source_dir)
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
    
    to_keep = {}
    for k, v in model_data.items():
        print(k, v['scores'])
        if len(v['scores']) > 1:
            to_keep[k] = v
    model_data = to_keep
    
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
            if '1e-4' in display_name and ('8192' in display_name or '4096' in display_name):
                pos = display_name.find('8192')
                display_name = display_name[:pos-1] + " + SFT"
            if '4096' in display_name:
                pos = display_name.find('4096')
                display_name = display_name[:pos] + " \\textbf{4096}"
            
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
                legend_label = f"{display_name}"
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
                        linewidth=10, markersize=20, 
                        label=legend_label, 
                        color=base_color, alpha=alpha)
    
    plt.xlabel('Number of Attempts (\\emph{k})', fontsize=28, fontweight='bold', labelpad=11)
    plt.ylabel('Pass@\\emph{k} Accuracy (\\%)', fontsize=28, fontweight='bold', labelpad=12)
    
    # Set title with dataset name and score type
    score_type_str = "Continuous" if use_continuous else "Binary"
    # Check if we have any model data to determine if filtering was applied
    filtering_applied = any(data.get('filtering_applied', False) for data in model_data.values())
    filter_str = " (Valid Questions Only)" if filtering_applied else ""
    
    # if dataset_name:
    #     plt.title(f'{dataset_name} - Pass@k Performance ({score_type_str} Score){filter_str}', fontsize=28, fontweight='bold', pad=30)
    # else:
    #     plt.title(f'Pass@k Curves - {judge} Judge ({score_type_str} Score){filter_str}', fontsize=28, fontweight='bold', pad=30)
    
    
    # Add horizontal reference lines to show filtered data reaches performance faster
    # Accuracy: horizontal line at unfiltered checkpoint 300 performance, extending to checkpoint 100
    if 'Qwen3-4B' in model_data and 16 in model_data['Qwen3-4B']['k_values']:
        unfiltered_300_accuracy = model_data['Qwen3-4B']['scores'][4]
        plt.axhline(y=unfiltered_300_accuracy, color='black', linestyle='--', 
                   alpha=0.7, linewidth=3, xmin=0.24, xmax=0.94)  # xmax=0.33 corresponds to checkpoint 100 if max is 300
        # plt.text(1.7, unfiltered_300_accuracy + 0.01, f'3x less data', 
        #         fontsize=22, color='black', fontweight='bold')
    
    if 'Qwen3-8B' in model_data and 16 in model_data['Qwen3-8B']['k_values']:
        unfiltered_300_accuracy = model_data['Qwen3-8B']['scores'][7] 
        plt.axhline(y=unfiltered_300_accuracy, color='black', linestyle='--', 
                   alpha=0.3, linewidth=5, xmin=0.56, xmax=0.95)  # xmax=0.33 corresponds to checkpoint 100 if max is 300
        
        # unfiltered_300_accuracy = model_data['Qwen3-8B']['scores'][7] 
        # plt.axhline(y=unfiltered_300_accuracy, color='black', linestyle='--', 
        #            alpha=0.7, linewidth=3, xmin=0.4, xmax=0.7)  # xmax=0.33 corresponds to checkpoint 100 if max is 300
        
        
        
        # plt.text(1.7, unfiltered_300_accuracy + 0.01, f'3x less data', 
        #         fontsize=22, color='black', fontweight='bold')
    
    # if dataset_name:
    #     plt.title(f'{dataset_name} - Pass@k Performance', fontsize=28, fontweight='bold', pad=30)
    # else:
    #     plt.title(f'Pass@k Curves - {judge} Judge', fontsize=28, fontweight='bold', pad=30)
    # plt.title(f'Performance on TheGuardian (with retrieval)', fontsize=34, fontweight='bold', pad=20)
    
    plt.legend(fontsize=28, frameon=True, fancybox=True, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis to log scale with powers of 2
    if model_data:
        all_k_values = set()
        for data in model_data.values():
            all_k_values.update(data['k_values'])
        all_k_values = sorted(all_k_values)
        
        plt.xscale('log', base=2)
        plt.xticks(all_k_values, [str(k) for k in all_k_values], fontsize=28)
    
    # plt.ylim(0, 1.0)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.gca().tick_params(axis='both', which='major', labelsize=28)
    
    # Add some padding around the plot
    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
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
    if args.source_dir:
        print(f"Validity filtering enabled: will exclude questions with final_question_valid=0")
    
    # Get data for all models
    print(f"\nLoading data for all models in {args.input_dir}")
    if args.source_dir:
        print(f"Using source directory for validity filtering: {args.source_dir}")
    model_data = get_model_data(
        args.input_dir,
        args.judge,
        source_dir=args.source_dir,
        max_k=args.max_k,
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
    validity_suffix = "_filtered" if args.source_dir else ""
    output_filename = f"passk_{dataset_suffix}_{args.judge}{majority_suffix}{continuous_suffix}{validity_suffix}.png"
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
