#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def is_correct(record: Dict[str, Any]) -> bool:
    """Check if any generation was correct according to score_Llama_4_Scout."""
    scores = record.get('score_Llama_4_Scout', [])
    if not scores:
        return False
    # Check if any generation got a score of 1.0
    for score_dict in scores:
        if isinstance(score_dict, dict):
            if any(v == 1.0 for v in score_dict.values()):
                return True
    return False

def analyze_failures(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze failure modes from the records."""
    failures = []
    correct = []
    
    for record in records:
        if is_correct(record):
            correct.append(record)
        else:
            failures.append(record)
    
    # Analyze failure categories
    category_counter = Counter()
    answer_type_counter = Counter()
    
    for failure in failures:
        answer_type = failure.get('answer_type', 'unknown')
        answer_type_counter[answer_type] += 1
        
        # Try to infer category from question or answer type
        # Extract the category from parentheses, e.g., "string (name)" -> "name"
        if '(' in answer_type and ')' in answer_type:
            category = answer_type.split('(')[1].split(')')[0].strip().lower()
        else:
            category = answer_type.split('(')[0].strip().lower() if '(' in answer_type else answer_type.lower()
        # Capitalize first letter for display
        category = category.capitalize()
        category_counter[category] += 1
    
    # Analyze response patterns for failures
    response_patterns = defaultdict(list)
    
    for failure in failures:
        responses = failure.get('response', [])
        extracted_answers = failure.get('extracted_answer', [])
        question_title = failure.get('question_title', 'Unknown')
        correct_answer = failure.get('answer', 'Unknown')
        
        # Get the most common extracted answer
        all_extracted = []
        for ext_dict in extracted_answers:
            if isinstance(ext_dict, dict):
                all_extracted.extend(ext_dict.keys())
        
        most_common_answer = Counter(all_extracted).most_common(1)
        model_answer = most_common_answer[0][0] if most_common_answer else "No answer"
        
        # Categorize the type of error
        error_type = "unknown"
        if model_answer.lower() == "unknown" or not model_answer:
            error_type = "no_answer"
        elif model_answer.lower() in str(correct_answer).lower() or str(correct_answer).lower() in model_answer.lower():
            error_type = "partial_match"
        else:
            error_type = "wrong_answer"
        
        response_patterns[error_type].append({
            'question': question_title,
            'model_answer': model_answer,
            'correct_answer': correct_answer,
            'answer_type': answer_type,
            'responses': responses[:1] if responses else []  # Store first response
        })
    
    return {
        'total': len(records),
        'failures': len(failures),
        'correct': len(correct),
        'failure_rate': len(failures) / len(records) if records else 0,
        'category_distribution': dict(category_counter),
        'answer_type_distribution': dict(answer_type_counter),
        'error_patterns': dict(response_patterns),
        'sample_failures': failures[:20]  # Top 20 for examples
    }

def generate_latex(analysis: Dict[str, Any]) -> str:
    """Generate LaTeX subsection from analysis."""
    
    latex = """\\subsection{Failure Mode Analysis}

"""
    
    # Overall statistics
    latex += f"""The model achieved a correctness rate of {analysis['correct']}/{analysis['total']} ({100*analysis['failure_rate']:.1f}\\% failure rate) according to the Llama 4 Scout scoring metric. 
This section analyzes the {analysis['failures']} questions where the model provided incorrect answers across all three generations.

"""
    
    # Category distribution
    latex += """\\subsubsection{Failure Distribution by Question Type}

"""
    
    if analysis['answer_type_distribution']:
        latex += "Table~\\ref{tab:failure_types} shows the distribution of failures by answer type.\n\n"
        latex += """\\begin{table}[h]
\\centering
\\begin{tabular}{|l|c|}
\\hline
\\textbf{Answer Type} & \\textbf{Count} \\\\
\\hline
"""
        
        sorted_types = sorted(analysis['answer_type_distribution'].items(), key=lambda x: x[1], reverse=True)
        for answer_type, count in sorted_types:
            latex += f"{answer_type.replace('_', '\\_')} & {count} \\\\\n"
        
        latex += """\\hline
\\end{tabular}
\\caption{Distribution of failures by answer type}
\\label{tab:failure_types}
\\end{table}

"""
    
    # Error patterns
    latex += """\\subsubsection{Error Patterns}

"""
    
    error_patterns = analysis['error_patterns']
    total_patterns = sum(len(v) for v in error_patterns.values())
    
    for error_type, examples in error_patterns.items():
        count = len(examples)
        percentage = (count / total_patterns * 100) if total_patterns > 0 else 0
        
        latex += f"\\textbf{{{error_type.replace('_', ' ').title()}}}: {count} failures ({percentage:.1f}\\%)\n\n"
        
        # Show a few examples
        if examples:
            latex += "Examples:\n\\begin{itemize}\n"
            for i, example in enumerate(examples[:5]):  # Show up to 5 examples
                question = example['question'].replace('_', '\\_')
                model_ans = str(example['model_answer']).replace('_', '\\_')
                correct_ans = str(example['correct_answer']).replace('_', '\\_')
                
                latex += f"\\item \\textbf{{Question}}: {question}\n"
                latex += "  \\begin{itemize}\n"
                latex += f"    \\item Model answer: {model_ans}\n"
                latex += f"    \\item Correct answer: {correct_ans}\n"
                latex += f"    \\item Type: {example['answer_type'].replace('_', '\\_')}\n"
                latex += "  \\end{itemize}\n"
            
            latex += "\\end{itemize}\n\n"
    
    # Most common failure categories
    latex += """\\subsubsection{Most Common Failure Categories}

"""
    
    if analysis['category_distribution']:
        sorted_cats = sorted(analysis['category_distribution'].items(), key=lambda x: x[1], reverse=True)
        latex += "The most common categories of failures were:\n\\begin{enumerate}\n"
        
        for i, (category, count) in enumerate(sorted_cats[:10], 1):  # Top 10
            percentage = (count / analysis['failures'] * 100) if analysis['failures'] > 0 else 0
            latex += f"\\item \\textbf{{{category.replace('_', ' ').title()}}}: {count} failures ({percentage:.1f}\\%)\n"
        
        latex += "\\end{enumerate}\n\n"
    
    # Sample failure analysis
    latex += """\\subsubsection{Sample Failure Analysis}

"""
    
    sample_failures = analysis['sample_failures'][:10]  # Top 10 for detailed analysis
    latex += "The following examples illustrate common failure modes:\n\n"
    
    for i, failure in enumerate(sample_failures, 1):
        question = failure.get('question_title', 'Unknown').replace('_', '\\_')
        correct_answer = str(failure.get('answer', 'Unknown')).replace('_', '\\_')
        answer_type = failure.get('answer_type', 'unknown').replace('_', '\\_')
        
        extracted_answers = failure.get('extracted_answer', [])
        all_extracted = []
        for ext_dict in extracted_answers:
            if isinstance(ext_dict, dict):
                all_extracted.extend(ext_dict.keys())
        
        model_answer = Counter(all_extracted).most_common(1)
        model_ans_str = model_answer[0][0] if model_answer else "No answer extracted"
        model_ans_str = model_ans_str.replace('_', '\\_')
        
        responses = failure.get('response', [])
        first_response = responses[0] if responses else "No response"
        # Truncate response for LaTeX
        if len(first_response) > 300:
            first_response = first_response[:300] + "..."
        first_response = first_response.replace('_', '\\_').replace('%', '\\%')
        
        latex += f"\\textbf{{Example {i}}}: {question}\n\n"
        latex += "\\begin{itemize}\n"
        latex += f"  \\item Correct answer: {correct_answer}\n"
        latex += f"  \\item Model answer: {model_ans_str}\n"
        latex += f"  \\item Answer type: {answer_type}\n"
        latex += f"  \\item Model reasoning (excerpt): {first_response}\n"
        latex += "\\end{itemize}\n\n"
    
    return latex

def main():
    filepath = '/fast/nchandak/forecasting/evals/freeform/manual/news5-retrieval_1000/Qwen3-8B-sft-rl_eval_size_1000_generations_3_num_articles_5.jsonl'
    
    print("Loading data...")
    records = load_jsonl(filepath)
    print(f"Loaded {len(records)} records")
    
    print("Analyzing failures...")
    analysis = analyze_failures(records)
    
    print(f"Total: {analysis['total']}")
    print(f"Failures: {analysis['failures']}")
    print(f"Correct: {analysis['correct']}")
    print(f"Failure rate: {analysis['failure_rate']*100:.1f}%")
    
    print("Generating LaTeX...")
    latex_content = generate_latex(analysis)
    
    output_file = '/home/nchandak/forecasting/failure_analysis.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX written to {output_file}")

if __name__ == '__main__':
    main()

