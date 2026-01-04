#!/usr/bin/env python3
"""
Answer Type Analyzer - Analyze and visualize answer type distributions in questions.

This module analyzes forecasting questions to understand the distribution of
answer types (String/Name, Numeric/Integer, String/Date, etc.). It provides:

1. **Answer Type Extraction**: Extract answer_type tags from XML-formatted questions
2. **Distribution Analysis**: Count frequency of each answer type
3. **Visualization**: Create pie charts showing answer type distributions
4. **Statistical Summary**: Print detailed statistics about answer types

The analyzer helps understand what kinds of questions are being generated
and identify potential biases or gaps in question diversity.

Key Features:
- Extract answer types from XML tags in generated questions
- Filter out noise and irrelevant entries
- Generate high-quality visualizations (pie charts)
- Provide detailed statistics and summaries
- Configurable threshold for displaying rare types

Example Usage:
    ```python
    from qgen.analysis.answer_type_analyzer import analyze_answer_types, create_pie_chart
    
    # Analyze answer types from a JSONL file
    answer_types = analyze_answer_types("questions.jsonl")
    
    # Create visualization
    create_pie_chart(answer_types)
    ```

Command Line Usage:
    ```bash
    python -m qgen.analysis.answer_type_analyzer --input_file questions.jsonl
    ```

Author: Forecasting Team
"""

import json
import re
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import argparse
import sys

def extract_answer_types(generated_questions):
    """Extract all answer_type values from the generated_questions field."""
    answer_types = []
    
    # Defensive: If generated_questions is not a string, skip
    if not isinstance(generated_questions, str):
        return answer_types
    
    # Find all answer_type tags using regex
    pattern = r'<answer_type>(.*?)</answer_type>'
    matches = re.findall(pattern, generated_questions, re.DOTALL)
    
    for match in matches:
        # Clean up the answer type (remove extra whitespace)
        answer_type = match.strip()
        if answer_type and answer_type != "No leakage found":
            answer_types.append(answer_type)
    
    return answer_types

def analyze_answer_types(input_file):
    """Read the combined JSONL file and analyze answer type distributions."""
    answer_types = []
    
    # Read the combined file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                generated_questions = data.get('final_question', '')
                
                # Defensive: If generated_questions is not a string, skip this line
                if not isinstance(generated_questions, str):
                    print(f"Warning: 'generated_questions' is not a string on line {line_num}. Skipping.")
                    continue

                # Extract answer types from this line
                line_answer_types = extract_answer_types(generated_questions)
                # line_answer_types = [data.get('answer_type', '').lower()]
                answer_types.extend(line_answer_types)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return answer_types

def get_majority_types(type_counts, threshold=1):
    """
    Return a dict of answer types that occur more than `threshold` times.
    All others are ignored (not grouped as 'Others', just omitted).
    """
    majority_types = {}
    for answer_type, count in type_counts.items():
        if count > threshold:
            majority_types[answer_type] = count
    return majority_types

def create_pie_chart(answer_types):
    """Create and display a pie chart of answer type distributions (show only majority types, omit 'Others')."""
    # Count the frequency of each answer type
    type_counts = Counter(answer_types)
    
    # Only keep answer types that occur more than threshold times
    threshold = 1
    majority_counts = get_majority_types(type_counts, threshold=threshold)
    
    if not majority_counts:
        print("No answer types occur more than the threshold. Nothing to plot.")
        return
    
    # Prepare data for the pie chart
    labels = list(majority_counts.keys())
    sizes = list(majority_counts.values())
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a color palette
    colors = sns.color_palette("husl", len(labels))
    
    # Create the figure with larger size
    fig, ax = plt.subplots(figsize=(16,12))
    
    # Create the pie chart (no 'Others', so no explode for 'Others')
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', 
        colors=colors, startangle=90,
        explode=[0 for _ in labels],
        shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    # Improve text styling
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Style the labels
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    
    # Add a title
    plt.title('Distribution of Answer Types in Generated Questions', 
              fontsize=20, fontweight='bold', pad=30, color='darkblue')
    
    # Create a legend (no 'Others')
    legend_labels = [f'{label} ({count})' for label, count in majority_counts.items()]
    
    plt.legend(wedges, legend_labels,
               title="Answer Types (Count)",
               title_fontsize=18,
               fontsize=12,
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1),
               frameon=True,
               fancybox=True,
               shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart with high quality
    plt.savefig('answer_types_distribution.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"{'ANSWER TYPE ANALYSIS SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"Total questions analyzed: {sum(type_counts.values())}")
    print(f"Number of unique answer types: {len(type_counts)}")
    print(f"Number of categories shown: {len(majority_counts)}")
    print(f"\nDetailed breakdown (majority types only):")
    print(f"{'-'*60}")
    
    for answer_type, count in Counter(majority_counts).most_common():
        percentage = (count / sum(type_counts.values())) * 100
        print(f"{answer_type:<30}: {count:>3} ({percentage:>5.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze answer types from a JSONL file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file")
    args = parser.parse_args()

    print(f"🔍 Analyzing answer types from file: {args.input_file}")
    
    # Extract all answer types
    answer_types = analyze_answer_types(args.input_file)
    
    if not answer_types:
        print("❌ No answer types found in the file.")
        return
    
    print(f"✅ Found {len(answer_types)} total answer types")
    
    # Create and display the pie chart
    create_pie_chart(answer_types)
    
    print(f"\n📊 Chart saved as 'answer_types_distribution.png'")

if __name__ == "__main__":
    main() 