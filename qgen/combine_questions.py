#!/usr/bin/env python3
import json
import re

def combine_files():
    # Read both files
    with open('debug/o4_dw_30_free1.jsonl', 'r') as f:
        o4_lines = [json.loads(line.strip()) for line in f if line.strip()]
    
    with open('debug/v3_dw_30_free1.jsonl', 'r') as f:
        v3_lines = [json.loads(line.strip()) for line in f if line.strip()]
    
    # Ensure both files have the same number of lines
    assert len(o4_lines) == len(v3_lines), f"Files have different lengths: {len(o4_lines)} vs {len(v3_lines)}"
    
    combined_lines = []
    
    for i, (o4_item, v3_item) in enumerate(zip(o4_lines, v3_lines)):
        # Start with o4 item as base
        combined_item = o4_item.copy()
        
        # Get the generated_questions from both
        o4_questions = o4_item['generated_questions']
        v3_questions = v3_item['generated_questions']
        
        # Convert v3 questions from q1 to q2
        v3_questions_modified = re.sub(r'<q1>', '<q2>', v3_questions)
        v3_questions_modified = re.sub(r'</q1>', '</q2>', v3_questions_modified)
        v3_questions_modified = re.sub(r'question_id>0<', 'question_id>1<', v3_questions_modified)
        
        combined_questions = o4_questions + "\n\n" + v3_questions_modified
        
        combined_item['generated_questions'] = combined_questions
        combined_lines.append(combined_item)
    
    # Write combined file
    with open('debug/combined_dw_30_free1.jsonl', 'w') as f:
        for item in combined_lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Combined {len(combined_lines)} lines into qgen/debug/combined_dw_30_free1.jsonl")

if __name__ == "__main__":
    combine_files() 