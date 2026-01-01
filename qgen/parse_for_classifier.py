import json
import re
from typing import Dict, List, Any, Optional
import os
import argparse
import random


def parse_generated_questions(generated_text: str, article_idx: int = -1) -> Dict[str, Any]:
    """
    Parse generated questions, reflection, and ranking from the model output.
    """
    parsed_data = {
        "gen_qs": [],
        "reflection": "",
        "ranking": []
    }
    
    # Check if generated_text is None or empty
    if not generated_text:
        print(f"Warning: Empty or None generated_text for article {article_idx}")
        return parsed_data
    
    # Extract JSON questions
    json_blocks = generated_text.split("```json")
    for block in json_blocks[1:]:  # Skip the first element which is empty
        if "```" in block:
            json_content = block.split("```")[0].strip()
            try:
                question_dict = json.loads(json_content)
                parsed_data["gen_qs"].append(question_dict)
            except json.JSONDecodeError:
                print(f"Error parsing JSON for article {article_idx}: {json_content[:100]}...")
    
    # Extract reflection (thinking)
    thinking_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
    if thinking_match:
        parsed_data["reflection"] = thinking_match.group(1).strip()
    
    # Extract ranking
    python_blocks = generated_text.split("```python")
    if len(python_blocks) > 1:
        ranking_block = python_blocks[1].split("```")[0].strip()
        try:
            # Safely evaluate the Python list expression
            ranking = eval(ranking_block)
            if isinstance(ranking, list):
                parsed_data["ranking"] = ranking
        except:
            print(f"Error parsing ranking for article {article_idx}: {ranking_block}")
    
    return parsed_data


def process_article(article_data: Dict[str, Any], article_idx: int = -1) -> Dict[str, Any]:
    """
    Process article data to extract and organize generated questions.
    """
    # Copy article metadata
    qgen_data = {
        "article_title": article_data.get("article_title", ""),
        "article_description": article_data.get("article_description", ""),
        "article_maintext": article_data.get("article_maintext", ""),
        "article_url": article_data.get("article_url", ""),
        "article_date_publish": article_data.get("article_date_publish", ""),
        "article_date_modify": article_data.get("article_date_modify", ""),
        "article_date_download": article_data.get("article_date_download", "")
    }
    
    # Parse generated questions
    parsed = parse_generated_questions(article_data.get("generated_questions", ""), article_idx)
    
    # Add to qgen_data
    qgen_data.update(parsed)
    
    return qgen_data


def extract_best_question(qgen_data: Dict[str, Any], article_idx: int = -1) -> Optional[Dict[str, Any]]:
    """
    Extract the best question based on ranking.
    """
    if not qgen_data.get("ranking") or not qgen_data.get("gen_qs"):
        print(f"Article {article_idx}: Missing ranking or questions")
        return None
    
    # Get the first ranked question ID
    try:
        best_question_id = str(qgen_data["ranking"][0])
        
        # Find the question with matching ID
        for question in qgen_data["gen_qs"]:
            # Check if question is a dictionary before trying to access its attributes
            if not isinstance(question, dict):
                print(f"Article {article_idx}: Question is not a dictionary: {type(question)}")
                continue
                
            if question.get("question_id") == best_question_id:
                return question
        
        print(f"Article {article_idx}: Could not find question with ID {best_question_id}")
                
    except (IndexError, KeyError) as e:
        print(f"Article {article_idx}: Error finding best question - {str(e)}")
    
    return None


def format_mcq_question(article_data: Dict[str, Any], question: Dict[str, Any], article_idx: int = -1, no_question_title: bool = False) -> Dict[str, Any]:
    """
    Format a question in the style of mcq_train.json.
    
    Args:
        article_data: Article data dictionary
        question: Question data dictionary
        article_idx: Index of the article
        no_question_title: If True, removes the question title from prompts
    """
    try:
        options = question.get("options", [])
        if not options:
            print(f"Article {article_idx}: Question has no options")
            options = ["Option A", "Option B", "Option C", "Option D"]
            
        answer_idx = question.get("answer", 0)
        if answer_idx >= len(options):
            print(f"Article {article_idx}: Invalid answer index {answer_idx} for options length {len(options)}")
            answer_idx = 0
        
        # Create a list of (option, is_answer) pairs
        option_pairs = [(opt, i == answer_idx) for i, opt in enumerate(options)]
        
        # Shuffle the options
        random.shuffle(option_pairs)
        
        # Unpack the shuffled options and find the new answer index
        shuffled_options = []
        new_answer_idx = 0
        for i, (opt, is_answer) in enumerate(option_pairs):
            shuffled_options.append(opt)
            if is_answer:
                new_answer_idx = i
        
        # Create options string for prompt
        options_str = ""
        for i, option in enumerate(shuffled_options):
            letter = chr(65 + i)  # A, B, C, D
            options_str += f"{letter}. {option}.\n"
        
        # Create the question title part (empty if no_question_title is True)
        question_title_part = "" if no_question_title else f"Question: {question.get('question_title', '')}\n"
        
        prompt = f"\n{question_title_part}{options_str}\n"
        
        # Full prompt with instructions
        full_prompt = (
            "You will be asked a forecasting question in multiple choice format. You have to choose the most likely option "
            "from the given options and also report your confidence level in your answer.\n\n"
            "Think thoroughly about each of the options and finally format your answer in the following format:\n\n"
            "<answer1>\nProvide exactly one option number from the choices above (e.g., A, B, C, etc.)\n</answer1>\n"
            "<answer2>\nProvide your confidence level in this answer as a decimal between 0 and 1 (e.g., 0.7 for 70% confidence)\n</answer2>\n\n"
            "IMPORTANT:\n"
            "- Your <answer1> MUST be exactly one of the option numbers listed above.\n"
            "- Your <answer2> MUST be a decimal between 0 and 1 representing your confidence.\n"
            "- Format your response exactly as shown with the <answer1> and <answer2> tags.\n\n"
            f"{question_title_part}{options_str}\n"
        )
        
        # Convert answer index to letter (A, B, C, D)
        answer_letter = chr(65 + new_answer_idx)
        
        # Safely get date strings and slice them only if they exist
        article_date_publish = article_data.get("article_date_publish", "")
        date_resolve_at = article_date_publish[:10] if article_date_publish else ""
        date_close = article_date_publish[:10] if article_date_publish else ""
        
        return {
            "date_resolve_at": date_resolve_at,  # Use only YYYY-MM-DD
            "extracted_urls": [],
            "question_type": "binary",
            "url": article_data.get("article_url", ""),
            "background": question.get("background", "Not available"),
            "resolution_criteria": "Not available",
            "is_resolved": True,
            "date_close": date_close,
            "question": question.get("question_title", ""),
            "data_source": "qgen",
            "resolution": answer_letter,
            "idx": 0,  # This will be updated in the main function
            "options": shuffled_options,
            "answer_idx": new_answer_idx,
            "answer": shuffled_options[new_answer_idx] if shuffled_options and 0 <= new_answer_idx < len(shuffled_options) else "",
            "prompt": prompt,
            "full_prompt": full_prompt
        }
    except Exception as e:
        print(f"Article {article_idx}: Error formatting MCQ question - {str(e)}")
        # Return a minimal valid structure to avoid downstream errors
        return {
            "date_resolve_at": "",
            "extracted_urls": [],
            "question_type": "binary",
            "url": "",
            "background": "Error processing question",
            "resolution_criteria": "Not available",
            "is_resolved": True,
            "date_close": "",
            "question": "Error processing question",
            "data_source": "qgen",
            "resolution": "A",
            "idx": 0,
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer_idx": 0,
            "answer": "Option A",
            "prompt": "",
            "full_prompt": ""
        }


def main(input_path: str, output_path: str, use_all_questions: bool = False, no_question_title: bool = False):
    """
    Process the input JSON or JSONL file and output the best questions in mcq_train.json format.
    
    Args:
        input_path: Path to the input JSON or JSONL file
        output_path: Path to save the output JSON file
        use_all_questions: If True, use all generated questions instead of just the best one
        no_question_title: If True, creates questions without the question title (options only)
    """
    # Read input file - use different loading methods based on file extension
    articles = []
    
    if input_path.endswith('.jsonl'):
        # Read JSONL file (line by line JSON objects)
        with open(input_path, 'r') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    articles.append(article)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON line in JSONL file: {line[:100]}...")
    else:
        # Read JSON file (single object or array)
        with open(input_path, 'r') as f:
            try:
                # Try to load as a single JSON object first
                articles = json.load(f)
                if not isinstance(articles, list):
                    articles = [articles]  # Convert to list if it's a single object
            except json.JSONDecodeError:
                print(f"Error parsing input file: {input_path}")
                return
    
    # Process each article
    qgen_data_list = []
    for idx, article in enumerate(articles):
        qgen_data = process_article(article, idx)
        qgen_data_list.append(qgen_data)
    
    # Track answer distributions
    original_answer_dist = {0: 0, 1: 0, 2: 0, 3: 0}  # Maps index to count
    shuffled_answer_dist = {'A': 0, 'B': 0, 'C': 0, 'D': 0}  # Maps letter to count
    
    # Track missing dates
    missing_date_count = 0
    
    # Extract questions
    formatted_questions = []
    for idx, data in enumerate(qgen_data_list):
        if use_all_questions:
            # Use all generated questions
            questions = data.get("gen_qs", [])
        else:
            # Use only the best question
            best_q = extract_best_question(data, idx)
            questions = [best_q] if best_q else []
        
        for q_idx, question in enumerate(questions):
            # Skip if question is None or not a dictionary
            if not question or not isinstance(question, dict):
                print(f"Article {idx}: Invalid question at index {q_idx}")
                continue
                
            # Track original answer distribution
            original_idx = question.get("answer", 0)
            if original_idx in original_answer_dist:
                original_answer_dist[original_idx] += 1
            
            # Check if date is missing
            if not data.get("article_date_publish"):
                missing_date_count += 1
            
            mcq_format = format_mcq_question(data, question, idx, no_question_title)
            mcq_format["idx"] = 10000 + idx * 100 + q_idx  # Start from 10000 to avoid conflicts
            
            # Track shuffled answer distribution
            answer_letter = mcq_format["resolution"]
            if answer_letter in shuffled_answer_dist:
                shuffled_answer_dist[answer_letter] += 1
                
            formatted_questions.append(mcq_format)
    
    # Print answer distributions
    print("\nOriginal answer distribution (by index):")
    for idx, count in original_answer_dist.items():
        print(f"  Index {idx}: {count} questions ({count/max(1, len(formatted_questions))*100:.1f}%)")
    
    print("\nShuffled answer distribution (by letter):")
    for letter, count in shuffled_answer_dist.items():
        print(f"  Option {letter}: {count} questions ({count/max(1, len(formatted_questions))*100:.1f}%)")
    
    # Print missing date information
    print(f"\nQuestions with missing date information: {missing_date_count} ({missing_date_count/max(1, len(formatted_questions))*100:.1f}%)")
    
    # Print information about the question title status
    print(f"\nQuestion titles included: {'No' if no_question_title else 'Yes'}")
    
    # Save output file
    with open(output_path, 'w') as f:
        json.dump(formatted_questions, f, indent=4)
    
    print(f"\nProcessed {len(articles)} articles")
    print(f"Extracted {len(formatted_questions)} questions")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process generated questions and convert to MCQ format')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to the input JSON or JSONL file containing generated questions')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Path to save the output JSON file with MCQ formatted questions')
    parser.add_argument('--all_questions', '-a', action='store_true',
                        help='Use all generated questions instead of just the best one')
    parser.add_argument('--no_question_title', '-n', action='store_true',
                        help='Create questions without the question title (options only)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get basename of input path
    input_basename = os.path.basename(args.input)
    
    # Modify output filename if no_question_title is True
    if args.no_question_title:
        base_name, ext = os.path.splitext(input_basename)
        input_basename = f"{base_name}_options_only{ext}"
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args.input, os.path.join(args.output_dir, input_basename), args.all_questions, args.no_question_title)