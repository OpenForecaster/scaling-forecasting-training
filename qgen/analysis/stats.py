#!/usr/bin/env python3
"""
Statistics analysis for question generation pipeline results.

This script analyzes the output from the question generation pipeline to provide
comprehensive statistics about the effectiveness of each step.
"""

import json
import argparse
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import os


def load_jsonl_file(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    return data


def extract_answer_from_question(question_text: str) -> str:
    """Extract the answer from a question XML text."""
    if not question_text:
        return ""
    
    # Look for <answer>...</answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', question_text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""


def determine_selected_question(choose_best_response: str, questions: List[str]) -> int:
    """
    Determine which question was selected by choose_best by comparing the final output
    with the original questions.
    
    Args:
        choose_best_response: The full response from choose_best
        questions: List of original questions [q1, q2, q3]
        
    Returns:
        Question number (1, 2, 3) that was selected, or 0 if none/unclear
    """
    if not choose_best_response or not questions:
        return 0
    
    # Check for "NO GOOD QUESTION" case
    if "NO GOOD QUESTION" in choose_best_response.upper():
        return 0
    
    # Extract the final question from the choose_best response
    final_q_match = re.search(r'<q1>(.*?)</q1>', choose_best_response, re.DOTALL)
    if not final_q_match:
        return 0
    
    final_question = final_q_match.group(1).strip()
    
    # Extract key identifiers from the final question
    final_title_match = re.search(r'<question_title>(.*?)</question_title>', final_question, re.DOTALL)
    final_answer_match = re.search(r'<answer>(.*?)</answer>', final_question, re.DOTALL)
    
    if not final_title_match or not final_answer_match:
        return 0
    
    final_title = final_title_match.group(1).strip()
    final_answer = final_answer_match.group(1).strip()
    
    # Compare with original questions to find the best match
    best_match_score = 0
    best_match_question = 0
    
    for i, orig_question in enumerate(questions, 1):
        if not orig_question:
            continue
        
        # Extract title and answer from original question
        orig_title_match = re.search(r'<question_title>(.*?)</question_title>', orig_question, re.DOTALL)
        orig_answer_match = re.search(r'<answer>(.*?)</answer>', orig_question, re.DOTALL)
        
        if not orig_title_match or not orig_answer_match:
            continue
        
        orig_title = orig_title_match.group(1).strip()
        orig_answer = orig_answer_match.group(1).strip()
        
        # Calculate similarity score based on title and answer matching
        score = 0
        
        # Title similarity (most important)
        if orig_title.lower() == final_title.lower():
            score += 100  # Exact match
        elif orig_title.lower() in final_title.lower() or final_title.lower() in orig_title.lower():
            score += 50   # Partial match
        
        # Answer similarity (very important)
        if orig_answer.lower() == final_answer.lower():
            score += 80   # Exact match
        elif orig_answer.lower() in final_answer.lower() or final_answer.lower() in orig_answer.lower():
            score += 40   # Partial match
        
        if score > best_match_score:
            best_match_score = score
            best_match_question = i
    
    # Return the best match if score is high enough, otherwise 0
    return best_match_question if best_match_score >= 80 else 0


def check_answer_leakage(question_text: str, answer: str) -> bool:
    """
    Check if the answer appears in the question text (excluding the answer tags).
    
    Args:
        question_text: Full question XML text
        answer: The actual answer
        
    Returns:
        True if leakage is detected, False otherwise
    """
    if not question_text or not answer:
        return False
    
    # Remove the answer tags to avoid false positives
    question_without_answer = re.sub(r'<answer>.*?</answer>', '', question_text, flags=re.DOTALL)
    
    # Check for exact match (case insensitive)
    return answer.lower() in question_without_answer.lower()


def analyze_pipeline_statistics(data: List[Dict], num_questions_per_article: int = 3) -> Dict:
    """
    Analyze the question generation pipeline statistics.
    
    Args:
        data: List of article processing results
        num_questions_per_article: Number of questions generated per article
        
    Returns:
        Dictionary containing comprehensive statistics
    """
    stats = {
        'source_articles': len(data),
        'total_questions_generated': 0,
        'questions_with_content': 0,
        'valid_questions': 0,
        'invalid_questions': 0,
        'articles_with_valid_questions': 0,
        'articles_with_no_good_question': 0,
        'articles_choose_best_completed': 0,
        'questions_with_leakage': 0,
        'questions_without_leakage': 0,
        'chosen_questions_with_leakage': 0,  # Before leakage removal
        'chosen_questions_without_leakage': 0,  # Before leakage removal
        'final_questions_with_leakage': 0,  # After leakage removal
        'final_questions_without_leakage': 0,  # After leakage removal
        'articles_by_validation_stage': defaultdict(int),
        'question_validity_by_position': defaultdict(lambda: {'valid': 0, 'invalid': 0, 'total': 0}),
        'choose_best_selection': defaultdict(int),  # Track which question was selected
        'leakage_details': [],
        'chosen_question_leakage_details': [],  # Before leakage removal
        'final_question_leakage_details': [],  # After leakage removal
    }
    
    for article in data:
        article_has_valid_question = False
        article_questions_generated = 0
        
        # Check each question for this article
        for q_num in range(1, num_questions_per_article + 1):
            question_key = f"q{q_num}"
            valid_key = f"q{q_num}_valid"
            
            question_text = article.get(question_key, "")
            
            if question_text and len(question_text.strip()) > 10:
                article_questions_generated += 1
                stats['total_questions_generated'] += 1
                stats['questions_with_content'] += 1
                stats['question_validity_by_position'][q_num]['total'] += 1
                
                # Check validation status
                is_valid = article.get(valid_key, 0) == 1
                if is_valid:
                    stats['valid_questions'] += 1
                    stats['question_validity_by_position'][q_num]['valid'] += 1
                    article_has_valid_question = True
                else:
                    stats['invalid_questions'] += 1
                    stats['question_validity_by_position'][q_num]['invalid'] += 1
                
                # Check for answer leakage
                answer = extract_answer_from_question(question_text)
                if answer:
                    has_leakage = check_answer_leakage(question_text, answer)
                    if has_leakage:
                        stats['questions_with_leakage'] += 1
                        stats['leakage_details'].append({
                            'article_url': article.get('url', 'unknown'),
                            'question_num': q_num,
                            'answer': answer,
                            'question_snippet': question_text[:200] + "..." if len(question_text) > 200 else question_text
                        })
                    else:
                        stats['questions_without_leakage'] += 1
            else:
                # Empty or very short question
                stats['question_validity_by_position'][q_num]['total'] += 1
        
        # Article-level statistics
        if article_has_valid_question:
            stats['articles_with_valid_questions'] += 1
        
        # Check no_good_question flag
        if article.get('no_good_question', 0) == 1:
            stats['articles_with_no_good_question'] += 1
        
        # Check choose_best completion and analyze selection
        if article.get('choose_best', 0) == 1:
            stats['articles_choose_best_completed'] += 1
            
            # Analyze which question was selected
            choose_best_response = article.get('choose_best_response', '')
            if choose_best_response and 'NO GOOD QUESTION' not in choose_best_response.upper():
                # Get all questions for this article
                article_questions = []
                for q_num in range(1, num_questions_per_article + 1):
                    question_text = article.get(f"q{q_num}", "")
                    article_questions.append(question_text)
                
                selected_q_num = determine_selected_question(choose_best_response, article_questions)
                if selected_q_num > 0:
                    stats['choose_best_selection'][selected_q_num] += 1
                
                # Check leakage in the chosen question (before leakage removal)
                chosen_q_match = re.search(r'<q1>(.*?)</q1>', choose_best_response, re.DOTALL)
                if chosen_q_match:
                    chosen_question_text = chosen_q_match.group(0)  # Include the q1 tags
                    chosen_answer = extract_answer_from_question(chosen_question_text)
                    if chosen_answer:
                        has_chosen_leakage = check_answer_leakage(chosen_question_text, chosen_answer)
                        if has_chosen_leakage:
                            stats['chosen_questions_with_leakage'] += 1
                            stats['chosen_question_leakage_details'].append({
                                'article_url': article.get('url', 'unknown'),
                                'selected_question_num': selected_q_num,
                                'answer': chosen_answer,
                                'question_snippet': chosen_question_text[:200] + "..." if len(chosen_question_text) > 200 else chosen_question_text
                            })
                        else:
                            stats['chosen_questions_without_leakage'] += 1
                
        # Check leakage in the final selected question (using final_question field)
        final_question_text = article.get('final_question', '')
        if final_question_text and len(final_question_text.strip()) > 10:
            final_answer = extract_answer_from_question(final_question_text)
            if final_answer:
                has_final_leakage = check_answer_leakage(final_question_text, final_answer)
                if has_final_leakage:
                    stats['final_questions_with_leakage'] += 1
                    # Determine which original question this came from for better analysis
                    selected_q_num = 0
                    if article.get('choose_best', 0) == 1:
                        choose_best_response = article.get('choose_best_response', '')
                        if choose_best_response:
                            article_questions = []
                            for q_num in range(1, num_questions_per_article + 1):
                                question_text = article.get(f"q{q_num}", "")
                                article_questions.append(question_text)
                            selected_q_num = determine_selected_question(choose_best_response, article_questions)
                    
                    stats['final_question_leakage_details'].append({
                        'article_url': article.get('url', 'unknown'),
                        'selected_question_num': selected_q_num,
                        'answer': final_answer,
                        'question_snippet': final_question_text[:200] + "..." if len(final_question_text) > 200 else final_question_text
                    })
                else:
                    stats['final_questions_without_leakage'] += 1
        
        # Track validation stage completion
        if article.get('individual_validation_done', 0) == 1:
            stats['articles_by_validation_stage']['individual_validation_completed'] += 1
        
        if article.get('choose_best', 0) == 1:
            stats['articles_by_validation_stage']['choose_best_completed'] += 1
        
        if article.get('leakage_check', 0) == 1:
            stats['articles_by_validation_stage']['leakage_check_completed'] += 1
    
    return stats


def print_comprehensive_statistics(stats: Dict, num_questions_per_article: int = 3):
    """Print comprehensive statistics in a well-formatted manner."""
    
    print("=" * 80)
    print("QUESTION GENERATION PIPELINE STATISTICS")
    print("=" * 80)
    
    # Basic statistics
    print("\n📊 BASIC STATISTICS:")
    print("-" * 40)
    print(f"Source articles processed:        {stats['source_articles']:,}")
    print(f"Total questions generated:       {stats['total_questions_generated']:,}")
    print(f"Questions with content:          {stats['questions_with_content']:,}")
    print(f"Average questions per article:   {stats['total_questions_generated'] / stats['source_articles']:.2f}")
    
    # Validation statistics
    print("\n✅ VALIDATION STATISTICS:")
    print("-" * 40)
    print(f"Valid questions:                 {stats['valid_questions']:,} ({stats['valid_questions']/stats['total_questions_generated']*100:.1f}%)")
    print(f"Invalid questions:               {stats['invalid_questions']:,} ({stats['invalid_questions']/stats['total_questions_generated']*100:.1f}%)")
    print(f"Articles with ≥1 valid question: {stats['articles_with_valid_questions']:,} ({stats['articles_with_valid_questions']/stats['source_articles']*100:.1f}%)")
    print(f"Articles with no good question:  {stats['articles_with_no_good_question']:,} ({stats['articles_with_no_good_question']/stats['source_articles']*100:.1f}%)")
    
    # Question position analysis
    print("\n📍 QUESTION POSITION ANALYSIS:")
    print("-" * 40)
    for q_num in range(1, num_questions_per_article + 1):
        q_stats = stats['question_validity_by_position'][q_num]
        if q_stats['total'] > 0:
            valid_rate = q_stats['valid'] / q_stats['total'] * 100
            print(f"Question {q_num}: {q_stats['valid']:,}/{q_stats['total']:,} valid ({valid_rate:.1f}%)")
    
    # Choose best selection analysis
    total_selections = sum(stats['choose_best_selection'].values())
    if total_selections > 0:
        print("\n🎯 CHOOSE BEST SELECTION ANALYSIS:")
        print("-" * 40)
        print(f"Total articles with question selected: {total_selections:,}")
        for q_num in range(1, num_questions_per_article + 1):
            selection_count = stats['choose_best_selection'][q_num]
            if total_selections > 0:
                selection_rate = selection_count / total_selections * 100
                print(f"Question {q_num} selected: {selection_count:,} times ({selection_rate:.1f}%)")
        
        # Calculate selection bias
        expected_rate = 100 / num_questions_per_article
        print(f"\nSelection bias analysis (expected: {expected_rate:.1f}% each):")
        for q_num in range(1, num_questions_per_article + 1):
            selection_count = stats['choose_best_selection'][q_num]
            if total_selections > 0:
                actual_rate = selection_count / total_selections * 100
                bias = actual_rate - expected_rate
                bias_indicator = "📈" if bias > 5 else "📉" if bias < -5 else "➡️"
                print(f"  Question {q_num}: {actual_rate:.1f}% ({bias:+.1f}% vs expected) {bias_indicator}")
    else:
        print("\n🎯 CHOOSE BEST SELECTION ANALYSIS:")
        print("-" * 40)
        print("No successful question selections found to analyze.")
    
    # Pipeline stage completion
    print("\n🔄 PIPELINE STAGE COMPLETION:")
    print("-" * 40)
    stage_stats = stats['articles_by_validation_stage']
    print(f"Individual validation completed: {stage_stats['individual_validation_completed']:,} ({stage_stats['individual_validation_completed']/stats['source_articles']*100:.1f}%)")
    print(f"Choose best completed:           {stage_stats['choose_best_completed']:,} ({stage_stats['choose_best_completed']/stats['source_articles']*100:.1f}%)")
    print(f"Leakage check completed:         {stage_stats['leakage_check_completed']:,} ({stage_stats['leakage_check_completed']/stats['source_articles']*100:.1f}%)")
    
    # Answer leakage analysis for all generated questions
    print("\n🔍 ANSWER LEAKAGE ANALYSIS - ALL GENERATED QUESTIONS:")
    print("-" * 60)
    total_questions_with_answers = stats['questions_with_leakage'] + stats['questions_without_leakage']
    if total_questions_with_answers > 0:
        leakage_rate = stats['questions_with_leakage'] / total_questions_with_answers * 100
        print(f"Questions with answer leakage:   {stats['questions_with_leakage']:,} ({leakage_rate:.1f}%)")
        print(f"Questions without leakage:       {stats['questions_without_leakage']:,} ({100-leakage_rate:.1f}%)")
        print(f"Total questions analyzed:        {total_questions_with_answers:,}")
    else:
        print("No questions with extractable answers found for leakage analysis.")
    
    # Answer leakage analysis for chosen questions (before leakage removal)
    print("\n🎯 ANSWER LEAKAGE ANALYSIS - CHOSEN QUESTIONS (BEFORE EDITING):")
    print("-" * 60)
    total_chosen_questions = stats['chosen_questions_with_leakage'] + stats['chosen_questions_without_leakage']
    if total_chosen_questions > 0:
        chosen_leakage_rate = stats['chosen_questions_with_leakage'] / total_chosen_questions * 100
        print(f"Chosen questions with leakage:   {stats['chosen_questions_with_leakage']:,} ({chosen_leakage_rate:.1f}%)")
        print(f"Chosen questions without leakage:{stats['chosen_questions_without_leakage']:,} ({100-chosen_leakage_rate:.1f}%)")
        print(f"Total chosen questions analyzed: {total_chosen_questions:,}")
    else:
        print("No chosen questions with extractable answers found for leakage analysis.")
    
    # Answer leakage analysis for final questions (after leakage removal)
    print("\n✨ ANSWER LEAKAGE ANALYSIS - FINAL QUESTIONS (AFTER EDITING):")
    print("-" * 60)
    total_final_questions = stats['final_questions_with_leakage'] + stats['final_questions_without_leakage']
    if total_final_questions > 0:
        final_leakage_rate = stats['final_questions_with_leakage'] / total_final_questions * 100
        print(f"Final questions with leakage:    {stats['final_questions_with_leakage']:,} ({final_leakage_rate:.1f}%)")
        print(f"Final questions without leakage: {stats['final_questions_without_leakage']:,} ({100-final_leakage_rate:.1f}%)")
        print(f"Total final questions analyzed:  {total_final_questions:,}")
        
        # Compare leakage rates across stages
        if total_chosen_questions > 0:
            leakage_removal_improvement = chosen_leakage_rate - final_leakage_rate
            if leakage_removal_improvement > 0:
                print(f"Leakage removal effectiveness: {leakage_removal_improvement:.1f}% reduction")
            elif leakage_removal_improvement < 0:
                print(f"Leakage removal degradation: {abs(leakage_removal_improvement):.1f}% increase")
            else:
                print("No change in leakage rate after editing")
        
        # Compare with overall generation leakage
        if total_questions_with_answers > 0:
            overall_improvement = leakage_rate - final_leakage_rate
            print(f"Overall pipeline leakage improvement: {overall_improvement:.1f}% reduction from generation to final")
    else:
        print("No final questions with extractable answers found for leakage analysis.")
    
    # Show some leakage examples for all three types
    if stats['leakage_details']:
        print(f"\n🚨 GENERATED QUESTION LEAKAGE EXAMPLES (showing first 3):")
        print("-" * 60)
        for i, example in enumerate(stats['leakage_details'][:3]):
            print(f"Example {i+1}:")
            print(f"  Question {example['question_num']}: Answer '{example['answer']}' found in question text")
            print(f"  URL: {example['article_url']}")
            print()
    
    if stats['chosen_question_leakage_details']:
        print(f"\n🎯 CHOSEN QUESTION LEAKAGE EXAMPLES (showing first 3):")
        print("-" * 60)
        for i, example in enumerate(stats['chosen_question_leakage_details'][:3]):
            print(f"Example {i+1}:")
            print(f"  Selected Question {example['selected_question_num']}: Answer '{example['answer']}' found in chosen question")
            print(f"  URL: {example['article_url']}")
            print()
    
    if stats['final_question_leakage_details']:
        print(f"\n✨ FINAL QUESTION LEAKAGE EXAMPLES (showing first 3):")
        print("-" * 60)
        for i, example in enumerate(stats['final_question_leakage_details'][:3]):
            print(f"Example {i+1}:")
            print(f"  Final Question: Answer '{example['answer']}' found in final question")
            print(f"  URL: {example['article_url']}")
            print()


def create_pipeline_effectiveness_table(stats: Dict):
    """Create a table showing the effectiveness of each pipeline step."""
    
    print("\n📈 PIPELINE EFFECTIVENESS TABLE:")
    print("=" * 80)
    
    # Calculate metrics for each stage
    source_articles = stats['source_articles']
    total_questions = stats['total_questions_generated']
    valid_questions = stats['valid_questions']
    articles_with_valid = stats['articles_with_valid_questions']
    choose_best_completed = stats['articles_by_validation_stage']['choose_best_completed']
    no_good_questions = stats['articles_with_no_good_question']
    leakage_free_questions = stats['questions_without_leakage']
    
    # Table header
    print(f"{'Step':<35} {'Success Rate':<15} {'Output':<20} {'Quality Metric'}")
    print("-" * 80)
    
    # Step 1: Question Generation
    gen_success_rate = (total_questions / (source_articles * 3)) * 100 if source_articles > 0 else 0
    print(f"{'1. Question Generation':<35} {gen_success_rate:.1f}%{'':<10} {total_questions:,} questions{'':<5} {gen_success_rate:.1f}% completion")
    
    # Step 2: Question Validation
    val_success_rate = (valid_questions / total_questions) * 100 if total_questions > 0 else 0
    print(f"{'2. Individual Validation':<35} {val_success_rate:.1f}%{'':<10} {valid_questions:,} valid{'':<8} {val_success_rate:.1f}% pass rate")
    
    # Step 3: Choose Best
    choose_success_rate = (articles_with_valid / source_articles) * 100 if source_articles > 0 else 0
    usable_articles = articles_with_valid - no_good_questions
    usability_rate = (usable_articles / source_articles) * 100 if source_articles > 0 else 0
    print(f"{'3. Choose Best':<35} {choose_success_rate:.1f}%{'':<10} {usable_articles:,} usable{'':<8} {usability_rate:.1f}% usability")
    
    # Step 4a: Leakage Check (All Questions)
    if leakage_free_questions > 0:
        leakage_success_rate = (leakage_free_questions / (stats['questions_with_leakage'] + stats['questions_without_leakage'])) * 100
        print(f"{'4a. Leakage Check (All Questions)':<35} {leakage_success_rate:.1f}%{'':<10} {leakage_free_questions:,} clean{'':<9} {leakage_success_rate:.1f}% leakage-free")
    
    # Step 4b: Chosen Questions (Before Editing)
    chosen_leakage_free = stats['chosen_questions_without_leakage']
    total_chosen_questions = stats['chosen_questions_with_leakage'] + stats['chosen_questions_without_leakage']
    if total_chosen_questions > 0:
        chosen_leakage_success_rate = (chosen_leakage_free / total_chosen_questions) * 100
        print(f"{'4b. Chosen Questions (Before Edit)':<35} {chosen_leakage_success_rate:.1f}%{'':<10} {chosen_leakage_free:,} clean{'':<9} {chosen_leakage_success_rate:.1f}% leakage-free")
    
    # Step 4c: Final Questions (After Editing)
    final_leakage_free = stats['final_questions_without_leakage']
    total_final_questions = stats['final_questions_with_leakage'] + stats['final_questions_without_leakage']
    if total_final_questions > 0:
        final_leakage_success_rate = (final_leakage_free / total_final_questions) * 100
        print(f"{'4c. Final Questions (After Edit)':<35} {final_leakage_success_rate:.1f}%{'':<10} {final_leakage_free:,} clean{'':<9} {final_leakage_success_rate:.1f}% leakage-free")
    
    print("-" * 80)
    
    # Overall pipeline efficiency
    final_usable_rate = (usable_articles / source_articles) * 100 if source_articles > 0 else 0
    print(f"{'OVERALL PIPELINE EFFICIENCY':<35} {final_usable_rate:.1f}%{'':<10} {usable_articles:,}/{source_articles:,}{'':<6} End-to-end success")


def analyze_failure_modes(data: List[Dict], num_questions_per_article: int = 3):
    """Analyze common failure modes in the pipeline."""
    
    print("\n🔍 FAILURE MODE ANALYSIS:")
    print("=" * 80)
    
    failure_modes = {
        'no_questions_generated': 0,
        'all_questions_invalid': 0,
        'partial_question_failure': 0,
        'choose_best_failed': 0,
        'high_leakage': 0
    }
    
    articles_by_valid_count = defaultdict(int)
    
    for article in data:
        valid_count = 0
        total_generated = 0
        leakage_count = 0
        
        # Count valid questions and leakage for this article
        for q_num in range(1, num_questions_per_article + 1):
            question_text = article.get(f"q{q_num}", "")
            if question_text and len(question_text.strip()) > 10:
                total_generated += 1
                if article.get(f"q{q_num}_valid", 0) == 1:
                    valid_count += 1
                
                # Check leakage
                answer = extract_answer_from_question(question_text)
                if answer and check_answer_leakage(question_text, answer):
                    leakage_count += 1
        
        articles_by_valid_count[valid_count] += 1
        
        # Classify failure modes
        if total_generated == 0:
            failure_modes['no_questions_generated'] += 1
        elif valid_count == 0 and total_generated > 0:
            failure_modes['all_questions_invalid'] += 1
        elif valid_count < total_generated and valid_count > 0:
            failure_modes['partial_question_failure'] += 1
        
        if article.get('choose_best', 0) != 1 and valid_count > 0:
            failure_modes['choose_best_failed'] += 1
        
        if leakage_count >= total_generated * 0.5:  # >50% questions have leakage
            failure_modes['high_leakage'] += 1
    
    # Print failure analysis
    total_articles = len(data)
    print(f"Articles with no questions generated:     {failure_modes['no_questions_generated']:,} ({failure_modes['no_questions_generated']/total_articles*100:.1f}%)")
    print(f"Articles with all questions invalid:      {failure_modes['all_questions_invalid']:,} ({failure_modes['all_questions_invalid']/total_articles*100:.1f}%)")
    print(f"Articles with partial question failure:   {failure_modes['partial_question_failure']:,} ({failure_modes['partial_question_failure']/total_articles*100:.1f}%)")
    print(f"Articles where choose_best failed:        {failure_modes['choose_best_failed']:,} ({failure_modes['choose_best_failed']/total_articles*100:.1f}%)")
    print(f"Articles with high answer leakage:        {failure_modes['high_leakage']:,} ({failure_modes['high_leakage']/total_articles*100:.1f}%)")
    
    # Distribution of valid questions per article
    print(f"\n📊 VALID QUESTIONS DISTRIBUTION:")
    print("-" * 40)
    for valid_count in sorted(articles_by_valid_count.keys()):
        count = articles_by_valid_count[valid_count]
        percentage = count / total_articles * 100
        print(f"{valid_count} valid questions: {count:,} articles ({percentage:.1f}%)")


def analyze_leakage_patterns(stats: Dict):
    """Analyze patterns in answer leakage."""
    
    if not stats['leakage_details'] and not stats['chosen_question_leakage_details'] and not stats['final_question_leakage_details']:
        return
    
    print(f"\n🕵️ DETAILED LEAKAGE ANALYSIS:")
    print("=" * 80)
    
    # Analyze leakage by question position for generated questions
    if stats['leakage_details']:
        leakage_by_position = defaultdict(int)
        for detail in stats['leakage_details']:
            leakage_by_position[detail['question_num']] += 1
        
        print("Generated questions - Leakage by position:")
        for q_num in sorted(leakage_by_position.keys()):
            count = leakage_by_position[q_num]
            print(f"  Question {q_num}: {count} cases")
        
        # Show most common leaked answers in generated questions
        answer_counts = defaultdict(int)
        for detail in stats['leakage_details']:
            answer_counts[detail['answer']] += 1
        
        print(f"\nGenerated questions - Most frequently leaked answers:")
        for answer, count in sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  '{answer}': {count} times")
    
    # Analyze leakage patterns for chosen questions (before editing)
    if stats['chosen_question_leakage_details']:
        chosen_leakage_by_selected = defaultdict(int)
        for detail in stats['chosen_question_leakage_details']:
            chosen_leakage_by_selected[detail['selected_question_num']] += 1
        
        print(f"\nChosen questions - Leakage by originally selected question:")
        for q_num in sorted(chosen_leakage_by_selected.keys()):
            count = chosen_leakage_by_selected[q_num]
            print(f"  Originally Question {q_num}: {count} cases")
        
        # Show most common leaked answers in chosen questions
        chosen_answer_counts = defaultdict(int)
        for detail in stats['chosen_question_leakage_details']:
            chosen_answer_counts[detail['answer']] += 1
        
        print(f"\nChosen questions - Most frequently leaked answers:")
        for answer, count in sorted(chosen_answer_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  '{answer}': {count} times")
    
    # Analyze leakage patterns for final questions (after editing)
    if stats['final_question_leakage_details']:
        final_leakage_by_selected = defaultdict(int)
        for detail in stats['final_question_leakage_details']:
            final_leakage_by_selected[detail['selected_question_num']] += 1
        
        print(f"\nFinal questions - Leakage by originally selected question:")
        for q_num in sorted(final_leakage_by_selected.keys()):
            count = final_leakage_by_selected[q_num]
            print(f"  Originally Question {q_num}: {count} cases")
        
        # Show most common leaked answers in final questions
        final_answer_counts = defaultdict(int)
        for detail in stats['final_question_leakage_details']:
            final_answer_counts[detail['answer']] += 1
        
        print(f"\nFinal questions - Most frequently leaked answers:")
        for answer, count in sorted(final_answer_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  '{answer}': {count} times")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze question generation pipeline statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stats.py /path/to/qgen_output.jsonl
    python stats.py /path/to/qgen_output.jsonl --questions-per-article 5
        """
    )

    parser.add_argument("--input-path", type=str, default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/recent/qgen/clean/"#clean/deepseek-chat-v3-0324_forbes-2023_50000_free_3.jsonl"
                        , help="Path to question generation output JSONL file or directory containing JSONL files")
    parser.add_argument("--questions-per-article", type=int, default=3, 
                       help="Number of questions generated per article (default: 3)")
    parser.add_argument("--show-leakage-examples", action="store_true",
                       help="Show detailed leakage examples")
    
    args = parser.parse_args()
    
    # Check if input path exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input path {args.input_path} does not exist")
        return 1

    data = []
    if os.path.isdir(args.input_path):
        # Load all .jsonl files in the directory
        jsonl_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.jsonl')]
        if not jsonl_files:
            print(f"Error: No .jsonl files found in directory {args.input_path}")
            return 1
        print(f"Loading data from directory: {args.input_path}")
        for file in jsonl_files:
            print(f"  Loading {file}")
            file_data = load_jsonl_file(file)
            if file_data:
                data.extend(file_data)
    else:
        # Single file
        print(f"Loading data from: {args.input_path}")
        data = load_jsonl_file(args.input_path)
    
    if not data:
        print("Error: No data loaded from input path")
        return 1
    
    print(f"Loaded {len(data)} articles")
    # Analyze statistics
    stats = analyze_pipeline_statistics(data, args.questions_per_article)
    
    # Print comprehensive statistics
    print_comprehensive_statistics(stats, args.questions_per_article)
    
    # Create pipeline effectiveness table
    create_pipeline_effectiveness_table(stats)
    
    # Analyze failure modes
    analyze_failure_modes(data, args.questions_per_article)
    
    # Analyze leakage patterns
    if args.show_leakage_examples:
        analyze_leakage_patterns(stats)
    
    # Summary recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 80)
    
    valid_rate = stats['valid_questions'] / stats['total_questions_generated'] * 100 if stats['total_questions_generated'] > 0 else 0
    leakage_rate = stats['questions_with_leakage'] / (stats['questions_with_leakage'] + stats['questions_without_leakage']) * 100 if (stats['questions_with_leakage'] + stats['questions_without_leakage']) > 0 else 0
    
    # Calculate final question leakage rate
    total_final_questions = stats['final_questions_with_leakage'] + stats['final_questions_without_leakage']
    final_leakage_rate = stats['final_questions_with_leakage'] / total_final_questions * 100 if total_final_questions > 0 else 0
    
    if valid_rate < 50:
        print("⚠️  Low validation rate (<50%) - Consider improving question generation prompts")
    
    if leakage_rate > 20:
        print("⚠️  High leakage rate in generated questions (>20%) - Consider improving leakage detection and prevention")
    
    if final_leakage_rate > 20 and total_final_questions > 0:
        print("⚠️  High leakage rate in final questions (>20%) - Selection process not effectively filtering leakage")
    
    if stats['articles_with_no_good_question'] / stats['source_articles'] > 0.3:
        print("⚠️  High no-good-question rate (>30%) - Consider relaxing validation criteria")
    
    usable_rate = (stats['articles_with_valid_questions'] - stats['articles_with_no_good_question']) / stats['source_articles'] * 100
    if usable_rate > 70:
        print("✅ Good pipeline efficiency (>70% usable articles)")
    elif usable_rate > 50:
        print("⚡ Moderate pipeline efficiency (50-70% usable articles)")
    else:
        print("❌ Low pipeline efficiency (<50% usable articles)")
    
    print(f"\nOverall pipeline converts {stats['source_articles']:,} articles → {stats['articles_with_valid_questions'] - stats['articles_with_no_good_question']:,} usable questions ({usable_rate:.1f}% success rate)")
    
    return 0


if __name__ == "__main__":
    exit(main())
