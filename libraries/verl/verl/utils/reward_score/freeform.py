import re
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

def extract_answer(completion: str) -> Optional[str]:
    """
    Extracts the final answer from the LLM's output.
    Returns the raw answer text without type conversion.
    """
    matches = re.finditer(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
    matches_list = list(matches)
    
    if not matches_list:
        return None
    
    # Get the last match
    last_match = matches_list[-1]
    answer_text = last_match.group(1).strip()
    
    return answer_text[:100]

def extract_probability(completion: str) -> Optional[float]:
    """
    Extracts the probability from the LLM's output.
    Returns the probability as a float.
    """
    matches = re.finditer(r"<probability>(.*?)<\/probability>", completion, re.DOTALL)
    matches_list = list(matches)

    if not matches_list:
        return None
    
    # Get the last match
    last_match = matches_list[-1]
    probability_text = last_match.group(1).strip()

    # Try to parse probability as float
    try:
        probability = float(probability_text)
        return probability
    except (ValueError, TypeError):
        return None 

def extract_multiple_answers_and_probabilities(completion: str) -> dict:
    """
    Extracts multiple answers and their corresponding probabilities from the LLM's output.
    Expected format: <answer1> outcome1 </answer1> <probability1> prob1 </probability1>
    
    Returns:
        Dictionary with answers as keys and probabilities as values
        If no valid answers found, returns {}
    """
    answer_prob_dict = {}
    
    # Find all answer-probability pairs
    # Look for pattern: <answerN> ... </answerN> <probabilityN> ... </probabilityN>
    answer_pattern = r"<answer(\d+)>(.*?)<\/answer\1>\s*<probability\1>(.*?)<\/probability\1>"
    matches = re.finditer(answer_pattern, completion, re.DOTALL)
    
    for match in matches:
        answer_num = match.group(1)
        answer_text = match.group(2).strip()
        prob_text = match.group(3).strip()
        
        # Try to parse probability as float
        try:
            probability = float(prob_text)
            # Validate probability is between 0 and 1
            if 0 <= probability <= 1:
                answer_prob_dict[answer_text] = probability
            else:
                logger.warning(f"Invalid probability value {probability} for answer {answer_num}")
        except (ValueError, TypeError):
            logger.warning(f"Could not parse probability '{prob_text}' for answer {answer_num}")
            continue
    
    # Log extraction results
    if answer_prob_dict:
        total_prob = sum(answer_prob_dict.values())
        logger.debug(f"Extracted {len(answer_prob_dict)} answers with total probability {total_prob:.3f}")
        if abs(total_prob - 1.0) > 0.1:  # Warning if probabilities don't sum to ~1
            logger.warning(f"Probabilities sum to {total_prob:.3f}, not 1.0")
    else:
        logger.debug("No valid answer-probability pairs found")
    
    return answer_prob_dict




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
        return -(1 + (probability ** 2))
        # return - (probability ** 2)


def compute_score(solution_str, ground_truth):
    
    answer_str = extract_answer(solution_str)
    probability = extract_probability(solution_str)
    brier_score = 0.0
    format_reward = 0.0
    
    if answer_str and probability and len(answer_str) > 0 and probability >= -0.1 and probability <= 1.1:
        is_correct = answer_str.lower().strip() == ground_truth.lower().strip()
        brier_score = 1 + calculate_brier_score(probability, is_correct)
        format_reward = 0
    else:
        format_reward = -1
    
    return brier_score + format_reward