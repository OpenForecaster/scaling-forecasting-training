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



def calculate_brier_score(probability: float, resolution: int) -> float:
    """
    Calculate Brier score using the formula from eval_freeform.py.
    
    Args:
        probability: Probability assigned to the answer YES (0-1)
        resolution: Resolution of the question (0 or 1)
        
    Returns:
        Brier score (range: [0, 1])
    """
    if resolution == 1:
        # If answer is correct: -(1 - p)^2
        return ((1 - probability) ** 2)
    else:
        # If answer is incorrect: -(1 + p^2)
        return  (probability ** 2)
        # return - (probability ** 2)


def compute_score(solution_str, ground_truth, extra_info):
    
    resolution = extra_info.get("resolution", -1)
    assert resolution != -1, "Resolution is not provided"
    
    probability = extract_probability(solution_str)
    brier_score = 0.25
    format_reward = 0.0
    
    if probability and probability >= -0.1 and probability <= 1.1:
        brier_score = calculate_brier_score(probability, resolution)
        format_reward = 0
    else:
        format_reward = -1
    
    return - brier_score + format_reward