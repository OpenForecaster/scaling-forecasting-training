"""
Utility functions for formatting forecasting prompts.

This module provides standardized prompt formatting for:
- Binary forecasting questions (YES/NO with probability)
- Freeform forecasting questions (specific answer with probability)
"""

def format_forecasting_prompt(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
) -> str:
    """
    Format a prompt for single outcome forecasting (non-binary questions).
    
    Args:
        question_title: The title of the forecasting question
        background: Background information about the question
        resolution_criteria: Criteria for how the question will be resolved
        answer_type: Expected type of answer (e.g., "number", "date", "text")
        
    Returns:
        Formatted prompt string for single outcome forecasting
    """
    prompt = f"""You will be asked a forecasting question (which might be from the past). You have to come up with the best guess for the final answer. Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Expected Answer Type: {answer_type}

Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. TRY HARD TO COME UP WITH THE BEST GUESS FOR THE FINAL ANSWER. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt


def format_forecasting_prompt_binary(
    question_title: str,
    background: str,
    resolution_criteria: str,
    question_start_date: str = "",
) -> str:
    """
    Format a binary forecasting prompt.
    
    Args:
        question_title: The title of the forecasting question
        background: Background information about the question
        resolution_criteria: Criteria for how the question will be resolved
        question_start_date: Optional start date for the question
        
    Returns:
        Formatted prompt string for binary forecasting
    """
    start_date_text = f"Question Start Date: {question_start_date}\n" if question_start_date else ""
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {question_title}
{start_date_text}Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

    return prompt

