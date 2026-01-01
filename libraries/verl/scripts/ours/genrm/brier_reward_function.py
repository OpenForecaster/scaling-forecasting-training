import json
import logging
import os
import re
from typing import Optional

from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests

# Ensure no proxy is used for local connections
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

BASE_URL = "http://127.0.0.1:30000"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32
MODEL_NAME = "qwen3-4b-non-think"
GENRM_PROMPT_TEMPLATE = """
The following is a math problem and an AI solution:

[Math Problem]

{problem}

[AI Solution]

{solution}

Your task is to review and critique the solution step by step, and output whether the AI solution is correct.

Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
""".strip()


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

def test_server_connection():
    """Test if the vLLM server is reachable."""
    try:
        session = requests.Session()
        session.trust_env = False
        session.proxies = {}
        
        # Test the models endpoint first
        models_url = f"{BASE_URL}/v1/models"
        response = session.get(models_url, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Successfully connected to vLLM server at {BASE_URL}")
            models_data = response.json()
            logger.info(f"Available models: {[model.get('id', 'unknown') for model in models_data.get('data', [])]}")
            return True
        else:
            logger.error(f"Server responded with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to vLLM server at {BASE_URL}: {e}")
        return False

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "THINK STEP BY STEP WHEN MATCHING THE STUDENT RESPONSE WITH THE GROUND TRUTH. If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

VERIFIER_PASS_TAG = "Final Decision: Yes"


def extract_last_boxed(text: str) -> str:
    """
    Extract the last occurrence of a boxed answer from the input text.
    
    Returns:
        The content inside the last \boxed{...} or None if not found.
    """
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """
    Try to extract the final answer from the text using several candidate patterns.
    
    Returns:
        The extracted answer as a string, or None if none of the patterns match.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]
    
    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()
    
    return last_match


def extract_solution(solution_str: str) -> str:
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)

def extract_score(solution_str: str) -> tuple[float, int]:
    # Find the latest occurrence of 0 or 1 in the solution string and return the one which occurs last 
    last0 = solution_str.rfind("0")
    last1 = solution_str.rfind("1")
    
    extraction = 0 
    if last0 != -1 or last1 != -1:
        extraction = 1
        
    if last1 > last0:
        return 1, extraction
    else:
        return 0, extraction

# Define the prompt templates for the judge
def get_judge_prompt_with_gt(question, target, response, incorrect_options=None, cot=True):
    """
    Generate a prompt for the judge with ground truth.
    
    Args:
        question: The question being asked
        target: The ground truth answer
        response: The response to judge
        incorrect_options: Optional string containing incorrect options
        cot: Whether to use a COT prompt
        
    Returns:
        A formatted prompt string for the judge
    """
    # The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased.

    prompt = f"""Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For a response to "match", it must have the same information as in the ground-truth (not less nor unnecessary extra). 
The response can be more specific than the ground-truth (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased. 
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be <= 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
"""

    if incorrect_options:
        prompt += f"\n{incorrect_options}"
        
    prompt += f"""Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags."""
    
    if cot:
        prompt += "\nThink step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS."
    else :
        prompt += "\nYOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS."
        
# Think step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS.
# YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS.

    return prompt




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



def calculate_brier_score_binary(probability: float, resolution: int) -> float:
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


def compute_score_binary(solution_dict, resolution):
    # Handle the case where solution_dict is a dictionary {answer: probability}
    if isinstance(solution_dict, dict) and len(solution_dict) > 0:
        # Get the first (and likely only) probability value
        probability = list(solution_dict.values())[0]
    else:
        # Fallback: try to extract probability from string if it's not a dict
        if isinstance(solution_dict, str):
            probability = extract_probability(solution_dict)
        else:
            probability = None
    
    brier_score = 0.25
    format_reward = 0.0
    
    if probability and probability >= -0.1 and probability <= 1.1:
        brier_score = calculate_brier_score_binary(probability, resolution)
        format_reward = 0
    else:
        format_reward = -1
    
    return -brier_score, format_reward, probability




def get_response(problem, solution_str, ground_truth):
    # Test server connection on first call
    if not hasattr(get_response, '_connection_tested'):
        get_response._connection_tested = True
        if not test_server_connection():
            logger.error("Cannot connect to vLLM server. Please check if the server is running and accessible.")
            return None
    
    answer_str = extract_answer(solution_str)
    
    if answer_str and len(answer_str) > 1:
        prompt = get_judge_prompt_with_gt(problem, ground_truth, answer_str)
    else:
        prompt = get_judge_prompt_with_gt(problem, ground_truth, "NO ANSWER")
    
    prompt += " /no_think"
                        
    
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Content-Type": "application/json"}
            chat_url = f"{BASE_URL}/v1/chat/completions"
            data = {"model": MODEL_NAME, "messages": messages}
            
            # Add session with no proxy to bypass potential proxy issues
            session = requests.Session()
            session.trust_env = False  # Ignore proxy environment variables
            session.proxies = {}  # No proxies
            
            output = session.post(chat_url, headers=headers, json=data, timeout=30)
            
            # Check if the request was successful
            if output.status_code != 200:
                print(f"HTTP Error {output.status_code}: {output.text}")
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2**attempt)
                    print(f"Retrying in {delay} seconds...")
                    sleep(delay)
                    continue
                else:
                    raise Exception(f"HTTP {output.status_code}: {output.text}")
            
            # Try to parse JSON response
            try:
                response_json = output.json()
            except json.JSONDecodeError as json_err:
                print(f"JSONDecodeError: {json_err}")
                print(f"Response text: {output.text[:500]}...")  # Print first 500 chars
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2**attempt)
                    print(f"Retrying in {delay} seconds...")
                    sleep(delay)
                    continue
                else:
                    raise Exception(f"Invalid JSON response: {output.text[:200]}")
            
            # Extract the response content
            if "choices" not in response_json or len(response_json["choices"]) == 0:
                print(f"Unexpected response format: {response_json}")
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2**attempt)
                    print(f"Retrying in {delay} seconds...")
                    sleep(delay)
                    continue
                else:
                    raise Exception(f"Invalid response format: {response_json}")
            
            response = response_json["choices"][0]["message"]["content"]
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            if "access denied" in error_msg or "squid" in error_msg or "proxy" in error_msg:
                print(f"Proxy/firewall blocking request: {e}")
                print("This appears to be a network configuration issue. Check proxy settings.")
            
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")
                print("Possible solutions:")
                print("1. Check if vLLM server is running on the correct port")
                print("2. Verify no proxy/firewall is blocking localhost connections")
                print("3. Try connecting directly: curl http://localhost:30000/v1/models")

    raise ConnectionRefusedError(f"Failed to run the model for {prompt}!")


def compute_reward(response):
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result == "True")
    except Exception as e:
        print(e)
    return reward_score


def compute_score(data_source, solution_str, ground_truth, extra_info):
    split = extra_info["split"]
    question_source = extra_info.get("question_source", "unknown")

    if "test" in split or "val" in split.lower() or "binary" in question_source.lower():
        from verl.utils.reward_score import default_compute_score

        func_rm_score = default_compute_score(data_source, solution_str, ground_truth, extra_info)
        return func_rm_score
    
    else:
        problem = extra_info["question"]
        matcher_response = get_response(problem, solution_str, ground_truth)
            
        score = 0.0
        format_reward = -1
        
        correctness, extraction_success = extract_score(matcher_response)
        
        try: 
            correctness = int(correctness)
        except:
            correctness = None
        
        last_ans = extract_answer(solution_str)
        final_prob = extract_probability(solution_str)
        if last_ans and final_prob:
            solution = {last_ans: final_prob}
        else:
            solution = {}
        
        reward_score = 0.0
        if correctness != None:
            outcomes = list(solution.keys())
            final_prob = 1.0 
            if len(outcomes) > 0 and outcomes[0] and solution[outcomes[0]] :
                final_prob = solution[outcomes[0]]
                format_reward = 0 
                
                # print(f"Extracted Answer: {outcomes[0]}")
                # print(f"Extracted Probability: {final_prob}")
            
            reward_score = 1 + calculate_brier_score(final_prob, int(correctness) == 1)
                

        return reward_score


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
            futures.append(future)

        results = [future.result() for future in futures]

    return results
