import logging
import os
import re
from typing import Optional
import torch
from vllm import LLM, SamplingParams
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl import DataProto
from tensordict import TensorDict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

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
        answer_text = answer_text[:100]
        prob_text = match.group(3).strip()
        prob_text = prob_text[:100]
        
        # Try to parse probability as float
        try:
            probability = float(prob_text)
            # Validate probability is between 0 and 1
            if 0 <= probability <= 1:
                answer_prob_dict[answer_text] = probability
            # else:
            #     logger.warning(f"Invalid probability value {probability} for answer {answer_num}")
        except (ValueError, TypeError):
            logger.warning(f"Could not parse probability '{prob_text}' for answer {answer_num}")
            continue
    
    # Log extraction results
    if answer_prob_dict:
        total_prob = sum(answer_prob_dict.values())
        logger.debug(f"Extracted {len(answer_prob_dict)} answers with total probability {total_prob:.3f}")
        # if total_prob - 1.0 > 0.1:  # Warning if probabilities don't sum to ~1
        #     logger.warning(f"Probabilities sum to {total_prob:.3f}, not 1.0")
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


def calculate_individual_brier_score(probability: float, is_correct: bool) -> float:
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
        return - (probability ** 2)
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


class RewardModelWorker(Worker):
    def __init__(self, config):
        """
        Initializes the reward model worker with its configuration and sampling parameters.
        """
        super().__init__()
        self.config = config
        print("Reward model initialized!!!")
        self.sampling_params = SamplingParams(temperature=0, max_tokens=512)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Initialize the language model and tokenizer on CUDA device 7.
        """
        # Set environment variables to optimize memory allocation
        # Note: Don't use expandable_segments with vLLM as it conflicts with memory pool
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Import Ray to check GPU resources
        import ray
        try:
            # Get Ray GPU resources
            gpu_resources = ray.available_resources().get('GPU', 0)
            print(f"Ray GPU resources available: {gpu_resources}")
        except:
            print("Could not get Ray GPU resources")
        
        # Debug: Check available GPUs
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Use the last available GPU for the verifier
        # This ensures the verifier doesn't conflict with training processes
        device_count = torch.cuda.device_count()
        if device_count > 0:
            # GPU 7 becomes index 4 when CUDA_VISIBLE_DEVICES=0,1,2,3,7
            verifier_device = device_count - 1  # This maps to physical GPU 7
            print(f"Setting verifier to use GPU {verifier_device}")
            torch.cuda.set_device(verifier_device)
            
            # Check available memory on the selected device
            memory_allocated = torch.cuda.memory_allocated(verifier_device)
            memory_reserved = torch.cuda.memory_reserved(verifier_device)
            memory_total = torch.cuda.get_device_properties(verifier_device).total_memory
            print(f"GPU {verifier_device} memory: {memory_allocated/1024**3:.2f}GB allocated, {memory_reserved/1024**3:.2f}GB reserved, {memory_total/1024**3:.2f}GB total")
        else:
            raise RuntimeError("No CUDA devices available")
        
        # Clear any existing CUDA cache
        torch.cuda.empty_cache()
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Verify we're on the correct device
        current_device = torch.cuda.current_device()
        print(f"Verifier initialized on GPU {current_device}")
        
        # Use lower memory utilization and smaller max model len for GPU 7
        self.llm = LLM(
            model=self.config.model.path, 
            gpu_memory_utilization=0.4,  # Increased from 0.3
            tensor_parallel_size=1,  # Use single GPU
            trust_remote_code=self.config.model.get("trust_remote_code", True),
            max_model_len=1024,  # Very small model length to save memory
            dtype="bfloat16",  # Use bfloat16 for memory efficiency
            # max_num_batched_tokens=2048,  # Limit batch size
            # max_num_seqs=4  # Limit number of sequences
        )
        self.tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=self.config.model.get("trust_remote_code", False)
        )
        self.llm.sleep(2)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute the reward model score for each data item.
        
        For every data instance, the function decodes the sequence of prompt and response
        tokens, extracts the solution, and then uses a language model to verify the answer.
        A reward score is then computed based on whether the verified answer is correct and the
        token length difference from the ground truth.
        
        Returns:
            A DataProto object containing the computed reward scores.
        """
        # torch.cuda.empty_cache()
        self.llm.wake_up()
        sequence_strs = []
        ground_truths = []
        questions = []
        valid_response_lengths = []
        resolutions = []
        question_sources = []
        multiple_outcomes_flags = []  # Track multiple_outcomes for each sample
        
        # Process each data item to create a sequence string and extract necessary fields.
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_lengths.append(valid_response_length)

            # Concatenate valid prompt and response tokens.
            sequence = torch.cat((valid_prompt_ids, response_ids[:valid_response_length]))
            sequence_str = self.tokenizer.decode(sequence[-1024:]) # avoid risk of getting too long answer extracted
            sequence_strs.append(sequence_str)

            # Extract question and ground truth from non-tensor batch.
            question = data_item.non_tensor_batch["extra_info"].get("question", "unknown")
            resolution = data_item.non_tensor_batch["extra_info"].get("resolution", -1)
            ground_truth = data_item.non_tensor_batch["reward_model"].get("ground_truth", "unknown")
            question_source = data_item.non_tensor_batch["extra_info"].get("question_source", "unknown")
            multiple_outcomes = data_item.non_tensor_batch["extra_info"].get("multiple_outcomes", False)
            
            questions.append(question)
            ground_truths.append(ground_truth)
            resolutions.append(resolution)
            question_sources.append(question_source)
            multiple_outcomes_flags.append(multiple_outcomes)
            
        # Extract solutions from the decoded sequences.
        solutions = []
        format_rewards = []
        for i, seq in enumerate(sequence_strs):
            answer = seq
        
            if "</think>" in seq:
                answer = seq.split("</think>")[1]
                
            # Extract the final answer based on format
            if multiple_outcomes_flags[i]:
                # Extract multiple answers and probabilities as dictionary
                answer_prob_dict = extract_multiple_answers_and_probabilities(answer)
                final_ans = answer_prob_dict  # Store dictionary of answers and probabilities
            else:
                # Extract single answer (keep original type, don't cast)
                last_ans = extract_answer(answer)
                final_prob = extract_probability(answer)
                if last_ans and final_prob:
                    final_ans = {last_ans: final_prob}
                elif last_ans:
                    final_ans = {last_ans: 1}
                else:
                    final_ans = {}
            
            solutions.append(final_ans)

        # Prepare messages for the verification prompt
        messages = []
        sample_indices = []  # Track which sample each prompt belongs to
        answer_indices = []  # Track which answer each prompt corresponds to
        
        for i in range(len(questions)):
            if multiple_outcomes_flags[i] and solutions[i] and isinstance(solutions[i], dict):
                # Multiple outcomes: judge each answer individually
                for answer_idx, (answer_text, prob) in enumerate(solutions[i].items()):
                    prompt = get_judge_prompt_with_gt(questions[i], ground_truths[i], answer_text)
                    if "qwen3" in self.config.model.path.lower():
                        prompt += " /no_think"
                    messages.append(prompt)
                    sample_indices.append(i)
                    answer_indices.append(answer_idx)
            else:
                # Single outcome: judge only the first (or only) answer
                if solutions[i] and isinstance(solutions[i], dict) and len(list(solutions[i].keys())) > 0:
                    prompt = get_judge_prompt_with_gt(questions[i], ground_truths[i], list(solutions[i].keys())[0])
                else:
                    prompt = get_judge_prompt_with_gt(questions[i], ground_truths[i], "NO ANSWER")
                
                if "qwen3" in self.config.model.path.lower():
                    prompt += " /no_think"
                            
                messages.append(prompt)
                sample_indices.append(i)
                answer_indices.append(0)  # Always index 0 for single outcome
        
        # Generate verification responses using the language model.
        outputs = self.llm.generate(messages, self.sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]

        # Initialize reward tensor with the same shape as responses.
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        acc_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extraction_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        brier_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Group results by sample index for multiple outcomes
        sample_results = {}
        for prompt_idx, response in enumerate(responses):
            sample_idx = sample_indices[prompt_idx]
            answer_idx = answer_indices[prompt_idx]
            
            # Extract judgment from response
            correctness, extraction_success = extract_score(response)
            
            if sample_idx not in sample_results:
                sample_results[sample_idx] = {
                    'judgments': [],
                    'extractions': [],
                    'responses': []
                }
            
            # Ensure we have enough space for this answer index
            while len(sample_results[sample_idx]['judgments']) <= answer_idx:
                sample_results[sample_idx]['judgments'].append(0)
                sample_results[sample_idx]['extractions'].append(0)
                sample_results[sample_idx]['responses'].append("")
            
            sample_results[sample_idx]['judgments'][answer_idx] = correctness
            sample_results[sample_idx]['extractions'][answer_idx] = extraction_success
            sample_results[sample_idx]['responses'][answer_idx] = response

        # Compute a reward score for each data item.
        for i in range(len(sequence_strs)):
            ground_truth = ground_truths[i]
            solution = solutions[i]
            og_response = sequence_strs[i]
            valid_response_length = valid_response_lengths[i]
            resolution = resolutions[i]
            question_source = question_sources[i]
            multiple_outcomes = multiple_outcomes_flags[i]
            
            correctness = 0
            extraction_success = 1
            final_prob = 1.0
            format_reward = -1
            score = 0 
            
            if "binary" in question_source or "metaculus" in question_source:
                assert resolution != -1, "Resolution is not provided"
                model_response = str(og_response)
                if "</think>" in model_response:
                    model_response = model_response.split("</think>")[1]
                
                score, format_reward, final_prob = compute_score_binary(model_response, resolution)
                correctness = 0
                extraction_success = 1
                
                if final_prob:
                    if resolution == 1:
                        if final_prob >= 0.5:
                            correctness = 1
                        else:
                            correctness = 0
                    else:
                        if final_prob < 0.5:
                            correctness = 1
                        else:
                            correctness = 0
                
            else:
                if multiple_outcomes and i in sample_results:
                    # Multiple outcomes: find the answer with highest probability and use its judgment
                    judgments = sample_results[i]['judgments']
                    extractions = sample_results[i]['extractions']
                    any_correct = False 
                    
                    # Get all answers and their probabilities
                    if solution and isinstance(solution, dict):
                        answer_items = list(solution.items())
                        if answer_items:
                            # Find the answer with the highest probability
                            best_answer_idx = 0
                            best_prob = answer_items[0][1] if len(answer_items) > 0 else 0
                            best_correctness = judgments[0] if len(judgments) > 0 else 0
                            any_correct = False
                            
                            for idx, (answer_text, prob) in enumerate(answer_items):
                                if prob > best_prob:
                                    best_prob = prob
                                    best_answer_idx = idx
                                    best_correctness = judgments[idx]
                                    
                                # Use the judgment for the best answer
                                if idx < len(judgments):
                                    correctness = judgments[idx]
                                    any_correct = any_correct or int(correctness) == 1
                                    extraction_success = extractions[idx] if idx < len(extractions) else 1
                                    final_prob = prob
                                    
                                    # Convert correctness to int for consistency
                                    try:
                                        correctness = int(correctness)
                                    except:
                                        correctness = 0
                                    
                                    # Calculate Brier score
                                    score += calculate_individual_brier_score(final_prob, int(correctness) == 1)
                            
                            if not any_correct:
                                score -= 1
                            
                            score += 1 
                            format_reward = 0
                            correctness = int(best_correctness)
                            
                            if i < 1:
                                print(f"Multiple outcomes - Best answer: {answer_items[best_answer_idx][0]}")
                                print(f"Best probability: {best_prob}")
                                print(f"Judgment: {correctness}")
                        else:
                            extraction_success = 0
                            score = 0  #  calculate_brier_score(final_prob, False)
                    else:
                        extraction_success = 0
                        score = 0  # 1 + calculate_brier_score(final_prob, False)
                else:
                    # Single outcome: use existing logic
                    if i in sample_results and len(sample_results[i]['judgments']) > 0:
                        correctness = sample_results[i]['judgments'][0]
                        extraction_success = sample_results[i]['extractions'][0]
                        matcher_response = sample_results[i]['responses'][0]
                    else:
                        correctness, extraction_success = extract_score("")
                        matcher_response = ""
                    
                    useful_response = og_response
                    if "</think>" in useful_response:
                        useful_response = useful_response.split("</think>")[1]
                    
                    try: 
                        correctness = int(correctness)
                    except:
                        correctness = None
                        
                    if correctness is not None:
                        outcomes = list(solution.keys()) if solution else []
                        final_prob = 1.0 
                        if len(outcomes) > 0 and outcomes[0] and solution[outcomes[0]]:
                            final_prob = solution[outcomes[0]]
                            format_reward = 0 
                            
                            # if i < 2:
                            #     print(f"Extracted Answer: {outcomes[0]}")
                            #     print(f"Extracted Probability: {final_prob}")
                        
                        score = 1 + calculate_brier_score(final_prob, int(correctness) == 1)
                
            if i < 1:
                print(f"OG Response: {og_response}")
                print(f"Ground Truth: {ground_truth}")
                if i in sample_results and len(sample_results[i]['responses']) > 0:
                    print(f"Matcher Response: {sample_results[i]['responses'][0]}")
                print(f"Solution: {solution}")
                print(f"Multiple Outcomes: {multiple_outcomes}")
                print(f"Correctness: {correctness}")
                print(f"Final Prob: {final_prob}")
                print(f"Score: {score}")
                print(f"Format Reward: {format_reward}")
                print(f"--------------------------------")
            
            if score is None or not isinstance(score, float):
                score = 0
            
            if correctness is None or not isinstance(correctness, int):
                correctness = 0
                
            if extraction_success is None or not isinstance(extraction_success, int):
                extraction_success = 0
            
            # Record the score at the final valid response token index.
            reward_tensor[i, valid_response_length - 1] = score + format_reward # + correctness
            acc_tensor[i, valid_response_length - 1] = correctness
            extraction_tensor[i, valid_response_length - 1] = extraction_success
            brier_tensor[i, valid_response_length - 1] = score

        batch = TensorDict({"rm_scores": reward_tensor, "acc_scores": acc_tensor, "extraction_scores": extraction_tensor, "brier_scores": brier_tensor}, batch_size=reward_tensor.shape[0])
        # batch = TensorDict({"rm_scores": reward_tensor}, batch_size=reward_tensor.shape[0])
        self.llm.sleep(2)
        torch.cuda.empty_cache()
        return DataProto(batch=batch)