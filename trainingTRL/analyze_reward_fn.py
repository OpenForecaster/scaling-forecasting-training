import re


# from train_group import *


def format_reward_func(completions, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    # Pre-compile regex pattern for better performance
    pattern = re.compile(r"^\s*<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>[\s\S]*?<answer>([\s\S]*?)<\/answer>\s*$")
    
    rewards = []
    for completion in completions:
        try:
            # Add synthetic <think> tag
            completion = "<think>" + completion
            
            # Use pre-compiled pattern to check format
            match = pattern.search(completion)
            
            # Simplified reward assignment
            rewards.append(1.0 if match and len(match.groups()) == 2 else 0.0)
            
        except Exception:
            rewards.append(0.0)
            
    return rewards


def brier_scoring_rule(completions, question, resolution, **kwargs):
    """
    Evaluates completions based on:
    2. Otherwise, check for a mathematically correct equation that uses all provided numbers exactly once.
    
    Args:
        completions (list[str]): Generated outputs.
        expecte_answer (list[str]): Expected answers.
    
    Returns:
        list[float]: Reward scores.
    """
    rewards = []
    pattern = re.compile(r"<answer>(.*?)<\/answer>", re.DOTALL)
    
    for completion, query, gt in zip(completions, question, resolution):
        try:
            # Prepend synthetic <think> and extract answer
            match = pattern.search("<think>" + completion)
            print(f"Match: {match}")
            if match is None:
                rewards.append(0)
                continue
                
            # Parse prediction, defaulting to 0.5 if invalid
            try:
                prediction = float(match.group(1).strip())
                if not 0 <= prediction <= 1:
                    prediction = 0.5
            except:
                prediction = 0.5
                
            # Calculate MSE loss
            y_true = float(gt)
            mse_loss = (prediction - y_true) ** 2
            rewards.append(-mse_loss)
            
            # Log interesting examples
            # close1 = (y_true > 0.99 and prediction < 0.001)
            # close0 = (y_true < 0.001 and prediction > 0.99)
            
            # if random.random() < 0.001 or close1 or close0:
            #     with open("raw_outputs/samples_brier2.txt", "a") as f:
            #         f.write(f"Prompt: {query}\n")
            #         f.write(f"Completion: {completion}\n") 
            #         f.write(f"y_pred: {prediction}\n")
            #         f.write(f"y_true: {y_true}\n")
            #         f.write("--------------------------------\n\n")
                    
        except Exception as e:
            logger.info(f"Exception: {e}")
            logger.info(f"Completion: {completion}")
            rewards.append(-0.25)  # Baseline accuracy
            
    return rewards



sample_text = """
Alright, let's break down what's known so far.

The planned launch is set for no earlier than Monday, April 17th. The market in question is whether the orbital test flight of Starship will launch before 7:05 AM local time. Based on the latest updates, the launch is currently scheduled for that exact time.

I should consider the history of Starship launches. Previous attempts have shown that while they often face delays, once a launch is scheduled, it's typically carried out unless major issues arise.

Given the latest information from SpaceX, there are no major complications reported prior to the 7:05 AM launch window. Therefore, my reasoning leans towards a yes.

So, the probability would be based on the likelihood of execution within the planned window without significant hiccups.

Final answer: 0.95
</think>

The Starship launch is expected to happen as scheduled, so the probability is high.

<answer>

</answer</answer>

**
```

Final answer would be based on the likelihood of the launch happening within the specified time frame without significant delays.

</answer

</answer

Putting it all together, the best estimate for whether the event will happen is likely high, so it's 0.95</answer
"""

completions = [sample_text]
resolution = [1.0]
question = ["Starship's orbital test flight will launch before 7:05 AM local time on April 17th"]

print(f"Format Reward Function: {format_reward_func(completions)}")
print(f"Brier Scoring Rule: {brier_scoring_rule(completions, question, resolution)}")
