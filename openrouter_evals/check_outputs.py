import json 
from typing import Optional, List
import re 

OUTPUT_DIR = "/fast/nchandak/forecasting/evals/halawi/"

# go through each file in the directory, load the json file, pretty print it 

import os

def extract_last_decimal(text: str) -> Optional[float]:
    """
    Extract the last decimal number (0.xxx) from the given text.
    Returns None if no valid number is found.
    """
    pattern = re.compile(r'0\.\d+')  # Match numbers like 0.xxx
    matches = pattern.findall(text)  # Find all occurrences
    
    if matches:
        return float(matches[-1])  # Return the last matched decimal
    return None

for file in os.listdir(OUTPUT_DIR):
    if file.endswith(".json"):
        with open(os.path.join(OUTPUT_DIR, file)) as f:
            data = json.load(f)
            # print(json.dumps(data, indent=4))
            print(file)
            for row in data:
                final_ans = extract_last_decimal(row["response"])
                print(f"Final answer OG: {row['final_answer']}, NEW: {final_ans}, Resolved: {row['resolution']}")
            print("\n\n\n")
            