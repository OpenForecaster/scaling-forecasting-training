# debug_bm25.py - Utility to inspect Hugging Face dataset format
import json
from datasets import load_dataset
import sys
from pprint import pprint

def load_metaculus_dataset(path: str = "nikhilchandak/metaculus-binary"):
    """Load the Metaculus dataset from Hugging Face."""
    return load_dataset(path)["train"]

def check_all_rows_for_strings(dataset_path: str = "nikhilchandak/metaculus-binary"):
    ds = load_metaculus_dataset(dataset_path)
    print("loaded dataset")
    batch = ds.select(range(0, 10))
    batch_dicts = batch.to_dict("records")
    print(batch_dicts)

if __name__ == "__main__":
    # Get dataset path from command line argument or use default
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "nikhilchandak/metaculus-binary"
    check_all_rows_for_strings(dataset_path)