"""
Data loading utilities for various forecasting datasets.
Provides functions to load and filter datasets from:
- Halawi forecasting dataset
- Metaculus binary questions
- Manifold markets
- Menge dataset
- Infinite Games
- Retrieved articles data (with BM25 ranking)
- MCQ (multiple choice) forecasting questions
"""

import datasets
import numpy as np 


def load_halawi_data(split="train", raw=False):
    path = "YuehHanChen/forecasting"
    if raw:
        path += "_raw"
        
    ds = datasets.load_dataset(path)[split]
    # print(ds.column_names)
    if raw:
        
        # Only keep rows with question_type == BINARY or binary
        ds = ds.filter(lambda x: x["question_type"].lower() == "binary")
        
        # Only keep rows with resolution == 1 or 1.0 or 0 or 0.0 (in str)
        # ds = ds.filter(lambda x: x["resolution"] in ["1.0", "0.0"]) 
        ds = ds.filter(lambda x: x["resolution"] in ["1", "1.0", "0", "0.0"])
    
    return ds


def load_metaculus_data(split="train", nr_forecasters=1):
    path = "nikhilchandak/metaculus-binary"
    ds = datasets.load_dataset(path)["train"]

    # date_resolve_at, date_begin, date_close, nr_forecasters 
    # Only keep rows with 
    
    # If split is train, only keep rows with date resolve at before June 30, 2024
    if split == "train":
        # ds = ds.filter(lambda x: x["date_resolve_at"] < "2024-06-30")
        ds = ds.filter(lambda x: x["date_begin"] < "2024-06-30")
        

    # If split is test, only keep rows with date resolve at after July 1 2024
    if split == "test":
        ds = ds.filter(lambda x: x["date_begin"] >= "2024-06-30")
        
        # Only keep rows with nr_forecasters > 10
        ds = ds.filter(lambda x: x["nr_forecasters"] >= nr_forecasters)

    return ds

def load_menge_data(split="validation", data_type="binary"):
    path = "/fast/nchandak/forecasting/datasets/menge/" + data_type + "_" + split + ".json"
        
    # Load dataset
    ds = datasets.Dataset.from_json(path)
    
    # Print column names 
    print("Menge Column names: ", ds.column_names)
    
    return ds


def load_manifold_data(split="train", nr_forecasters=1):
    path = "/fast/nchandak/forecasting/datasets/manifold"
    if split == "distill":
        path += "/binary_mini.json"
    elif split == "validation":
        # path += "/binary_mini.json"
        path += "/manifold_binary_validation_set.json"
    elif split == "test":
        path += "/binary_test.json"
        
    # Load dataset
    ds = datasets.Dataset.from_json(path)
    
    # Print column names 
    print("Manifold Column names: ", ds.column_names)
    
    # Apply same filtering as metaculus data
    # if split == "train":
    #     ds = ds.filter(lambda x: x["date_resolve_at"] < "2024-06-30")
    
    # if split == "test":
    #     ds = ds.filter(lambda x: x["date_begin"] >= "2024-05-30")
        # ds = ds.filter(lambda x: x["nr_forecasters"] >= nr_forecasters)
        
    return ds

def filter_halawi_data(ds, begin_date="2023-01-01", end_date="2023-06-01"):
    useful_subset = ds.filter(lambda x: x["date_begin"] > begin_date and x["date_resolve_at"] < end_date)
    return useful_subset



def load_infinitegames_data(split="train", nr_forecasters=1):
    if "balanced" in split:
        path = "/fast/nchandak/forecasting/datasets/infinitegames/binary_balanced_test.json"
    else:
        path = "/fast/nchandak/forecasting/datasets/infinitegames/binary_test.json"
    
    
    # path = "/fast/nchandak/forecasting/datasets/infinitegames/binary_balanced_test.json"
    ds = datasets.Dataset.from_json(path)
    
    print("Column names: ", ds.column_names)
    
    # ds = ds.select(range(10))
    # resolution value counts
    # print(np.unique(ds["resolution"], return_counts=True))
        
    return ds


def load_retreived_data(split="train", data_type="retrieval_metaculus", nr_forecasters=1):
    prefix = "/fast/sgoel/forecasting/news/retrieval/"
    path = prefix + data_type + "/"
    # path = "/fast/sgoel/forecasting/news/retrieval/metaculus-binary_apnews_7_365/"
    
    # Load the entire dataset
    dataset = datasets.load_from_disk(path)
    
    # If the dataset has splits and a specific split is requested
    if hasattr(dataset, 'keys') and split in dataset:
        print("Split found in dataset: ", split)
        dataset = dataset[split]

    ds = dataset
    print("Length before split: ", len(ds))
    # If split is train, only keep rows with date resolve at before June 30, 2024
    if split == "train":
        # ds = ds.filter(lambda x: x["date_resolve_at"] < "2024-06-30")
        ds = ds.filter(lambda x: x["date_begin"] < "2024-06-30")
        

    # If split is test, only keep rows with date resolve at after July 1 2024
    if split == "test":
        ds = ds.filter(lambda x: x["date_begin"] >= "2024-06-30")
        
        # Only keep rows with nr_forecasters > 10
        ds = ds.filter(lambda x: x["nr_forecasters"] >= nr_forecasters)
    
    
    # keep only rows with date_close between 2018-01-01 and 2021-12-31
    ds = ds.filter(lambda x: x["date_close"] >= "2018-01-01" and x["date_close"] <= "2021-12-31")
    
    print("Length after split: ", len(ds))
    # Filter columns for which retrieved_articles is empty
    ds = ds.filter(lambda x: len(x["retrieved_articles"]) >= 3)
    
    # print column names
    print("Column names: ", ds.column_names)
    # print first 10 rows
    # print(dataset[:10])
    # print length of the dataset
    print("Length of dataset only keeping rows with retrieved articles: ", len(ds))
    
    # create prompt for each row
    # Create a new column with the prompts
    # Check if we should use retrieval based on data_type
    use_retrieval = "without" not in data_type
    print(f"Using retrieval: {use_retrieval}")
    
    def create_prompt_for_row(row):
        return create_retreived_prompt(
            row["question"], 
            row["background"], 
            row["resolution_criteria"], 
            row["date_begin"], 
            row["date_close"], 
            row["retrieved_articles"] if use_retrieval else []
        )
    
    # Apply the function to create prompts for all rows
    ds = ds.map(lambda row: {"prompt": create_prompt_for_row(row)})
    # ds = ds.select(range(5,6))
    
    # pretty print the prompt iteratively 
    # for i, row in enumerate(ds):
    #     print(row["prompt"])
    #     print("-"*100)
    
    return ds


def create_retreived_prompt(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str,
    retrieved_articles: list[dict] = []
) -> str:
    """
    Format the prompt given the row data.
    """
    
    prefix = f"""
Question: {question}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}"""

    if len(retrieved_articles) > 0:
        prefix += "\n\nWe have retrieved the following articles for this question from cleaned Common Crawl news data using the BM25 ranking algorithm (so it is possible that some of them might not be too relevant to the question):"
        for article in retrieved_articles:
            prefix += f"\n\nTitle: {article['title']}"
            prefix += f"\nURL: {article['url']}"
            prefix += f"\nDate Published: {article['date_publish']}"
            prefix += f"\nContent: {article['maintext']}"
    else:
        prefix += "\n"
            
    return prefix

def add_idx_column(ds: datasets.Dataset) -> datasets.Dataset:
    return ds.map(lambda x, idx: {"idx": idx}, with_indices=True)

def load_mcq_manifold_data(split="train", nr_forecasters=1, volume=4000):
    path = "/fast/nchandak/forecasting/datasets/manifold/mcq_raw_train.json"
    if split == "test":
        path = "/fast/nchandak/forecasting/datasets/manifold/mcq_test.json"
        
    ds = datasets.Dataset.from_json(path)
    
    # Add idx column
    ds = add_idx_column(ds)
    
    print("Column names: ", ds.column_names)
    
    # Apply same filtering as metaculus data
    # if split == "train":
    #     ds = ds.filter(lambda x: x["date_resolve_at"] < "2024-06-30")
    
    # if split == "test":
    #     ds = ds.filter(lambda x: x["date_begin"] >= "2024-05-30")
        # ds = ds.filter(lambda x: x["nr_forecasters"] >= nr_forecasters)
        
    # Extend to mcq prompt
    ds = ds.map(lambda x: {"full_prompt": ask_probabilities(x['prompt'])})
    # ds = ds.map(lambda x: {"full_prompt": detailed_mcq_prompt(x['prompt'])})
    
    # Filter nr_forecasters column
    ds = ds.filter(lambda x: x["nr_forecasters"] >= nr_forecasters)
    
    # Filter volume column
    ds = ds.filter(lambda x: x["volume"] >= volume)
    
    print("Length after filtering: ", len(ds))
    return ds


def load_mcq_metaculus_data(split="test", nr_forecasters=1):
    path = "/fast/nchandak/forecasting/datasets/metaculus/mcq_raw.json"
    ds = datasets.Dataset.from_json(path)
    
    # Add idx column
    ds = add_idx_column(ds)
    
    # print column names
    print("Column names: ", ds.column_names)
    
    # Make full prompt
    ds = ds.map(lambda x: {"full_prompt": extend_mcq_prompt(x['prompt'])})
    
    # Filter nr_forecasters column
    ds = ds.filter(lambda x: x["nr_forecasters"] >= nr_forecasters)
    
    # If split is train, only keep rows with date resolve at before June 30, 2024
    if split == "train":
        # ds = ds.filter(lambda x: x["date_resolve_at"] < "2024-06-30")
        ds = ds.filter(lambda x: x["date_begin"] < "2024-06-30")
        

    # If split is test, only keep rows with date resolve at after July 1 2024
    if split == "test":
        ds = ds.filter(lambda x: x["date_begin"] >= "2024-06-30")

    return ds

def ask_probabilities(content: str) -> str:
    return f"""You will be asked a forecasting question in multiple choice format. You have to come up with the best estimate for the answer choices. Show your work (reasoning) in <think> </think> tags. After wrapping up your reasoning, return your confidence in each of the answer choices in <answer> </answer> tags. 
Think thoroughly about each of the options and finally format your answer in the following format:

<think> .. </think>
<answer> <A> 0.a </A> <B> 0.b </B> <C> 0.c </C> <D> 0.d> </D> </answer>

IMPORTANT:
- Your <answer> MUST contain the probabilities of each option inside XML tags.
- Each probability inside option XML tags MUST be a decimal between 0 and 1 representing your confidence in that option choice and they should sum to 1 across all options.
- Format your response exactly as shown with the <think> and <answer> tags.

{content}
"""

def extend_mcq_prompt(content: str) -> str:
    return f"""You will be asked a forecasting question in multiple choice format. You have to choose the most likely option from the given options and also report your confidence level in your answer.

Think thoroughly about each of the options and finally format your answer in the following format:

<answer1>
Provide exactly one option number from the choices above (e.g., A, B, C, etc.)
</answer1>
<answer2>
Provide your confidence level in this answer as a decimal between 0 and 1 (e.g., 0.7 for 70% confidence)
</answer2>

IMPORTANT:
- Your <answer1> MUST be exactly one of the option numbers listed above.
- Your <answer2> MUST be a decimal between 0 and 1 representing your confidence.
- Format your response exactly as shown with the <answer1> and <answer2> tags.

{content}
"""

if __name__ == "__main__":
    # ds = load_halawi_data("train", raw=True)
    
    # # print column names 
    # print(ds.column_names)
    # resolutions = ds["resolution"]
    
    # # Count number of 0s and 1s in the resolution column
    # print(np.unique(resolutions, return_counts=True))    
    
    # ds = load_metaculus_data(split="train")
    # ds = load_manifold_data(split="test")
    # print first 10 rows of the dataset
    # print(ds[:10])
    # print length of the dataset of a column
    # ds = load_retreived_data(split="train")
    ds = load_mcq_manifold_data(nr_forecasters=1, volume=4000)
    # print(len(ds["question"]))