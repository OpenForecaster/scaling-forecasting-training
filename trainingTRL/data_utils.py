from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np 
from pprint import pprint

def load_halawi_data(split="train", raw=False):
    path = "YuehHanChen/forecasting"
    if raw:
        path += "_raw"
        
    ds = load_dataset(path)[split]
    # print(ds.column_names)
    if raw:
        
        # Only keep rows with question_type == BINARY or binary
        ds = ds.filter(lambda x: x["question_type"].lower() == "binary")
        
        # Only keep rows with resolution == 1 or 1.0 or 0 or 0.0 (in str)
        # ds = ds.filter(lambda x: x["resolution"] in ["1.0", "0.0"]) 
        ds = ds.filter(lambda x: x["resolution"] in ["1", "1.0", "0", "0.0"])
        
        """
        Remove questions that start with "Will I"
        Remove rows for which both background and resolution contain "Not applicable/available"
        Remove question for which both background and resolution are empty
        Remove rows whose question length is < 10 characters
        """
        # ds = ds.filter(lambda x: not x["question"].startswith("Will I"))
        print("Length of dataset before filtering: ", len(ds["question"]))
        
        ds = ds.filter(lambda x: "Not applicable" not in x["background"] and "Not applicable" not in x["resolution"])
        ds = ds.filter(lambda x: x["background"] != "" and x["resolution"] != "")
        
        # print("Length of dataset after filtering background and resolution: ", len(ds["question"]))
        
        ds = ds.filter(lambda x: len(x["question"]) > 10)
        
        # print("Length of dataset after filtering question length: ", len(ds["question"]))
        
        # Remove question contain "In my", "Will I", "Will my" (lowercase)
        ds = ds.filter(lambda x: "in my" not in x["question"].lower() and "will i" not in x["question"].lower() and "will my" not in x["question"].lower())
        
        # print("Length of dataset after filtering question contain 'in my', 'will i', 'will my': ", len(ds["question"]))
        
        # Remove questions after 2023-06-01
        ds = ds.filter(lambda x: x["date_begin"] < "2023-06-01")
        
        # print("Length of dataset after filtering date_begin: ", len(ds["question"]))
        
        # Print length of dataset
        print("Length of final dataset after filtering: ", len(ds["question"]))
    
    return ds

def filter_halawi_data(ds, begin_date="2023-01-01", end_date="2023-06-01"):
    useful_subset = ds.filter(lambda x: x["date_begin"] > begin_date and x["date_resolve_at"] < end_date)
    return useful_subset


def load_manifold_and_metaculus_data(split="train", raw=False):
    path1 = "/fast/nchandak/forecasting/datasets/manifold/binary_"
    path2 = "/fast/nchandak/forecasting/datasets/metaculus/binary_"

    if raw:
        path1 += "raw_"
        path2 += "raw_"
        
    # add split to path1 and path2
    path1 += split + ".json"
    path2 += split + ".json"
    
    # Load dataset
    ds1 = Dataset.from_json(path1)
    ds2 = Dataset.from_json(path2)
    
    # print length of individual datasets
    print("Length of dataset 1: ", len(ds1))
    print("Length of dataset 2: ", len(ds2))
    
    # only keep columns which are present in both datasets
    to_remove = []
    for col in ds1.column_names:
        if col not in ds2.column_names:
            to_remove.append(col)
            
    ds1 = ds1.remove_columns(to_remove)
    
    to_remove = []
    for col in ds2.column_names:
        if col not in ds1.column_names:
            to_remove.append(col)
            
    ds2 = ds2.remove_columns(to_remove)
    
    # Concatenate the two datasets
    ds = concatenate_datasets([ds1, ds2])
    print("Length of concatenated dataset: ", len(ds))
    
    # Print the row with the longest prompt length
    # max_length = 0
    # max_row = None
    # for row in ds:
    #     length = len(row["question"]) + len(row["background"]) + len(row["resolution_criteria"])
    #     if length > max_length:
    #         max_length = length
    #         max_row = row
    
    # print("Max row: ", max_row)
    # print("Max length: ", max_length)
    
    # Remove all rows where prompt length (question + background + resolution criteria) is more than 20000 characters
    ds = ds.filter(lambda x: len(x["question"]) + len(x["background"]) + len(x["resolution_criteria"]) <= 20000)
    print("Length of dataset after removing rows with prompt length > 20000: ", len(ds))
    
    # print("Length of dataset after removing duplicates: ", len(ds))
    return ds
    
        

def load_metaculus_data(split="train", nr_forecasters=1):
    path = "nikhilchandak/metaculus-binary"
    ds = load_dataset(path)["train"]

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

if __name__ == "__main__":
    # ds = load_halawi_data(split="train", raw=True)

    # Print all the unique values in the column "resolution"
    
    # uniques = np.unique(ds["resolution"])
    # print("Num uniques: ", len(uniques))
    # pprint(uniques)
    
    # print("Length of dataset: ", len(ds))
    # # For each representative value in uniques, print a few rows with that value
    # for unique in uniques:
    #     filtered = ds.filter(lambda x: x["resolution"] == unique)
    #     if len(filtered) > 10:
    #         # Randomly select 10 rows 
    #         filtered = filtered.shuffle()
    #         filtered = filtered.select(range(10))
            
    #     # Pretty print the question, background, is_resolved for each row (explicitly) of the filtered dataset
    #     for row in filtered:
    #         print("Question: ", row["question"])
    #         # print("Background: ", row["background"])
    #         print("Is Resolved: ", row["is_resolved"])
    #         print("Resolution: ", row["resolution"])
    #         # Print date_begin, date_resolve_at, date_close
    #         print("Date Begin: ", row["date_begin"])
    #         print("Date Resolve At: ", row["date_resolve_at"])
    #         print("Date Close: ", row["date_close"])
            
    #         print("\n\n")
    #     # print("\n\n\n\n")
        
    ds = load_manifold_and_metaculus_data(split="train", raw=True)
    
    
    
    
    