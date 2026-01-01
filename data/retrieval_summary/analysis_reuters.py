import datasets


def load_retreived_data(split="train", data_type="retrieval_metaculus", nr_forecasters=1):
    prefix = "/fast/sgoel/forecasting/news/retrieval/"
    path = prefix + data_type + "/"
    # path = "/fast/sgoel/forecasting/news/retrieval/metaculus-binary_apnews_7_365/"
    path = "/fast/sgoel/forecasting/news/retrieval/metaculus-binary_reuters_7_365/"
    
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
    # ds = ds.map(lambda row: {"prompt": create_prompt_for_row(row)})
    # ds = ds.select(range(5,6))
    
    # pretty print the prompt iteratively 
    # for i, row in enumerate(ds):
    #     print(row["prompt"])
    #     print("-"*100)
    
    return ds


def halawi_100(row):
    question = row["question"]
    background = row["background"]
    retrieved_articles = row["retrieved_articles"]
    article = retrieved_articles[0]["maintext"]
    return f"""I want to make the following article shorter (condense it to no more than 100 words).
Article: {article}
When doing this task for me, please do not remove any details that would be helpful for making
considerations about the following forecasting question.
Forecasting Question: {question}
Question Background: {background}
"""

def main():
    ds = load_retreived_data(split="train", data_type="retrieval_metaculus", nr_forecasters=1)
    # print(ds[:10])
    max_len = max([len(row["retrieved_articles"]) for row in ds])
    print("Max length of retrieved articles: ", max_len)
    ds2 = ds.select(range(10))
    for i, row in enumerate(ds2):
        print(row["question"])
        # print(row["tokenized_query"])
        print("Length of retrieved articles: ", len(row["retrieved_articles"]))
        for i, article in enumerate(row["retrieved_articles"]):
        #     print(f"Article {i+1}:", article.keys())
            print("Length of maintext: ", len(article["maintext"]))
        #     for key in article.keys():
        #         if key != "maintext":
        #             print(key, article[key])
        #     # print(article["summary"])
        #     print("-"*100)

if __name__ == "__main__":
    main()
