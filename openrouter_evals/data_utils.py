from datasets import load_dataset
import numpy as np 
from datasets import Dataset
from datasets import load_from_disk
import datetime

def load_halawi_data(split="train", raw=False):
    path = "YuehHanChen/forecasting"
    if raw:
        path += "_raw"
        
    ds = load_dataset(path)[split]
    # print value counts of the resolution column (list) use unique
    print(np.unique(ds["resolution"], return_counts=True))
    # print(ds.column_names)
    if raw:
        
        # Only keep rows with question_type == BINARY or binary
        ds = ds.filter(lambda x: x["question_type"].lower() == "binary")
        
        # Only keep rows with resolution == 1 or 1.0 or 0 or 0.0 (in str)
        # ds = ds.filter(lambda x: x["resolution"] in ["1.0", "0.0"]) 
        ds = ds.filter(lambda x: x["resolution"] in ["1", "1.0", "0", "0.0"])
    
    return ds

def filter_halawi_data(ds, begin_date="2023-01-01", end_date="2023-06-01"):
    useful_subset = ds.filter(lambda x: x["date_begin"] > begin_date and x["date_resolve_at"] < end_date)
    return useful_subset

def load_menge_data(split="validation", data_type="binary"):
    path = "/fast/nchandak/forecasting/datasets/menge/" + data_type + "_" + split + ".json"
        
    # Load dataset
    ds = Dataset.from_json(path)
    
    # Print column names 
    print("Menge Column names: ", ds.column_names)
    
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
    ds = Dataset.from_json(path)
    
    # Print column names 
    print("Manifold Split: ", split)
    print("Column names: ", ds.column_names)
    
    # Apply same filtering as metaculus data
    # if split == "train":
    #     ds = ds.filter(lambda x: x["date_resolve_at"] < "2024-06-30")
    
    # if split == "test":
    #     ds = ds.filter(lambda x: x["date_begin"] >= "2024-05-30")
        # ds = ds.filter(lambda x: x["nr_forecasters"] >= nr_forecasters)
        
    return ds


def add_idx_column(dataset: Dataset) -> Dataset:
    """
    Adds an 'idx' column to the dataset, storing the original row index.
    """
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)


def load_mcq_manifold_data(split="train", data_type="zeroshot", nr_forecasters=1, volume=4000):
    path = "/fast/nchandak/forecasting/datasets/manifold/mcq_raw_train.json"
    if split == "test":
        path = "/fast/nchandak/forecasting/datasets/manifold/mcq_test.json"
        
    ds = Dataset.from_json(path)
    
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
    func = ask_just_probabilities if data_type == "zeroshot" else ask_probabilities
    if data_type == "prior":
        func = detailed_mcq_prompt
    elif data_type == "thinkbetter":
        func = mcq_prompt_claude_style
        
    print(f"Using {data_type} prompt")
    # ds = ds.map(lambda x: {"full_prompt": ask_probabilities(x['prompt'])})
    # ds = ds.map(lambda x: {"full_prompt": ask_just_probabilities(x['prompt'])})
    # ds = ds.map(lambda x: {"full_prompt": detailed_mcq_prompt(x['prompt'])})
    ds = ds.map(lambda x: {"full_prompt": func(x['prompt'])})
    
    # Filter nr_forecasters column
    ds = ds.filter(lambda x: x["nr_forecasters"] >= nr_forecasters)
    
    # Filter volume column
    ds = ds.filter(lambda x: x["volume"] >= volume)
    
    return ds

def load_mcq_metaculus_data(split="test", nr_forecasters=1):
    path = "/fast/nchandak/forecasting/datasets/metaculus/mcq_raw.json"
    ds = Dataset.from_json(path)
    
    # Add idx column
    ds = add_idx_column(ds)
    
    # print column names
    print("Column names: ", ds.column_names)
    
    # Make full prompt
    # ds = ds.map(lambda x: {"full_prompt": extend_mcq_prompt(x['prompt'])})
    ds = ds.map(lambda x: {"full_prompt": detailed_mcq_prompt(x['prompt'])})
    
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

def load_infinitegames_data(split="train", nr_forecasters=1):
    if "balanced" in split:
        path = "/fast/nchandak/forecasting/datasets/infinitegames/binary_balanced_test.json"
    else:
        path = "/fast/nchandak/forecasting/datasets/infinitegames/binary_test.json"
    
    ds = Dataset.from_json(path)
    
    print("Column names: ", ds.column_names)
    
    # resolution value counts
    print(np.unique(ds["resolution"], return_counts=True))
        
    return ds


def load_retreived_data(split="train", save_json=False):
    path = "/fast/sgoel/forecasting/news/retrieval/metaculus-binary_reuters_7_365/"
    
    # Load the entire dataset
    dataset = load_from_disk(path)
    
    # If the dataset has splits and a specific split is requested
    if hasattr(dataset, 'keys') and split in dataset:
        dataset = dataset[split]
    
    # Save dataset as JSON if requested
    if save_json:
        json_path = path.rstrip('/') + '.json'
        print(f"Saving dataset to {json_path}")
        dataset.to_json(json_path)
        print(f"Dataset saved to {json_path}")
    
    
    # Filter columns for which retrieved_articles is empty
    dataset = dataset.filter(lambda x: x["retrieved_articles"] != [])
    
    # print column names
    print("Column names: ", dataset.column_names)
    # print first 10 rows
    # print(dataset[:10])
    # print length of the dataset
    print(len(dataset))
    
    return dataset

def format_prompt(question: str, background: str, resolution_criteria: str, date_close: str) -> str:
    return f"""I will ask you a forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened.

Question: {question}
Resolution Criteria: {resolution_criteria}

Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
"""


def extend_mcq_prompt123(content: str) -> str:
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


def ask_just_probabilities(content: str) -> str:
    return f"""You will be asked a forecasting question in multiple choice format. You should report ONLY your probability for each of the answer choices in <answer> </answer> tags in the following format:

<answer> <A> 0.a </A> <B> 0.b </B> <C> 0.c </C> <D> 0.d> </D> </answer>

IMPORTANT:
- <answer> MUST contain the probabilities of each option inside XML tags.
- Each probability inside option XML tags MUST be a decimal between 0 and 1 representing your confidence in that option choice and they should sum to 1 across all options.

{content}
"""

def detailed_mcq_prompt(content: str, steps = 10) -> str:
    return f"""You will be asked a forecasting question in multiple choice format. You have to choose the most likely option from the given options and also report your confidence level in your answer.

You have to respond in a sequence of <think> </think> tags where you reason, and <prior_n> </prior_n> tags where you report your intermediate priors (over the options) in the following format. Inside prior tags, report your confidence over each of the answer choices in the below format, for example <prior_1> <A> 0.2 </A> <B> 0.3 </B> <C> 0.1 </C> <D> 0.4> </D> </prior_1>. The probability reported should be a decimal between 0 and 1 for each option, and sum to 1 across all options. Do not insert anything except this probability inside the option tags. Also, do not add anything except the options (with their respective probabilities) inside the prior tags.

As you think about the question, keep reporting intermediate priors based on your thought process in <prior_n> </prior_n> where n is the current count of the prior tag in your response. In the end, report your final probabilities in <prior_final> </prior_final> tags. You should update your prior over at least {steps} steps of <think> <prior> reasoning.

IMPORTANT:
- Consider all possible factors that might influence the outcome of the question.
- In each <think> tag, you MUST reason step by step about how you will update your answer or probability for the options in the subsequent <prior_n+1> tag. Think thoroughly and deeply about each of the options. 
- Each <prior> MUST contain the probabilities of each option inside XML tags.
- Each probability inside option XML tags MUST be a decimal between 0 and 1 representing your confidence in that option choice.
- You should update your prior at least {steps} times and report it.
- You should give super detailed reasoning for each step inside the <think> tag.

So the final format looks like:
<think> </think>
<prior_1> <A> 0.a </A> <B> 0.b </B> <C> 0.c </C> <D> 0.d> </D> </prior_1>
<think> </think>
<prior_2> <A> 0.a </A> <B> 0.b </B> <C> 0.c </C> <D> 0.d> </D> </prior_2>
....
<think> </think>
<prior_final> <A> 0.a </A> <B> 0.b </B> <C> 0.c </C> <D> 0.d> </D> </prior_final>

{content}
"""

def load_paleka_data(split="spanned"):
    path = "/fast/nchandak/forecasting/datasets/paleka/20240701_20240831"
    if split == "spanned" or "gpt4o" in split:
        path += "_gpt-4o_spanned_resolved"
        
    path += ".jsonl"
        
    ds = Dataset.from_json(path)
    
    # print column names
    print("Paleka Column names: ", ds.column_names)
    
    print(f"Length of dataset: {len(ds)}")
    
    # Resolution value counts
    print("Resolution value counts: ", np.unique(ds["resolution"], return_counts=True))
    print("Resolution date value counts: ", np.unique(ds["resolution_date"], return_counts=True))
    # Transform resolution column from True/False to 1/0
    # ds = ds.map(lambda x: {"resolution": 1 if x["resolution"] == True else 0})
    
    # print first 10 rows
    # for i, row in enumerate(ds):
    #     print(row["title"])
    #     print(row["body"])
    #     print(row["resolution_date"])
    #     print(row["resolution"])
    #     print("-"*100)
    #     if i > 5:
    #         break
    
    # Add prompt column
    ds = ds.map(lambda x: {"prompt": format_prompt(question=x["title"], background="", resolution_criteria=x["body"], date_close=x["resolution_date"])})
    
    return ds 

    # pretty print the prompt iteratively 
    for i, row in enumerate(ds):
        print(row["prompt"])
        print("-"*100)
        if i > 5:
            break
    
    # Create a train subset for questions with resolution_date before 2024-08-01
    # create datetime object for 2024-08-01
    date_2024_08_01 = datetime.datetime(2024, 8, 1)
    train = ds.filter(lambda x: x["resolution_date"] < date_2024_08_01)
    test = ds.filter(lambda x: x["resolution_date"] >= date_2024_08_01)
    
    # print length of the train and test subsets
    print(f"Length of train subset: {len(train)}")
    print(f"Length of test subset: {len(test)}")
    
    # Resolution value counts
    print("Train Resolution value counts: ", np.unique(train["resolution"], return_counts=True))
    print("Test Resolution value counts: ", np.unique(test["resolution"], return_counts=True))
    
    return train, test

def mcq_prompt_claude_style(content: str, steps = 10) -> str:
    
    return f"""You will be asked a forecasting question in multiple choice format. You have to choose the most likely option from the given options and also report your confidence level in your answer.

You have to respond in a sequence of <think> </think> tags where you reason, and <prior_n> </prior_n> tags where you report your intermediate priors (over the options) in the following format. Inside prior tags, report your confidence over each of the answer choices in the below format, for example <prior_1> <A> 0.2 </A> <B> 0.3 </B> <C> 0.1 </C> <D> 0.4> </D> </prior_1>. The probability reported should be a decimal between 0 and 1 for each option, and sum to 1 across all options. Do not insert anything except this probability inside the option tags. Also, do not add anything except the options (with their respective probabilities) inside the prior tags.

As you think about the question, keep reporting intermediate priors based on your thought process in <prior_n> </prior_n> where n is the current count of the prior tag in your response. In the end, report your final probabilities in <prior_final> </prior_final> tags. You should update your prior over at least {steps} steps of <think> <prior> reasoning.

So the final format looks like:
<think> </think>
<prior_1> <A> 0.a </A> <B> 0.b </B> <C> 0.c </C> <D> 0.d> </D> </prior_1>
<think> </think>
<prior_2> <A> 0.a </A> <B> 0.b </B> <C> 0.c </C> <D> 0.d> </D> </prior_2>
....
<think> </think>
<prior_final> <A> 0.a </A> <B> 0.b </B> <C> 0.c </C> <D> 0.d> </D> </prior_final>

Here is an example question and response:

Question: What will Microsoft's stock price be at the end of March?\nA. 390.00 - 399.99\nB. 380.00 - 389.99\nC. 410.00 - 419.99\nD. 420.00 - 429.99

Background: Resolves to MSFT\u2019s closing price on \nGoogle Finance\n on the last trading day of the month\nCheck out the dashboard for more \nTony's Stock Markets

Question close date: 2024-03-28

Response: <think>\nI need to forecast Microsoft's stock price at the end of March 2024. Let me start by establishing a baseline. I should consider:\n1. Current price of Microsoft stock\n2. Recent trends in the stock price\n3. Market conditions and economic factors\n4. Microsoft-specific news or events that might impact the stock\n\nFirst, I need to establish what the current price is (as of when this question is being asked). Without real-time data, I'll have to reason based on general knowledge. Microsoft stock has been performing well in recent years, especially with their AI initiatives and cloud business growth. Let me start with a relatively uniform prior across the options, with slightly higher weight to options near the current price range.\n</think>\n\n<prior_1>\n<A> 0.25 </A>\n<B> 0.25 </B>\n<C> 0.25 </C>\n<D> 0.25 </D>\n</prior_1>\n

<think>\nI need to refine my estimate based on what I know about Microsoft's recent stock performance. Microsoft has been performing strongly due to its cloud business (Azure) and AI investments. The company has been showing consistent growth. \n\nWithout knowing the exact current price, I'll reason about the likely movement. Tech stocks have generally been performing well in 2023 and early 2024, with AI being a major driver. Microsoft's partnership with OpenAI and integration of AI into their products has been viewed positively by the market.\n\nGiven these factors, I think there's a slightly higher chance of the stock price being in the higher ranges by the end of March. Let me adjust my probabilities accordingly.\n</think>\n\n<prior_2>\n<A> 0.23 </A>\n<B> 0.22 </B>\n<C> 0.28 </C>\n<D> 0.27 </D>\n</prior_2>\n

<think>\nI should consider the time horizon. The question asks about the end of March, which is likely a few weeks or a month away from when the question was posed. In this timeframe, significant price movements are possible but not guaranteed.\n\nMicrosoft's stock, like most large tech companies, tends to move with broader market trends but also responds to company-specific news. Let's consider potential events that might occur before the end of March:\n- Earnings reports (though Microsoft's next earnings might be after March)\n- Federal Reserve decisions on interest rates\n- General market sentiment toward tech stocks\n- Any major product announcements or strategic shifts\n\nGiven the relatively short time horizon, dramatic price changes would require significant news or market shifts. Let me adjust my probabilities based on this reasoning.\n</think>\n\n<prior_3>\n<A> 0.26 </A>\n<B> 0.24 </B>\n<C> 0.27 </C>\n<D> 0.23 </D>\n</prior_3>\n

<think>\nI should try to narrow down the current price range more precisely to make a better forecast. Microsoft has been one of the largest companies by market cap, often trading above $350 per share in recent times.\n\nIf the current price is already in one of these ranges, that range becomes more likely as the end point, since stock prices follow a random walk in the short term with a slight upward drift for growing companies.\n\nLet me assume the current price is somewhere in the $380-400 range based on recent performance. If that's the case, options A and B become more likely as they represent smaller deviations from the current price. Option C represents moderate growth, and option D represents stronger growth in a relatively short period.\n\nI'll adjust my probabilities with this in mind, giving more weight to options A and B.\n</think>\n\n<prior_4>\n<A> 0.30 </A>\n<B> 0.28 </B>\n<C> 0.25 </C>\n<D> 0.17 </D>\n</prior_4>\n

<think>\nI should also consider the broader market conditions. The stock market has been performing well recently, with many tech stocks reaching new highs. However, there are also concerns about inflation, interest rates, and potential market corrections.\n\nThe Federal Reserve's decisions on interest rates could impact the market before the end of March. Higher interest rates typically put pressure on growth stocks like Microsoft, while lower rates or signals of future cuts could boost the stock.\n\nGiven the recent strong performance of tech stocks and Microsoft specifically, there might be some resistance to further significant gains in the short term. This would make the higher price ranges (C and D) somewhat less likely.\n\nLet me adjust my probabilities based on these market considerations.\n</think>\n\n<prior_5>\n<A> 0.32 </A>\n<B> 0.30 </B>\n<C> 0.23 </C>\n<D> 0.15 </D>\n</prior_5>\n

<think>\nLet me consider Microsoft-specific factors that could influence the stock price by the end of March:\n\n1. AI integration: Microsoft continues to integrate OpenAI's technology into its products, which has been viewed positively by investors.\n2. Cloud business (Azure): Growth in this segment has been a key driver of Microsoft's stock performance.\n3. Competition: Increased competition in the cloud and AI space could impact investor sentiment.\n4. Regulatory concerns: Potential antitrust or privacy regulations could affect Microsoft's outlook.\n\nGiven Microsoft's strong position in AI and cloud computing, these factors generally support a positive outlook. However, the stock price may already reflect much of this optimism.\n\nI'll adjust my probabilities slightly to reflect these Microsoft-specific considerations.\n</think>\n\n<prior_6>\n<A> 0.33 </A>\n<B> 0.29 </B>\n<C> 0.24 </C>\n<D> 0.14 </D>\n</prior_6>\n

<think>\nI should also consider the historical volatility of Microsoft's stock. As a large, established company, Microsoft typically doesn't experience extreme price swings in short periods unless there's significant news.\n\nThe ranges provided in the options represent approximately 2.5% differences between each option. This level of movement is certainly possible within a month for Microsoft stock, but larger movements (to options C and D) would likely require positive catalysts.\n\nGiven the relatively stable nature of Microsoft as a mature tech company, I should probably give more weight to options that represent smaller deviations from what I believe is the current price range.\n\nLet me adjust my probabilities based on this consideration of volatility.\n</think>\n\n<prior_7>\n<A> 0.35 </A>\n<B> 0.30 </B>\n<C> 0.22 </C>\n<D> 0.13 </D>\n</prior_7>\n

<think>\nI should also consider seasonal patterns or end-of-quarter effects that might influence the stock price at the end of March.\n\nMarch marks the end of the first quarter, and there can sometimes be portfolio rebalancing by institutional investors that affects large-cap stocks like Microsoft. However, these effects are often unpredictable and may not significantly impact the price in a consistent direction.\n\nMicrosoft might provide updates or guidance near the end of the quarter, which could move the stock price. However, their formal earnings report for Q1 would likely come after March.\n\nGiven these considerations, I don't see strong reasons to significantly adjust my probabilities based on end-of-quarter effects, but I'll make slight adjustments to reflect the additional uncertainty this introduces.\n</think>\n\n<prior_8>\n<A> 0.34 </A>\n<B> 0.31 </B>\n<C> 0.22 </C>\n<D> 0.13 </D>\n</prior_8>\n

<think>\nLet me consider the time remaining until the end of March. The question close date is March 28, 2024, which suggests this is when the forecast will be evaluated.\n\nWith potentially several weeks of trading between when the question was posed and the end date, there's time for meaningful price movements. The longer the time horizon, the more opportunity for drift toward the higher or lower price ranges.\n\nGiven that tech stocks have generally been on an upward trajectory, especially those involved in AI, there's a reasonable chance that Microsoft could move toward the higher ranges (C or D) if this trend continues. However, the baseline probability should still favor smaller movements (options A and B).\n\nI'll adjust my probabilities to slightly increase the chances of the higher ranges given the potential time for upward movement.\n</think>\n\n<prior_9>\n<A> 0.33 </A>\n<B> 0.30 </B>\n<C> 0.23 </C>\n<D> 0.14 </D>\n</prior_9>\n

<think>\nI should also consider the specific price ranges more carefully. The options are:\nA. 390.00 - 399.99\nB. 380.00 - 389.99\nC. 410.00 - 419.99\nD. 420.00 - 429.99\n\nIf Microsoft's current stock price is already in range A or B, then these become more likely as end points simply due to the anchoring effect and the random walk nature of stock prices in the short term.\n\nWithout knowing the exact current price, I'm working with limited information. However, based on Microsoft's strong performance and the general upward trend in tech stocks, I believe the current price is likely in the range of options A or B.\n\nGiven this assumption, and considering all the factors I've discussed (AI growth, cloud business strength, market conditions, time horizon, etc.), I'll make a final adjustment to my probabilities.\n</think>\n\n<prior_10>\n<A> 0.35 </A>\n<B> 0.30 </B>\n<C> 0.22 </C>\n<D> 0.13 </D>\n</prior_10>\n

<think>\nFor my final assessment, I'll integrate all the factors I've considered:\n\n1. Microsoft's strong position in AI and cloud computing supports a positive outlook.\n2. The relatively short time horizon (until end of March) limits the probability of large price movements.\n3. The stock likely follows a random walk with a slight upward drift.\n4. Market conditions and interest rate decisions could impact the stock before the end of March.\n5. As a large, established company, Microsoft typically has lower volatility than smaller tech companies.\n\nGiven all these considerations, I believe the most likely outcome is that Microsoft's stock will be in the 390.00 - 399.99 range (option A) at the end of March, followed by the 380.00 - 389.99 range (option B). The higher ranges (C and D) are less likely but still possible, especially if there's positive news or a strong market rally.\n\nMy final probabilities reflect this assessment.\n</think>\n\n<prior_final>\n<A> 0.36 </A>\n<B> 0.31 </B>\n<C> 0.21 </C>\n<D> 0.12 </D>\n</prior_final>


IMPORTANT:
- Consider all possible factors that might influence the outcome of the question.
- In each <think> tag, you MUST reason step by step about how you will update your answer or probability for the options in the subsequent <prior_n+1> tag. Think thoroughly and deeply about each of the options. 
- Each <prior> MUST contain the probabilities of each option inside XML tags.
- Each probability inside option XML tags MUST be a decimal between 0 and 1 representing your confidence in that option choice.
- You should update your prior at least {steps} times and report it.
- You should give super detailed reasoning for each step inside the <think> tag.

Actual question follows below:

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
    # print(len(ds["question"]))
    
    # ds = load_mcq_manifold_data(volume=4000)
    # print(len(ds))
    # # print(ds.column_names)
    # for i, row in enumerate(ds):
    #     print(row["full_prompt"])
    #     print("-"*100)
    #     if i > 5:
    #         break
    
    # ds = load_mcq_metaculus_data(nr_forecasters=10)
    ds = load_mcq_manifold_data(volume=4000)
    print("Length of dataset: ", len(ds))
    print(ds.column_names)
    for i, row in enumerate(ds):
        print(row["full_prompt"])
        print(row["idx"])
        
        print("-"*100)
        if i > 5:
            break