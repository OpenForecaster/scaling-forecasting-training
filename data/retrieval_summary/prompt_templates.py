"""
This module contains various prompt templates for summarizing news articles.
Each template function takes article content and potentially forecasting question details
and returns a prompt string to be used for summarization.
"""

def basic_summary(article, question_title, background, target_length=100):
    """
    Basic summarization prompt without any forecasting context.
    
    Args:
        article (str): The article text to summarize
        question_title (str): The forecasting question title
        background (str): Background information about the forecasting question
        target_length (int): Target length of the summary in words
        
    Returns:
        str: Formatted prompt for summarization
    """
    return f"""Please summarize the following article in NO MORE than {target_length} words:

Article: {article}

You should ONLY output a concise and informative summary of AT MOST {target_length} words and nothing else.
"""

def forecast_focused_summary(article, question_title, background, target_length=100):
    """
    Summarization prompt that includes forecasting question to focus the summary.
    
    Args:
        article (str): The article text to summarize
        question_title (str): The forecasting question title
        background (str): Background information about the forecasting question
        target_length (int): Target length of the summary in words
        
    Returns:
        str: Formatted prompt for summarization
    """
    return f"""Summarize the following article in NO MORE than {target_length} words.

Article: {article}

When creating this summary, focus on information that would be helpful for answering this forecasting question:
Question: {question_title}
Question Background: {background}

Provide a concise and informative summary relevant to this forecasting question. You should ONLY output a summary of AT MOST {target_length} words and nothing else.
"""

def key_facts_summary(article, question_title, background, target_length=100):
    """
    Summarization prompt focused on extracting key facts only.
    
    Args:
        article (str): The article text to summarize
        question_title (str): The forecasting question title
        background (str): Background information about the forecasting question
        target_length (int): Target length of the summary in words
        
    Returns:
        str: Formatted prompt for summarization
    """
    return f"""Extract and list the most important facts from the following article in NO MORE than {target_length} words:

Article: {article}

You should ONLY output the key factual information in a clear, concise manner in AT MOST {target_length} words. Do not include any other text.
"""

def forecast_evidence_summary(article, question_title, background, target_length=100):
    """
    Summarization prompt focused specifically on evidence related to the forecasting question.
    
    Args:
        article (str): The article text to summarize
        question_title (str): The forecasting question title
        background (str): Background information about the forecasting question
        target_length (int): Target length of the summary in words
        
    Returns:
        str: Formatted prompt for summarization
    """
    return f"""Analyze the following article and summarize in {target_length} words or less. Output ONLY the information that provides evidence relevant to this forecasting question:

Forecasting Question: {question_title}
Question Background: {background}
Article: {article}

Focus exclusively on evidence from the article that could help answer the forecasting question.

You should ONLY output a summary of AT MOST {target_length} words and nothing else.
"""

def timeline_oriented_summary(article, question_title, background, target_length=100):
    """
    Summarization prompt that emphasizes chronology and timeline of events.
    
    Args:
        article (str): The article text to summarize
        question_title (str): The forecasting question title
        background (str): Background information about the forecasting question
        target_length (int): Target length of the summary in words
        
    Returns:
        str: Formatted prompt for summarization
    """
    return f"""Create a chronological summary of the events described in this article in NO MORE than {target_length} words.

Article: {article}

Focus on the timeline of events, key dates, and the sequence of developments. 

You should ONLY output a summary of AT MOST {target_length} words and nothing else.
"""


def halawi(article, question_title, background, target_length=100):
    """
    Halawi's summarization prompt that focuses on preserving details relevant to forecasting.
    
    Args:
        article (str): The article text to summarize
        question_title (str): The forecasting question title
        background (str): Background information about the forecasting question
        target_length (int): Target length of the summary in words
        
    Returns:
        str: Formatted prompt for summarization
    """
    return f"""I want you to make the following article shorter (condense it to NO MORE than {target_length} words).
Article: {article}

When doing this task for me, please do not remove any details that would be helpful for making considerations about the following forecasting question.

Forecasting Question: {question_title}
Question Background: {background}

You should ONLY output a summary of AT MOST {target_length} words and nothing else.
"""

def create_forecast_summarization_prompt(article: str, question_title: str, background: str, target_length: int = 100) -> str:
    prompt = f"""You are an expert analyst helping forecast future events. Your task is to read the following news article, extract information that could help with forecasting the answer to the following question, and the SUMMARIZE THE ARTICLE BASED ON THE CRITERIA BELOW.

Forecasting Question: "{question_title}"
Question Background: "{background}"

First, think carefully about the forecasting question, what underlying uncertainties do you have about its answer? What factors would influence the outcomes?

Then, from the article that follows, focus on identifying:
- Key facts, trends, or data points related to the question
- New knowledge that you did not know before which could influence the outcome of the question
- Statements by experts, officials, or stakeholders that indicate future intentions or expectations
- Quantitative data (e.g., economic indicators, timelines, probabilities) and any forward-looking projections
- Underlying drivers, risks, or conditions that could influence the outcome
- Clever ways of reasoning or breaking down the problem that could help with the forecasting question.

Article:
\"\"\"
{article.strip()}
\"\"\"

YOUR JOB IS TO SUMMARIZE THE ARTICLE BASED ON THE ABOVE CRITERIA IN AT MOST {target_length} WORDS (CAN BE LESS). Focus on the information potentially relevant to the forecasting question AS DESCRIBED ABOVE. Avoid general background, unrelated details, or redundant information.

OUTPUT YOUR FINAL SUMMARY UNDER {target_length} WORDS INSIDE <summary> </summary> tags. BE DIRECT AND TO THE POINT. DO NOT INCLUDE YOUR OWN OPINION OR ANY META COMMENTARY, DISCUSSION, OR EXPLANATION IN YOUR FINAL SUMMARY.
"""
    return prompt

# Get all prompt functions in this module
def get_all_prompt_functions():
    """Returns a dictionary of all prompt functions defined in this module"""
    return {
        "basic_summary": basic_summary,
        "forecast_focused_summary": forecast_focused_summary,
        "key_facts_summary": key_facts_summary,
        "forecast_evidence_summary": forecast_evidence_summary,
        "timeline_oriented_summary": timeline_oriented_summary,
        "halawi": halawi,
        "create_forecast_summarization_prompt": create_forecast_summarization_prompt
    }

# Target length variants
TARGET_LENGTHS = [50, 100, 200] 