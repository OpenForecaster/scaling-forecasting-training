"""
Forecasting Question Generator - Core module for generating questions from news articles.

This module provides the main ForecastingQuestionGenerator class, which orchestrates
a complete pipeline for creating forecasting questions:

1. **Question Generation**: Generates multiple-choice or free-form questions from articles
2. **Individual Question Extraction**: Parses generated questions into separate entries
3. **Question Validation**: Validates each question for quality and correctness
4. **Best Question Selection**: Chooses the highest-quality question from candidates
5. **Leakage Checking**: Detects and removes answer leakage from questions

The generator supports both MCQ (multiple-choice) and free-form short-answer questions,
with extensive prompt engineering to ensure high-quality, forecastable questions.

Key Features:
- Async processing for efficient batch operations
- Incremental saving to prevent data loss
- Comprehensive validation and quality checks
- Support for multiple inference engines (VLLM, OpenRouter, etc.)
- Extensive logging for debugging and monitoring

Example Usage:
    ```python
    from qgen.qgen_core.question_generator import ForecastingQuestionGenerator
    from qgen.inference.openrouter_inference import OpenRouterInference
    
    # Initialize inference engine
    engine = OpenRouterInference(model="deepseek/deepseek-chat-v3-0324")
    
    # Initialize generator for free-form questions
    generator = ForecastingQuestionGenerator(
        inference_engine=engine,
        use_freeq=True,
        num_questions=3,
        check_leakage=True,
        choose_best=True,
        validate_questions=True
    )
    
    # Run complete pipeline
    results = await generator.run_pipeline(
        articles=articles,
        output_path="generated_questions.jsonl",
        batch_size=5
    )
    ```

Author: Forecasting Team
"""

import os
import json
import logging
from typing import List, Dict
import time # Added for potential delays

logger = logging.getLogger(__name__)

class ForecastingQuestionGenerator:
    def __init__(
        self, 
        inference_engine,
        use_freeq: bool = False,
        num_questions: int = 3,
        check_leakage: bool = False,
        leakage_engine = None,
        choose_best: bool = False,
        choose_engine = None,
        validate_questions: bool = False,
    ):
        """
        Initialize the forecasting question generator.
        
        Args:
            inference_engine: Engine for text generation (must implement BaseInference)
            use_freeq: If True, generate free-form short answer questions instead of MCQs
        """
        self.inference_engine = inference_engine
        self.use_freeq = use_freeq
        self.num_questions_per_article = num_questions
        self.check_leakage = check_leakage
        self.leakage_engine = leakage_engine
        self.choose_best = choose_best
        self.choose_engine = choose_engine
        self.validate_questions = validate_questions
        if not self.leakage_engine:
            self.leakage_engine = self.inference_engine
        
    def format_prompt(self, article: Dict) -> str:
        """
        Format the prompt for generating forecasting questions.
        
        Args:
            article: Article dictionary containing the content
            
        Returns:
            Formatted prompt for the LLM
        """
        if self.use_freeq:
            return self._format_freeq_prompt(article)
        else:
            return self._format_mcq_prompt(article)
    
    def _format_mcq_prompt(self, article: Dict) -> str:
        """Format prompt for multiple choice questions (original format)."""
        source_article = f"Title: {article.get('title', '')}\n\n"
        
        if 'description' in article and article['description']:
            source_article += f"Description: {article['description']}\n\n"
            
        if 'maintext' in article and article['maintext']:
            source_article += f"Content: {article['maintext']}\n\n"
            
        if 'date_publish' in article and article['date_publish']:
            source_article += f"Published Date: {article['date_publish']}\n"
        
        prompt = f"""
**Task:** Based on the provided news article, generate **5 high quality** forecasting questions which are multiple-choice format (MCQs) with 4 options each, as JSONs.
Forecasting questions are about predicting future events. Here, the predictor will have a knowledge cutoff before the article is published and no access to the article, so a forecasting question has to be posed about information explicitly stated in the article.
The correct answer should be specified as the index of the option in the options list. The JSON format should be: 
question_title: str,  background: str, options: List[str], answer: int

**Example Format**:
{{
    "question_id": "0",
    "question_title": "Who will win the nobel prize in Literature in 2016?",
    "background": "The nobel prize in literature is awarded to authors for their outstanding contributions to literature. The prize is awarded annually by the swedish academy.",
    "options": ["Thomas Pynchon", "Bob Dylan", "Haruki Murakami", "Cormac McCarthy"],
    "answer": 1
}}

Each question must follow the structured guidelines below.

### **Guidelines for Creating Multiple-Choice Forecasting Questions**

**Title Guidelines**
- **MCQ not Binary**: The question should not be a binary yes / no question, that is, do not ask questions starting with "Will". It should be in MCQ format with 4 options. 
- **Answerable based on article**: Each question must have a definitive answer based on information explicitly stated in the article. The other 3 options must surely be incorrect, again based on information in the article.
- **Not about historical knowledge**: The question should not be about recall of facts or events known before the article publish date. 
- **Direct and Precise**: Titles must be straightforward and unambiguous, avoiding vague terms. It should be in future tense, not past or perfect. 
- **Resolution Criteria**: Include resolution criteria in the question, for example resolution dates such as "by {{month_name}}, {{year}}?" or "in {{month_name}}, {{year}}?", and source of resolution such as "based on {{news source}}", "as said by {{official name}}", etc.
- **No references to article or future information**: Do not refer to the specific article, such as by saying "in the article". The forecaster does not have access to it or any information beyond the article publish date.

**MCQ Format**
- **Faithfulness to Article**: The answer should be based on information explicitly stated in the article, and not implications or your own knowledge.
- **Overspecificity**: The question should not be about the exact amount of something, which is often difficult to predict. Instead it should be about what happened, or predicting ranges. 
- **Four Options**: Provide four distinct options with exactly one being the correct prediction. The remaining three must be incorrect.
- **Option Overlap**: The options should represent disjoint outcomes. Do not include redundant options.
- **Concise**: The options should be as concise as possible while being clear and unambiguous.

**Background Guidelines**
- **Should not help answer**: The background must not directly help answer the forecasting question. Do not include any knowledge from the article or elsewhere that helps eliminate any of the options.
- **Necessary Context**: Only include information necessary to understand the question.
- **No Additional Knowledge**: Do not add any knowledge beyond the provided article.

**Diversity Requirements**
- **Variety in Question Types**: Generate questions covering different aspects of the article's content.
- **Different Domains**: If the article covers multiple topics, create questions across different areas.
- **Varied Timeframes**: Include questions with different resolution timeframes when possible.
- **Distinct Focuses**: Each question should focus on a different key piece of information from the article.

Generate 5 high-quality, diverse multiple-choice forecasting questions based on the provided article with the question id as "0", "1", "2", "3", and "4" with each question as a separate JSON object wrapped in ```json and ```. Do not include any analysis, ranking, or additional commentary.

Article:
{source_article}

**Required Output Format**:
```json
{
    "question_id": "0",
    "question_title": "[Question 1]",
    "background": "[Background 1]",
    "options": ["[Option A]", "[Option B]", "[Option C]", "[Option D]"],
    "answer": [index]
}
```

```json
{
    "question_id": "1",
    "question_title": "[Question 2]",
    "background": "[Background 2]",
    "options": ["[Option A]", "[Option B]", "[Option C]", "[Option D]"],
    "answer": [index]
}
```

```json
{
    "question_id": "2",
    "question_title": "[Question 3]",
    "background": "[Background 3]",
    "options": ["[Option A]", "[Option B]", "[Option C]", "[Option D]"],
    "answer": [index]
}
```

```json
{
    "question_id": "3",
    "question_title": "[Question 4]",
    "background": "[Background 4]",
    "options": ["[Option A]", "[Option B]", "[Option C]", "[Option D]"],
    "answer": [index]
}
```

```json
{
    "question_id": "4",
    "question_title": "[Question 5]",
    "background": "[Background 5]",
    "options": ["[Option A]", "[Option B]", "[Option C]", "[Option D]"],
    "answer": [index]
}
```
"""
        return prompt
    
    def _format_freeq_prompt(self, article: Dict) -> str:
        """Format prompt for free-form short answer questions using XML format."""
        source_article = f"Title: {article.get('title', '')}\n\n"
        publish_date = None
        
        if 'description' in article and article['description']:
            source_article += f"Description: {article['description']}\n\n"
            
        if 'maintext' in article and article['maintext']:
            source_article += f"Content: {article['maintext']}\n\n"
            
        # if 'date_download' in article and article['date_download']:
        #     publish_date = article['date_download']
        
        if 'date_publish' in article and article['date_publish']:
            publish_date = article['date_publish']
            
        if 'date_modify' in article and article['date_modify']:
            publish_date = article['date_modify']
            
        if publish_date:
            source_article += f"Published Date: {publish_date}\n"
            
        
        prompt = f"""ASSUME TODAY's DATE IS {publish_date}. PLEASE DO THE FOLLOWING TASK BASED ON THIS ASSUMPTION.""" if publish_date else ""
        
        if self.num_questions_per_article == 1:
            prompt += f"""
**Task:** Based on the provided news article, generate {self.num_questions_per_article} high quality forecasting question which has a short answer (1 - 3 words), using the XML format specified below.
Forecasting question should be posed in a way to predict future events. Here, the predictor will have a knowledge cutoff before the article is published and no access to the article, so a forecasting question has to be posed about information explicitly stated in the article. The question should be stated in a forward-looking manner (towards the future).
The correct answer should be a specific, short text response. The answer should be a WELL DEFINED, SPECIFIC term which the answerer can come up with on its own, without access to the news article. 

**Example Format**:
<q1>
<question_id>0</question_id>
<question_title>Who will win the Nobel Prize in Literature in 2016?</question_title>
<background>Question Start Date: 10th January 2016. The Nobel Prize in Literature is awarded annually by the Swedish Academy to authors for their outstanding contributions to literature.</background>
<resolution_criteria> 
<ul>
    <li>
      <b>Source of Truth</b>: The question will resolve when the Swedish Academy publicly announces the official 2016 Nobel Prize in Literature laureate(s)—typically via a press release on NobelPrize.org (expected on or about October 13, 2016).  
    </li>
    <li>
      <b>Resolution Date</b>: The resolution occurs on the calendar date when the 2016 laureate(s) are formally named
      (typically mid-October 2016). 
    </li>
    <li>
      <b>Accepted Answer Format</b>: The full name of the laureate exactly as given in the announcement should be provided. If more than one person shares the prize, all names must be listed in the same order as the official communiqué.
    </li>
</ul>
</resolution_criteria>
<answer>Bob Dylan</answer>
<answer_type>String (Name)</answer_type>
</q1>

The question should follow the structured guidelines below.

### **Guidelines for Creating Short Answer Forecasting Questions**

**Title Question Guidelines**
- **Quality**: The question should be of HIGH QUALITY and hard to answer without access to the article. It should not be about any minute details in the article. THE QUESTION SHOULD BE SUCH THAT ITS ANSWER REVEALS A KEY PIECE OF INFORMATION, FROM THE ARTICLE, WHICH HAS MAXIMAL IMPACT.
- **Specific and Answerable**: The question to be created SHOULD BE FREE-FORM and have a unique, specific answer (a single word, or short phrase) without access to the article. The answer to the question should be definite, well-defined and NOT NUMERIC. IT SHOULD ALSO NOT BE UNCERTAIN like "above XYZ" OR A RANGE LIKE "between XYZ and ABC". Avoid creating binary questions (yes/no, either/or) or questions with a list of specific options (multiple choice).
- **Answerable based on article**: Each question must have a CLEAR AND DEFINITE answer based on information stated in the article. Given the question, the content of the article should be able to resolve the answer to the question INDISPUTABLY WITHOUT ANY AMBIGUITY OR UNCERTAINTY. THE ARTICLE SHOULD NOT STATE THAT THE ANSWER IS TENTATIVE OR AN ESTIMATE OR LIKELY. The answer SHOULD HAVE HAPPENED BY NOW.
- **Temporal Information**: The question should not be about recall of (past) facts or events known before the article publish date. Include any temporal information necessary to answer the question (like by which month, year, etc.) in the question. The question should always be posed in a forward-looking manner. 
- **Direct and Precise**: Titles must be straightforward and unambiguous, avoiding vague terms. Use future tense when appropriate.
- **Resolution Criteria**: ALWAYS INCLUDE A BRIEF RESOLUTION CRITERIA in the question title. This is often the date by which the question will be resolved. For example, resolution dates such as "by {{month_name}}, {{year}}?" or "in {{month_name}}, {{year}}?". THE RESOLUTION DATE SHOULD BE BASED ON (AND FAITHFUL TO) THE CONTENT OR PUBLICATION DATE OF THE ARTICLE.
- **No references to article or future information**: DO NOT refer to the specific article, such as by saying "in the article". The forecaster does not have access to the article, its metadata or any information beyond the article publish date.
- **Question Types**: Focus on "Who", "What", "When", "Where" questions that have concrete answers.
- **Understandability**: The question title should have ALL the information to be understandable by a 10 year old. It should be independently understandable without the article.
- **Tense**. ALWAYS POSE THE QUESTION IN A FORWARD-LOOKING MANNER. THE QUESTION SHOULD BE IN FUTURE TENSE. Try to use phrases like "What will", "Who will", "When will", "Where will", "How much/many will" etc. It should appear as a forecasting question and not past prediction. 


**Answer Guidelines**
- **Faithfulness to Article**: The answer should be based on information explicitly stated in the article, and not implications or your own knowledge. IT SHOULD BE STATED VERBATIM IN THE ARTICLE.
- **Non-Numeric**: The answer should not be a number or a percentage. It can be a word, phrase, date, location, etc BUT NOT MORE THAN 3 WORDS.
- **Definite** - Given the question and the article, the answer should be CLEAR, CONCRETE, CERTAIN AND DERIVABLE from the article. It should be short, WELL-DEFINED TERM and not uncertain or vague. It SHOULD NOT BE A RANGE like "between XYZ and ABC" or "above XYZ" or "below PQR".
- **Resolved** - The answer MUST be something that has already happened or is happening now. It should be resolved given today's date and not be something that will happen in the future.
- **Specificity**: The answer should be specific enough to be unambiguous. Avoid overly general answers.
- **Conciseness**: Keep answers short - typically 1-3 words, occasionally a short phrase if necessary.
- **Exactness**: For names, use the exact names mentioned (full name, if possible).
- **Uniqueness**: The answer should be unique and THE ONLY CORRECT ANSWER to the question. 
- **No Ambiguity**: The answer should be indisputable and not be open to multiple interpretations. IT SHOULD BE PRECISE AND NOT A RANGE OR UNCERTAIN ESTIMATE.

**Background Guidelines**
- **Mention Question Opening Date**: ALWAYS INCLUDE THE START DATE OF THE QUESTION IN THE BACKGROUND. IT SHOULD BE AT LEAST A FEW DAYS (OR WEEKS IF THE QUESTION IS ABOUT A LONG-TERM EVENT) BEFORE THE ARTICLE'S PUBLISH DATE AND ALSO BEFORE THE RESOLUTION DATE OF THE QUESTION. CONSEQUENTLY, THE BACKGROUND SHOULD NOT CONTAIN ANY INFORMATION WHICH HAS HAPPENED AFTER THE START DATE OF THE QUESTION.
- **Necessary Context**: The answerer does not have access to the article, so include MINIMAL CONTEXT required to understand the question keeping in mind the question opening date. Do not give (extra) details of the event from the article as background. If required, EITHER pose the event as a hypothetical scenario as if it were to happen in the future OR describe it as happening (unfolding) in real time. Describe any unfamiliar terms or concepts in the question title. 
- **SHOULD NOT HELP ANSWER**: WHILE PROVIDING THE CONTEXT, DO NOT REFER OR MENTION OR LEAK THE ACTUAL ANSWER. The background must not help answer the forecasting question. DO NOT INCLUDE ANY INFORMATION from the article or elsewhere that either directly or indirectly (even partially) reveals the answer.
- **No Additional Knowledge**: Do not add any knowledge beyond what is required to understand the question. Only include information necessary to understand the question and its context. 
- **Tense**. ALWAYS POSE THE BACKGROUND INFORMATION IN CURRENT TENSE. Only provide minimal information which is known until the question opening date.

**Resolution Criteria**
- **Necessary Criteria**: State the EXACT conditions by which the outcome will be judged. Include the criteria which determines how the question will be resolved. state the conditions by which the outcome will be judged. 
- **Date and Source of Resolution**: Always state the date and the source by which the question will be resolved. For example, resolution dates such as "by {{month_name}}, {{year}}?" or "in {{month_name}}, {{year}}?", and potential source(s) of resolution such as "based on {{news source}}", "reports from {{official name}}", etc. THE RESOLUTION DATE SHOULD BE CHOSEN THOUGHTFULLY AS THE ANSWER'S VALIDITY AND SOUNDNESS DEPENDS ON IT. THE RESOLUTION DATE SHOULD BE SUCH THAT THE ANSWER CAN BE RESOLVED DEFINITELY AND INDISPUTABLY FROM THE CONTENT OR PUBLICATION DATE OF THE ARTICLE. IT SHOULD MENTION BY WHEN IS THE OUTCOME OF THE QUESTION EXPECTED TO HAPPEN. HOWEVER, IT SHOULD NOT LEAK OR MENTION ANYTHING ABOUT THE ARTICLE. 
- **Details**: Be as detailed as possible in creating the resolution criteria for resolving the question as cleanly as possible. There should be no ambiguity in the resolution criteria.
- **Expectation and Format of Answer**: Based on the actual answer, the resolution criteria should state how precise the expected answer should be and in what format it should be. For example, if the actual answer is a date, the resolution criteria should specify how detailed the expected date should be -- only year, or both month and year, or day, month, and year all together. DO NOT GIVE THE ACTUAL DATE (ANSWER). If the actual answer is a percentage, then the criteria should state the expected answer should be a percentage. DO NOT GIVE THE ACTUAL PERCENTAGE.  If the actual answer is in certain unit, then the criteria should specify that. THE RESOLUTION CRITERIA SHOULD MAKE IT EXACTLY CLEAR AND PRECISE WHAT IS EXPECTED FROM THE ANSWERER AND IN WHAT FORMAT AND HOW IT WILL BE CHECKED LATER. IF GIVING AN EXAMPLE, IT SHOULD BE VERY GENERIC AND AS FAR AWAY FROM THE ACTUAL ANSWER AS POSSIBLE.
- **SHOULD NOT HELP ANSWER**: The resolution criteria must not directly help answer the forecasting question. DO NOT INCLUDE ANY INFORMATION from the article or elsewhere that either directly or indirectly (even partially) reveals the answer. DO NOT REFER OR MENTION OR LEAK THE ACTUAL ANSWER HERE.

**Answer Type Guidelines**
- **Expected Format**: The answer type should be either "numeric (XYZ)" if the answer is a number (of any kind) or "string (XYZ)" in all other cases. In numeric cases, XYZ should be the exact type of number expected. For example, "numeric (integer)", "numeric (decimal)", "numeric (percentage)", "numeric (whole number)", etc. In string cases, XYZ should broadly be the category of string expected. For example, "string (name)", "string (date)", "string (location)", etc. If the category is not clear, use "string (any)". HOWEVER, ALWAYS TRY TO CREATE QUESTIONS WHERE THE ANSWER CATEGORY IS CLEAR AND PRECISE.

**Question Quality Criteria**
- **Forecastable**: The question should be something that could reasonably be predicted or forecasted before the article's publication.
- **Towards the future**: THE QUESTION SHOULD BE POSED IN A FORWARD-LOOKING MANNER.
- **Interesting**: The question should be about a meaningful event or outcome, not trivial details.
- **Impactful**: The question should be such that if its answer is forecasted ahead of time, it should have significant (downstream) impact (relevant to high number of people).
- **Difficulty**: While the question should be hard to answer without access to the article, it should also not be unreasonably difficult.
- **Verifiable**: The answer should be something that can be EXACTLY verified from the article itself.
- **Time-bound**: Include clear timeframes or deadlines when relevant.
- **Free-form**: If possible, avoid creating binary questions (yes/no, either/or) or questions with a list of specific options (multiple choice).

Generate {self.num_questions_per_article} high-quality short answer forecasting questions based on the provided article. Use the XML format with question_id value "0". DO NOT INCLUDE ANY ANALYSIS, RANKING, OR ADDITIONAL COMMENTARY.

Article:
{source_article}

**Required Output Format**:
<q1>
<question_id>0</question_id>
<question_title>[Question 1]</question_title>
<background>[Background 1]</background>
<resolution_criteria>[Resolution Criteria 1]</resolution_criteria>
<answer>[Answer 1]</answer>
<answer_type>[Answer Type 1]</answer_type>
</q1>
"""
        
        
        elif self.num_questions_per_article > 1:
            prompt += f"""
**Task:** Based on the provided news article, generate {self.num_questions_per_article} high-quality, DIVERSE forecasting questions which have a short answer (1 - 3 words), using the XML format specified below.
Each forecasting question should be posed in a way to predict future events. Here, the predictor will have a knowledge cutoff before the article is published and no access to the article, so a forecasting question has to be posed about information explicitly stated in the article. The question should be stated in a forward-looking manner (towards the future).
The correct answer should be a specific, short text response. The answer should be a WELL DEFINED, SPECIFIC term which the answerer can come up with on its own, without access to the news article. 

**Example Format**:
<q1>
<question_id>0</question_id>
<question_title>Who will win the Nobel Prize in Literature in 2016?</question_title>
<background>Question Start Date: 10th January 2016. The Nobel Prize in Literature is awarded annually by the Swedish Academy to authors for their outstanding contributions to literature.</background>
<resolution_criteria> 
<ul>
    <li>
      <b>Source of Truth</b>: The question will resolve when the Swedish Academy publicly announces the official 2016 Nobel Prize in Literature laureate(s)—typically via a press release on NobelPrize.org (expected on or about October 13, 2016).  
    </li>
    <li>
      <b>Resolution Date</b>: The resolution occurs on the calendar date when the 2016 laureate(s) are formally named
      (typically mid-October 2016). 
    </li>
    <li>
      <b>Accepted Answer Format</b>: The full name of the laureate exactly as given in the announcement should be provided. If more than one person shares the prize, all names must be listed in the same order as the official communiqué.
    </li>
</ul>
</resolution_criteria>
<answer>Bob Dylan</answer>
<answer_type>String (Name)</answer_type>
</q1>

The question should follow the structured guidelines below.

### **Guidelines for Creating Short Answer Forecasting Questions**

**Title Question Guidelines**
- **Quality**: The question should be of HIGH QUALITY and hard to answer without access to the article. It should not be about any minute details in the article. THE QUESTION SHOULD BE SUCH THAT ITS ANSWER REVEALS A KEY PIECE OF INFORMATION, FROM THE ARTICLE, WHICH HAS MAXIMAL IMPACT.
- **Specific and Answerable**: The question to be created SHOULD BE FREE-FORM and have a unique, specific answer (a single word, or short phrase) without access to the article. The answer to the question should be definite, well-defined and NOT NUMERIC. IT SHOULD ALSO NOT BE UNCERTAIN like "above XYZ" OR A RANGE LIKE "between XYZ and ABC". Avoid creating binary questions (yes/no, either/or) or questions with a list of specific options (multiple choice).
- **Answerable based on article**: Each question must have a CLEAR AND DEFINITE answer based on information stated in the article. Given the question, the content of the article should be able to resolve the answer to the question INDISPUTABLY WITHOUT ANY AMBIGUITY OR UNCERTAINTY. THE ARTICLE SHOULD NOT STATE THAT THE ANSWER IS TENTATIVE OR AN ESTIMATE OR LIKELY. The answer SHOULD HAVE HAPPENED BY NOW.
- **Temporal Information**: The question should not be about recall of (past) facts or events known before the article publish date. Include any temporal information necessary to answer the question (like by which month, year, etc.) in the question. The question should always be posed in a forward-looking manner. 
- **Direct and Precise**: Titles must be straightforward and unambiguous, avoiding vague terms. Use future tense when appropriate.
- **Resolution Criteria**: ALWAYS INCLUDE A BRIEF RESOLUTION CRITERIA in the question title. This is often the date by which the question will be resolved. For example, resolution dates such as "by {{month_name}}, {{year}}?" or "in {{month_name}}, {{year}}?". THE RESOLUTION DATE SHOULD BE BASED ON (AND FAITHFUL TO) THE CONTENT OR PUBLICATION DATE OF THE ARTICLE.
- **No references to article or future information**: DO NOT refer to the specific article, such as by saying "in the article". The forecaster does not have access to the article, its metadata or any information beyond the article publish date.
- **Question Types**: Focus on "Who", "What", "When", "Where" questions that have concrete answers.
- **Understandability**: The question title should have ALL the information to be understandable by a 10 year old. It should be independently understandable without the article.
- **Tense**. ALWAYS POSE THE QUESTION IN A FORWARD-LOOKING MANNER. THE QUESTION SHOULD BE IN FUTURE TENSE. Try to use phrases like "What will", "Who will", "When will", "Where will", "How much/many will" etc. It should appear as a forecasting question and not past prediction. 

**Answer Guidelines**
- **Faithfulness to Article**: The answer should be based on information explicitly stated in the article, and not implications or your own knowledge. IT SHOULD BE STATED VERBATIM IN THE ARTICLE.
- **Non-Numeric**: The answer should not be a number or a percentage. It can be a word, phrase, date, location, etc BUT NOT MORE THAN 3 WORDS.
- **Definite** - Given the question and the article, the answer should be CLEAR, CONCRETE, CERTAIN AND DERIVABLE from the article. It should be short, WELL-DEFINED TERM and not uncertain or vague. It SHOULD NOT BE A RANGE like "between XYZ and ABC" or "above XYZ" or "below PQR".
- **Resolved** - The answer MUST be something that has already happened or is happening now. It should be resolved given today's date and not be something that will happen in the future.
- **Specificity**: The answer should be specific enough to be unambiguous. Avoid overly general answers.
- **Conciseness**: Keep answers short - typically 1-3 words, occasionally a short phrase if necessary.
- **Exactness**: For names, use the exact names mentioned (full name, if possible).
- **Uniqueness**: The answer should be unique and THE ONLY CORRECT ANSWER to the question. 
- **No Ambiguity**: The answer should be indisputable and not be open to multiple interpretations. IT SHOULD BE PRECISE AND NOT A RANGE OR UNCERTAIN ESTIMATE.

**Background Guidelines**
- **Mention Question Opening Date**: ALWAYS INCLUDE THE START DATE OF THE QUESTION IN THE BACKGROUND. IT SHOULD BE AT LEAST A FEW DAYS (OR WEEKS IF THE QUESTION IS ABOUT A LONG-TERM EVENT) BEFORE THE ARTICLE'S PUBLISH DATE AND ALSO BEFORE THE RESOLUTION DATE OF THE QUESTION. CONSEQUENTLY, THE BACKGROUND SHOULD NOT CONTAIN ANY INFORMATION WHICH HAS HAPPENED AFTER THE START DATE OF THE QUESTION.
- **Necessary Context**: The answerer does not have access to the article, so include MINIMAL CONTEXT required to understand the question keeping in mind the question opening date. Do not give (extra) details of the event from the article as background. If required, EITHER pose the event as a hypothetical scenario as if it were to happen in the future OR describe it as happening (unfolding) in real time. Describe any unfamiliar terms or concepts in the question title. 
- **SHOULD NOT HELP ANSWER**: WHILE PROVIDING THE CONTEXT, DO NOT REFER OR MENTION OR LEAK THE ACTUAL ANSWER. The background must not help answer the forecasting question. DO NOT INCLUDE ANY INFORMATION from the article or elsewhere that either directly or indirectly (even partially) reveals the answer.
- **No Additional Knowledge**: Do not add any knowledge beyond what is required to understand the question. Only include information necessary to understand the question and its context. 
- **Tense**. ALWAYS POSE THE BACKGROUND INFORMATION IN CURRENT TENSE. Only provide minimal information which is known until the question opening date.

**Resolution Criteria**
- **Necessary Criteria**: State the EXACT conditions by which the outcome will be judged. Include the criteria which determines how the question will be resolved. state the conditions by which the outcome will be judged. 
- **Date and Source of Resolution**: Always state the date and the source by which the question will be resolved. For example, resolution dates such as "by {{month_name}}, {{year}}?" or "in {{month_name}}, {{year}}?", and potential source(s) of resolution such as "based on {{news source}}", "reports from {{official name}}", etc. THE RESOLUTION DATE SHOULD BE CHOSEN THOUGHTFULLY AS THE ANSWER'S VALIDITY AND SOUNDNESS DEPENDS ON IT. THE RESOLUTION DATE SHOULD BE SUCH THAT THE ANSWER CAN BE RESOLVED DEFINITELY AND INDISPUTABLY FROM THE CONTENT OR PUBLICATION DATE OF THE ARTICLE. IT SHOULD MENTION BY WHEN IS THE OUTCOME OF THE QUESTION EXPECTED TO HAPPEN. HOWEVER, IT SHOULD NOT LEAK OR MENTION ANYTHING ABOUT THE ARTICLE. 
- **Details**: Be as detailed as possible in creating the resolution criteria for resolving the question as cleanly as possible. There should be no ambiguity in the resolution criteria.
- **Expectation and Format of Answer**: Based on the actual answer, the resolution criteria should state how precise the expected answer should be and in what format it should be. For example, if the actual answer is a date, the resolution criteria should specify how detailed the expected date should be -- only year, or both month and year, or day, month, and year all together. DO NOT GIVE THE ACTUAL DATE (ANSWER). If the actual answer is a percentage, then the criteria should state the expected answer should be a percentage. DO NOT GIVE THE ACTUAL PERCENTAGE.  If the actual answer is in certain unit, then the criteria should specify that. THE RESOLUTION CRITERIA SHOULD MAKE IT EXACTLY CLEAR AND PRECISE WHAT IS EXPECTED FROM THE ANSWERER AND IN WHAT FORMAT AND HOW IT WILL BE CHECKED LATER. IF GIVING AN EXAMPLE, IT SHOULD BE VERY GENERIC AND AS FAR AWAY FROM THE ACTUAL ANSWER AS POSSIBLE.
- **SHOULD NOT HELP ANSWER**: The resolution criteria must not directly help answer the forecasting question. DO NOT INCLUDE ANY INFORMATION from the article or elsewhere that either directly or indirectly (even partially) reveals the answer. DO NOT REFER OR MENTION OR LEAK THE ACTUAL ANSWER HERE.

**Answer Type Guidelines**
- **Expected Format**: The answer type should be either "numeric (XYZ)" if the answer is a number (of any kind) or "string (XYZ)" in all other cases. In numeric cases, XYZ should be the exact type of number expected. For example, "numeric (integer)", "numeric (decimal)", "numeric (percentage)", "numeric (whole number)", etc. In string cases, XYZ should broadly be the category of string expected. For example, "string (name)", "string (date)", "string (location)", etc. If the category is not clear, use "string (any)". HOWEVER, ALWAYS TRY TO CREATE QUESTIONS WHERE THE ANSWER CATEGORY IS CLEAR AND PRECISE.

**Question Quality Criteria**
- **Forecastable**: The question should be something that could reasonably be predicted or forecasted before the article's publication.
- **Towards the future**: THE QUESTION SHOULD BE POSED IN A FORWARD-LOOKING MANNER.
- **Interesting**: The question should be about a meaningful event or outcome, not trivial details.
- **Impactful**: The question should be such that if its answer is forecasted ahead of time, it should have significant (downstream) impact (relevant to high number of people).
- **Difficulty**: While the question should be hard to answer without access to the article, it should also not be unreasonably difficult.
- **Verifiable**: The answer should be something that can be EXACTLY verified from the article itself.
- **Time-bound**: Include clear timeframes or deadlines when relevant.
- **Free-form**: If possible, avoid creating binary questions (yes/no, either/or) or questions with a list of specific options (multiple choice).

Generate {self.num_questions_per_article} high-quality, DIVERSE short answer forecasting questions based on the provided article. Use the XML format with question_id value "0", "1", "2", etc. DO NOT INCLUDE ANY ANALYSIS, RANKING, OR ADDITIONAL COMMENTARY.

Article:
{source_article}

**Required Output Format**:
<q1>
<question_id>0</question_id>
<question_title>[Question 1]</question_title>
<background>[Background 1]</background>
<resolution_criteria>[Resolution Criteria 1]</resolution_criteria>
<answer>[Answer 1]</answer>
<answer_type>[Answer Type 1]</answer_type>
</q1>
..
<q{self.num_questions_per_article}>
<question_id>{self.num_questions_per_article - 1}</question_id>
<question_title>[Question {self.num_questions_per_article}]</question_title>
<background>[Background {self.num_questions_per_article}]</background>
<resolution_criteria>[Resolution Criteria {self.num_questions_per_article}]</resolution_criteria>
<answer>[Answer {self.num_questions_per_article}]</answer>
<answer_type>[Answer Type {self.num_questions_per_article}]</answer_type>
</q{self.num_questions_per_article}>
"""
        
        
        return prompt

    def _load_existing_results(self, output_path: str) -> Dict[str, Dict]:
        """Loads existing results from the output file."""
        if not os.path.exists(output_path):
            return {}
        
        existing_data = []
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            item = json.loads(line)
                            existing_data.append(item)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON line in {output_path}. Skipping line.")
                
            # Use article_url as the key for quick lookup
            return {result.get("url", f"missing_url_{i}"): result for i, result in enumerate(existing_data)}
        except Exception as e:
            logger.error(f"Error loading existing results from {output_path}: {e}")
            return {}

    def _append_new_results(self, new_results: List[Dict], output_path: str) -> None:
        """Appends new results to the output file without reloading existing data."""
        try:
            # Ensure output directory exists (only if output_path has a directory component)
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only if there's a directory component
                os.makedirs(output_dir, exist_ok=True)
            
            # Append new results to the file
            with open(output_path, 'a') as f:
                for result in new_results:
                    f.write(json.dumps(result) + '\n')
                    
            logger.info(f"Appended {len(new_results)} new results to {output_path}")
        except Exception as e:
            logger.error(f"Error appending new results to {output_path}: {e}")

    async def generate_questions(self, articles: List[Dict], output_path: str, batch_size: int = 5, regenerate: bool = False) -> List[Dict]:
        """
        Generate forecasting questions based on the configured method,
        loading existing results and saving incrementally.

        Args:
            articles: List of article dictionaries
            output_path: Path to save the results to (and load from)
            batch_size: Number of articles to process in parallel
            regenerate: If True, ignore existing results and start fresh.

        Returns:
            List of results containing the generated questions
        """
        if regenerate and os.path.exists(output_path):
            logger.info(f"Regenerate flag set. Removing existing results file: {output_path}")
            os.remove(output_path)
        
        # Create a new file if it doesn't exist or if regenerate is True
        if not os.path.exists(output_path):
            # Only create directory if output_path has a directory component
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only if there's a directory component
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w') as f:
                pass  # Create empty file
        
        # Load existing results once at the beginning
        existing_results_map = self._load_existing_results(output_path)
        logger.info(f"Loaded {len(existing_results_map)} existing results.")
        
        # Track which articles need to be reprocessed due to empty or invalid generated questions
        to_reprocess_urls = set()
        for url, result in existing_results_map.items():
            generated_text = result.get("generated_questions", "")
            if not isinstance(generated_text, str):
                to_reprocess_urls.add(url)
                logger.info(f"Marking article with URL {url} for reprocessing due to non-string generated questions")
                continue
            
            all_q_present = True 
            for q_num in range(1, self.num_questions_per_article + 1):
                if f"<q{q_num}>" not in generated_text or f"</q{q_num}>" not in generated_text or f"<question_id>{q_num - 1}</question_id>" not in generated_text:
                    all_q_present = False
                    break
                
            # Check if generated_questions is empty, None, or contains an error message
            if (not generated_text or not all_q_present or 
                generated_text is None or 
                (isinstance(generated_text, str) and 
                 (generated_text.strip() == "" or "ERROR:" in generated_text))):
                to_reprocess_urls.add(url)
                logger.info(f"Marking article with URL {url} for reprocessing due to empty or invalid generated questions")
        
    
        # Build a map from url to article for quick lookup
        article_url_to_article = {}
        for article in articles:
            article_url = article.get("url", "")
            if not article_url:
                content_hash = hash(article.get('title', '') + article.get('maintext', ''))
                article_url = f"no_url_{content_hash}"
            article_url_to_article[article_url] = article

        # Determine which articles need to be processed:
        # 1. Articles not present in existing_results_map (new)
        # 2. Articles present but in to_reprocess_urls (invalid/empty)
        pending_articles = []
        pending_urls = set()
        for article_url, article in article_url_to_article.items():
            if article_url not in existing_results_map or article_url in to_reprocess_urls:
                pending_articles.append(article)
                pending_urls.add(article_url)

        # Prepare the final results: start with all valid existing results (not in to_reprocess)
        final_results_map = {url: result for url, result in existing_results_map.items() if url not in to_reprocess_urls}

        if not pending_articles:
            logger.info("No new articles to process. All results loaded from existing file.")
            return list(final_results_map.values())

        logger.info(f"Loaded {len(final_results_map)} existing results. Processing {len(pending_articles)} new or reprocessing articles.")

        prompts = [self.format_prompt(article) for article in pending_articles]

        # Process pending articles in batches
        for i in range(0, len(pending_articles), batch_size):
            batch_articles = pending_articles[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            logger.info(f"Processing batch {i//batch_size + 1}/{(len(pending_articles) + batch_size - 1)//batch_size}...")
            # Generate completions using the inference engine
            try:
                generated_texts = await self.inference_engine.generate(batch_prompts, batch_size=len(batch_prompts))
                generated_texts = [text['response'] for text in generated_texts]
            except Exception as e:
                logger.error(f"Error during inference engine generation for batch starting at index {i}: {e}")
                # Add placeholder results for failed batch to avoid reprocessing on next run
                generated_texts = ["ERROR: Generation failed"] * len(batch_prompts)

            # Create a list for the batch results
            batch_results = []

            # Pair the generated texts with article info for the current batch
            for j, article in enumerate(batch_articles):
                article_url = article.get("url", "")
                if not article_url:
                    content_hash = hash(article.get('title', '') + article.get('maintext', ''))
                    article_url = f"no_url_{content_hash}"

                article_result = {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "maintext": article.get("maintext", ""),
                    "url": article_url,
                    "date_publish": article.get("date_publish", ""),
                    "date_modify": article.get("date_modify", ""),
                    "date_download": article.get("date_download", ""),
                    "generated_questions": generated_texts[j] if j < len(generated_texts) else "ERROR: Index out of bounds",
                }
                batch_results.append(article_result)
                # Update or add to final_results_map
                final_results_map[article_url] = article_result

            # Append only the new batch results to the file
            self._append_new_results(batch_results, output_path)

        logger.info(f"Finished question generation. Total results: {len(final_results_map)}")
        return list(final_results_map.values())
    
    
    async def extract_individual_questions(self, output_path: str, batch_size: int = 5) -> List[Dict]:
        """
        Extract individual questions from generated questions text.
        Creates ordered list of questions and saves them as q1, q2, q3, etc.

        Args:
            output_path: Path to load and save results
            batch_size: Number of articles to process in parallel

        Returns:
            List of results with individual questions extracted
        """
        # Load existing results
        existing_results = self._load_existing_results(output_path)
        if not existing_results:
            logger.warning("No existing results found for question extraction.")
            return []

        # Filter articles that need question extraction
        pending_urls = []
        pending_results = []
        for url, result in existing_results.items():
            if result.get("questions_extracted", 0) != 1:
                pending_urls.append(url)
                pending_results.append(result)

        if not pending_results:
            logger.info("All articles already have individual questions extracted.")
            return list(existing_results.values())

        logger.info(f"Extracting individual questions for {len(pending_results)} articles.")

        # Process each article
        for i, (url, result) in enumerate(zip(pending_urls, pending_results)):
            # logger.info(f"Extracting questions for article {i+1}/{len(pending_results)}")
            
            generated_text = result.get("generated_questions", "")
            if not generated_text or generated_text.startswith("ERROR:"):
                # Mark as processed but with no valid questions
                existing_results[url]["questions_extracted"] = 1
                for q_num in range(1, self.num_questions_per_article + 1):
                    existing_results[url][f"q{q_num}"] = ""
                continue
            
            # Extract individual questions using regex
            extracted_questions = self._extract_questions_from_text(generated_text)
            
            # Save each question as q1, q2, q3, etc.
            for q_num in range(1, self.num_questions_per_article + 1):
                if q_num <= len(extracted_questions):
                    existing_results[url][f"q{q_num}"] = extracted_questions[q_num - 1]
                else:
                    existing_results[url][f"q{q_num}"] = ""
            
            # Mark as processed
            existing_results[url]["questions_extracted"] = 1
            
            # logger.info(f"  Extracted {len(extracted_questions)} questions for article")

        # Save updated results
        self.save_results(list(existing_results.values()), output_path)
        
        logger.info(f"Finished question extraction. Updated {len(pending_results)} articles.")
        return list(existing_results.values())

    def update_individual_validation_done(self, output_path: str) -> None:
        """
        Update the individual_validation_done field for all articles.
        """
        existing_results = self._load_existing_results(output_path)
        for url in existing_results.keys():
            for q_num in range(1, self.num_questions_per_article + 1):
                validation_text = existing_results[url].get(f"q{q_num}_validation_response", "")
                if not validation_text:
                    continue
                
                # Extract 1 or 0 from the response
                last1 = validation_text.rfind("1")
                last0 = validation_text.rfind("0")
                
                if last1 > last0:
                    existing_results[url][f"q{q_num}_valid"] = 1
                else:
                    existing_results[url][f"q{q_num}_valid"] = 0
                        
        self.save_results(list(existing_results.values()), output_path)
    
    async def validate_individual_questions(self, output_path: str, batch_size: int = 5) -> List[Dict]:
        """
        Validate each individual question and save results as q1_valid, q2_valid, etc.

        Args:
            output_path: Path to load and save results
            batch_size: Number of articles to process in parallel

        Returns:
            List of results with individual question validation
        """
        if not self.validate_questions or not self.choose_engine:
            logger.info("Individual validation functionality not enabled or engine not available.")
            return []
        
        self.update_individual_validation_done(output_path)
        # Load existing results
        existing_results = self._load_existing_results(output_path)
        if not existing_results:
            logger.warning("No existing results found for individual question validation.")
            return []

        # Filter articles that need individual validation
        pending_urls = []
        pending_results = []
        for url, result in existing_results.items():
            condition = result.get("individual_validation_done", 0) != 1
            for i in range(1, self.num_questions_per_article + 1):
                condition = condition or len(result.get(f"q{i}_validation_response", "")) <= 1
                
            if condition:
                # logger.info(f"Article {url} needs individual validation")
                # logger.info(f"Individual validation done: {result.get('individual_validation_done', 0)}")
                # for j in range(1, self.num_questions_per_article + 1):
                #     logger.info(f"Length of q{j}_validation_response: {len(result.get(f'q{j}_validation_response', ''))}")
                #     logger.info(f"q{j}_valid: {result.get(f'q{j}_valid', 0)}")
                    
                #     if len(result.get(f"q{j}_validation_response", "")) <= 10:
                #         logger.info(f"q{j}_validation_response: {result.get(f'q{j}_validation_response', '')}")
                #         logger.info(f"q{j}_valid: {result.get(f'q{j}_valid', 0)}")
                #         logger.info(f"Question: {result.get(f'q{j}', '')}")
                #         logger.info(f"Generated questions: {result.get('generated_questions', '')}")
                        
                    # logger.info(f"q{i}_validation_response: {result.get(f'q{i}_validation_response', '')}")
                    
                pending_urls.append(url)
                pending_results.append(result)

        # return list(existing_results.values())
    
        if not pending_results:
            # Calculate validation statistics
            total_questions = 0
            valid_questions = 0
            articles_with_valid_questions = 0
            for result in existing_results.values():
                article_has_valid = False
                for q_num in range(1, self.num_questions_per_article + 1):
                    if result.get(f"q{q_num}", ""):
                        total_questions += 1
                        if result.get(f"q{q_num}_valid", 0) == 1:
                            valid_questions += 1
                            article_has_valid = True
                
                if article_has_valid:
                    articles_with_valid_questions += 1

            logger.info(f"All articles already have individual question validation completed. {valid_questions}/{total_questions} questions are valid. {articles_with_valid_questions}/{len(existing_results)} articles have at least one valid question.")
            return list(existing_results.values())

        logger.info(f"Processing individual question validation for {len(pending_results)} articles.")

        # Process in batches
        for i in range(0, len(pending_results), batch_size):
            batch_results = pending_results[i:i+batch_size]
            batch_urls = pending_urls[i:i+batch_size]

            logger.info(f"Processing individual validation batch {i//batch_size + 1}/{(len(pending_results) + batch_size - 1)//batch_size}...")
            
            # Collect all questions that need validation from this batch
            validation_prompts = []
            validation_info = []  # (url_index, question_number)
            
            for j, (url, result) in enumerate(zip(batch_urls, batch_results)):
                for q_num in range(1, self.num_questions_per_article + 1):
                    question_text = result.get(f"q{q_num}", "")
                    if question_text and len(question_text.strip()) > 10 and len(result.get(f"q{q_num}_validation_response", "")) <= 10:
                        prompt = self._format_question_validation_prompt(question_text, result)
                        validation_prompts.append(prompt)
                        validation_info.append((j, q_num))

            logger.info(f"Length of validation prompts: {len(validation_prompts)}")
            # Run validation on all collected questions
            if validation_prompts:
                try:
                    validation_results = await self.choose_engine.generate(validation_prompts, batch_size=len(validation_prompts))
                    validation_results = [text['response'] for text in validation_results]
                    # Process validation results
                    for (url_idx, q_num), validation_text in zip(validation_info, validation_results):
                        if not validation_text:
                            continue
                        
                        # logger.info(f"Length of validation result: {len(validation_text)}")
                        url = batch_urls[url_idx]
                        
                        # Extract 1 or 0 from the response
                        last1 = validation_text.rfind("1")
                        last0 = validation_text.rfind("0")
                        
                        if last1 > last0:
                            existing_results[url][f"q{q_num}_valid"] = 1
                        else:
                            existing_results[url][f"q{q_num}_valid"] = 0
                        
                        # logger.info(f"Article {url} q{q_num} valid: {existing_results[url][f'q{q_num}_valid']}")
                        existing_results[url][f"q{q_num}_validation_response"] = validation_text
                        
                except Exception as e:
                    logger.error(f"Error during individual validation for batch starting at index {i}: {e}")
                    # Mark as failed but set default values
                    # for url in batch_urls:
                    #     for q_num in range(1, self.num_questions_per_article + 1):
                    #         existing_results[url][f"q{q_num}_valid"] = 0
            
            # Mark all articles in this batch as processed
            for url in batch_urls:
                all_present = True 
                # Set validation to 0 for empty questions
                for q_num in range(1, self.num_questions_per_article + 1):
                    if f"q{q_num}_valid" not in existing_results[url] or f"q{q_num}_validation_response" not in existing_results[url]:
                        all_present = False
                        break
                    
                if all_present:
                    existing_results[url]["individual_validation_done"] = 1

            # Save updated results
            self.save_results(list(existing_results.values()), output_path)
            
        # Calculate and log validation statistics
        total_questions = 0
        valid_questions = 0
        articles_with_valid_questions = 0
        
        for result in existing_results.values():
            article_has_valid = False
            for q_num in range(1, self.num_questions_per_article + 1):
                if result.get(f"q{q_num}", ""):
                    total_questions += 1
                    if result.get(f"q{q_num}_valid", 0) == 1:
                        valid_questions += 1
                        article_has_valid = True
            
            if article_has_valid:
                articles_with_valid_questions += 1

        # Save updated results
        self.save_results(list(existing_results.values()), output_path)
        
        logger.info(f"Finished individual validation. {valid_questions}/{total_questions} questions are valid.")
        logger.info(f"{articles_with_valid_questions}/{len(existing_results)} articles have at least one valid question.")
        return list(existing_results.values())

    async def choose_best_questions(self, output_path: str, batch_size: int = 5) -> List[Dict]:
        """
        Choose the best question from generated questions for each article.
        Only processes articles that don't already have choose_best field set to 1.

        Args:
            output_path: Path to load and save results
            batch_size: Number of articles to process in parallel

        Returns:
            List of results with choose_best field added
        """
        if not self.choose_best or not self.choose_engine:
            logger.info("Choose best functionality not enabled or engine not available.")
            return []

        # Load existing results
        existing_results = self._load_existing_results(output_path)
        if not existing_results:
            logger.warning("No existing results found for choose_best processing.")
            return []

        # Filter articles that need choose_best processing
        # Only process articles that have at least one valid question
        pending_urls = []
        pending_results = []
        for url, result in existing_results.items():
            if result.get("choose_best", 0) != 1:
            # if True:
                # Check if article has any valid questions
                has_valid_question = False
                for q_num in range(1, self.num_questions_per_article + 1):
                    if result.get(f"q{q_num}_valid", 0) == 1:
                        has_valid_question = True
                        break
                
                if has_valid_question:
                    pending_urls.append(url)
                    pending_results.append(result)
                else:
                    # Mark as processed with no good question since no valid questions
                    existing_results[url]["choose_best"] = 1
                    existing_results[url]["no_good_question"] = 1
                    logger.info(f"Article {url}: No valid questions, skipping choose_best")

        if not pending_results:
            logger.info("All articles already have choose_best processing completed or no valid questions.")
            return list(existing_results.values())

        logger.info(f"Processing choose_best for {len(pending_results)} articles with valid questions.")

        # Process in batches
        for i in range(0, len(pending_results), batch_size):
            batch_results = pending_results[i:i+batch_size]
            batch_urls = pending_urls[i:i+batch_size]

            logger.info(f"Processing choose_best batch {i//batch_size + 1}/{(len(pending_results) + batch_size - 1)//batch_size}...")
            
            # Prepare choose_best prompts with only valid questions
            choose_prompts = []
            for result in batch_results:
                # Collect only valid questions for this article
                valid_questions_text = self._prepare_valid_questions_text(result)
                prompt = self._format_choose_best_prompt(valid_questions_text, result)
                choose_prompts.append(prompt)

            try:
                choose_results = await self.choose_engine.generate(choose_prompts, batch_size=len(choose_prompts))
                choose_results = [text['response'] for text in choose_results]
                
                # Update results with choose_best field
                for j, (url, result, choose_text) in enumerate(zip(batch_urls, batch_results, choose_results)):
                    # Update the result in the existing_results dictionary
                    extracted_question = self.extract_final_question(choose_text)
                    existing_results[url]["choose_best_response"] = choose_text
                    
                    if "no good question" in choose_text.lower():
                        existing_results[url]["choose_best"] = 1
                        existing_results[url]["no_good_question"] = 1
                        
                    else:
                        existing_results[url]["no_good_question"] = 0
                        
                        # Whether to rerun choose best or not (if there was formatting error in output etc.) assuming one good question exists
                        if len(extracted_question) <= 10:
                            existing_results[url]["choose_best"] = 0
                        else:
                            existing_results[url]["choose_best"] = 1
                        
            except Exception as e:
                logger.error(f"Error during choose_best processing for batch starting at index {i}: {e}")
                # Mark as failed but don't set choose_best to 1
                for url in batch_urls:
                    existing_results[url]["choose_best"] = 0

        # Save updated results
        self.save_results(list(existing_results.values()), output_path)
        
        logger.info(f"Finished choose_best processing. Updated {len(pending_results)} articles.")
        return list(existing_results.values())

    async def check_questions_leakage(self, output_path: str, batch_size: int = 5) -> List[Dict]:
        """
        Check for answer leakage in generated questions and fix if necessary.
        Only processes articles that don't already have leakage_check field set to 1.

        Args:
            output_path: Path to load and save results
            batch_size: Number of articles to process in parallel

        Returns:
            List of results with leakage_check field added
        """
        if not self.check_leakage or not self.leakage_engine:
            logger.info("Leakage check functionality not enabled or engine not available.")
            return []

        # Load existing results
        existing_results = self._load_existing_results(output_path)
        if not existing_results:
            logger.warning("No existing results found for leakage checking.")
            return []

        # Filter articles that need leakage checking
        pending_urls = []
        pending_results = []
        for url, result in existing_results.items():
            # Skip articles with no good questions
            if result.get("no_good_question", 0) == 1:
                continue
            
            # Only process articles that have completed choose_best successfully
            if result.get("choose_best", 0) != 1:
                continue
            
            # Skip articles that already have leakage checking done
            if result.get("leakage_check", 0) != 1:
            # if True:
                pending_urls.append(url)
                pending_results.append(result)

        if not pending_results:
            logger.info("All articles already have leakage checking completed.")
            return list(existing_results.values())

        logger.info(f"Processing leakage checking for {len(pending_results)} articles.")

        # Process in batches
        for i in range(0, len(pending_results), batch_size):
            batch_results = pending_results[i:i+batch_size]
            batch_urls = pending_urls[i:i+batch_size]

            logger.info(f"Processing leakage check batch {i//batch_size + 1}/{(len(pending_results) + batch_size - 1)//batch_size}...")
            
            # Prepare leakage check prompts
            questions = []
            for result in batch_results:
                response = result.get("generated_questions", "")
                if result.get("choose_best", 0) == 1:
                    response = result.get("choose_best_response", "")
                    
                final_question = self.extract_final_question(response)
                if len(final_question) > 10:
                    questions.append(final_question)
                else:
                    questions.append("SKIP")
            
            leakage_prompts = self._prepare_leakage_check_prompts(
                questions,
                batch_results
            )
            
            # Filter out SKIP prompts and get valid prompts for batch processing
            valid_prompts = []
            valid_indices = []
            for j, prompt in enumerate(leakage_prompts):
                if prompt != "SKIP":
                    valid_prompts.append(prompt)
                    valid_indices.append(j)

            try:
                if valid_prompts:
                    corrected_batch = await self.leakage_engine.generate(valid_prompts, batch_size=len(valid_prompts))
                    corrected_batch = [text['response'] for text in corrected_batch]
                    
                    # Update only the valid indices with corrected texts
                    for idx, corrected_text in zip(valid_indices, corrected_batch):
                        if idx < len(batch_urls):
                            url = batch_urls[idx]
                            existing_results[url]["leakage_check_response"] = corrected_text
                            existing_results[url]["leakage_check"] = 1
                            existing_results[url]["final_question"] = self.extract_final_question(corrected_text)
                    
                    # Mark remaining articles as processed (even if they were skipped)
                    for j, url in enumerate(batch_urls):
                        if j not in valid_indices:                            
                            existing_results[url]["final_question"] = self.extract_final_question(existing_results[url]["generated_questions"])
                            # existing_results[url]["leakage_check"] = 1
                            
            except Exception as e:
                logger.error(f"Error during leakage checking for batch starting at index {i}: {e}")
                # Mark as failed but don't set leakage_check to 1
                for url in batch_urls:
                    existing_results[url]["leakage_check"] = 0

        # Save updated results
        self.save_results(list(existing_results.values()), output_path)
        
        logger.info(f"Finished leakage checking. Updated {len(pending_results)} articles.")
        return list(existing_results.values())



    async def run_pipeline(self, articles: List[Dict], output_path: str, batch_size: int = 5, regenerate: bool = False) -> List[Dict]:
        """
        Complete processing pipeline with individual question validation.
        
        Pipeline:
        1. Generate questions
        2. Extract individual questions 
        3. Validate each question individually (q1_valid, q2_valid, etc.)
        4. Choose best question from valid questions (if any valid)
        5. Check leakage on chosen question
        
        Args:
            articles: List of article dictionaries
            output_path: Path to save the results to (and load from)
            batch_size: Number of articles to process in parallel
            regenerate: If True, ignore existing results and start fresh.

        Returns:
            List of results containing the generated questions with all processing completed
        """
        logger.info("Starting complete processing pipeline...")
        
        # Step 1: Generate questions
        logger.info("Step 1: Generating questions...")
        results = await self.generate_questions(articles, output_path, batch_size, regenerate)
        
        # Step 2: Extract individual questions
        logger.info("Step 2: Extracting individual questions...")
        results = await self.extract_individual_questions(output_path, batch_size)
        
        # Step 3: Validate individual questions
        if self.validate_questions and self.choose_engine:
            logger.info("Step 3: Validating individual questions...")
            results = await self.validate_individual_questions(output_path, batch_size)
        
        # Step 4: Choose best questions (if enabled and valid questions exist)
        if self.choose_best and self.choose_engine:
            logger.info("Step 4: Choosing best questions from valid ones...")
            results = await self.choose_best_questions(output_path, batch_size)
        
        # Step 5: Check leakage (if enabled)
        if self.check_leakage and self.leakage_engine:
            logger.info("Step 5: Checking for leakage...")
            results = await self.check_questions_leakage(output_path, batch_size)
        
        logger.info("Complete processing pipeline finished.")
        return results

    def _extract_questions_from_text(self, generated_text: str) -> List[str]:
        """
        Extract individual questions from generated questions text.
        
        Args:
            generated_text: The full generated questions text
            
        Returns:
            List of individual question strings (in XML format)
        """
        import re
        
        if not generated_text:
            return []
        
        questions = []
        
        if self.use_freeq:
            # For free-form questions, look for <q1>, <q2>, etc. blocks
            for q_num in range(1, self.num_questions_per_article + 1):
                pattern = f'<q{q_num}>(.*?)</q{q_num}>'
                match = re.search(pattern, generated_text, re.DOTALL)
                if match:
                    question_content = match.group(1).strip()
                    full_question = f"<q1>\n{question_content}\n</q1>"
                    questions.append(full_question)
                else:
                    # Try alternative patterns
                    alt_pattern = f'<question_id>{q_num-1}</question_id>(.*?)(?=<question_id>|$)'
                    alt_match = re.search(alt_pattern, generated_text, re.DOTALL)
                    if alt_match:
                        question_content = alt_match.group(1).strip()
                        # Wrap in proper XML structure
                        question_content = f"<question_id>{q_num-1}</question_id>\n{question_content}"
                        full_question = f"<q1>\n{question_content}\n</q1>"
                        questions.append(full_question)
        else:
            # For MCQ questions, look for JSON blocks
            json_pattern = r'```json\s*(.*?)\s*```'
            json_matches = re.findall(json_pattern, generated_text, re.DOTALL)
            
            for match in json_matches[:self.num_questions_per_article]:
                questions.append(match.strip())
        
        # logger.info(f"Extracted {len(questions)} questions from generated text")
        return questions

    def extract_final_question(self, choose_best_output: str) -> str:
        """
        Extract the final question content from choose_best output.
        Always takes the last match if multiple questions are found.
        
        Args:
            choose_best_output: The output from choose_best processing
            
        Returns:
            The extracted question content in XML format, or empty string if not found
        """
        import re
        
        if not choose_best_output:
            return ""
        
        # Check for "NO GOOD QUESTION" case
        if "NO GOOD QUESTION" in choose_best_output.upper():
            return ""
        
        # # Extract all q1 blocks using regex and take the last one
        # # This pattern captures everything between <q1> and </q1> tags
        # pattern = r'<q1>(.*?)</q1>'
        # matches = re.findall(pattern, choose_best_output, re.DOTALL)
        
        # if matches:
        #     # Return the last (most recent) complete q1 block including the tags
        #     return f"<q1>{matches[-1]}</q1>"
        
        
        # # Fallback: try to extract without requiring exact q1 tags
        # # Look for the question structure with question_id, question_title, etc.
        # question_pattern = r'<question_id>.*?</question_id>.*?<question_title>.*?</question_title>.*?<background>.*?</background>.*?<resolution_criteria>.*?</resolution_criteria>.*?<answer>.*?</answer>.*?<answer_type>.*?</answer_type>'
        # matches = re.findall(question_pattern, choose_best_output, re.DOTALL)
        
        # if matches:
        #     content = matches[-1]
        #     if not content.startswith('<q1>'):
        #         content = f"<q1>\n{content}\n</q1>"
        #     return content
        
        
        # Fallback: For each tag, find the last opening tag and extract from there to its closing tag
        def extract_last_tag_block(text, tag):
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            last_open = text.rfind(open_tag)
            if last_open == -1:
                return ""
            start = last_open
            end = text.find(close_tag, start)
            if end == -1:
                return ""
            end += len(close_tag)
            return text[start:end]

        tags = [
            "question_id",
            "question_title",
            "background",
            "resolution_criteria",
            "answer",
            "answer_type"
        ]
        
        blocks = []
        for tag in tags:
            block = extract_last_tag_block(choose_best_output, tag)
            if not block:
                break
            blocks.append(block)

        if len(blocks) == len(tags):
            content = "\n".join(blocks)
            if not content.startswith('<q1>'):
                content = f"<q1>\n{content}\n</q1>"
            fallback_matches = [content]
        else:
            fallback_matches = []
            
        if fallback_matches:
            # Take the last match and wrap in q1 tags if not already present
            content = fallback_matches[-1]
            if not content.startswith('<q1>'):
                content = f"<q1>\n{content}\n</q1>"
            return content
        
        # If no valid question structure found, return empty string
        # logger.warning("Could not extract valid question from choose_best output")
        return ""
        
    def save_results(self, results: List[Dict], output_path: str) -> None:
        """
        Save the final generated questions to a JSONL file. (Mainly for consistency)
        This is now a full rewrite operation, used only if explicitly called.
        """
        try:
            # Ensure output directory exists (only if output_path has a directory component)
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only if there's a directory component
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
                    
            logger.info(f"Final save completed. Saved {len(results)} results to {output_path}")
        except Exception as e:
            logger.error(f"Error during final save to {output_path}: {e}")

    def _extract_article_content(self, article: Dict) -> str:
        """Extract relevant content from article for leakage checking."""
        content_parts = []
        
        if article.get('title'):
            content_parts.append(f"Title: {article['title']}")
        
        if article.get('description'):
            content_parts.append(f"Description: {article['description']}")
        
        if article.get('maintext'):
            content_parts.append(f"Content: {article['maintext']}")
        
        return "\n\n".join(content_parts)

    def _prepare_leakage_check_prompts(self, generated_texts: List[str], articles: List[Dict]) -> List[str]:
        """
        Prepare batch prompts for leakage checking.
        
        Args:
            generated_texts: List of generated question texts
            articles: List of corresponding article dictionaries
            
        Returns:
            List of prompts for batch leakage checking
        """
        prompts = []
        
        for generated_text, article in zip(generated_texts, articles):
            if not generated_text or generated_text.startswith("ERROR:") or len(generated_text) < 10:
                # Skip invalid texts, add placeholder
                prompts.append("SKIP")
                continue
                
            article_content = self._extract_article_content(article)
            
            if self.use_freeq:
                prompt = self._format_freeq_leakage_prompt(generated_text, article_content)
            else:
                prompt = self._format_mcq_leakage_prompt(generated_text, article_content)
            
            prompts.append(prompt)
        
        return prompts
    

    
    def _prepare_valid_questions_text(self, result: Dict) -> str:
        """
        Prepare text containing only valid questions for an article.
        
        Args:
            result: Article result dictionary with individual questions and validation
            
        Returns:
            Text containing only valid questions formatted for choose_best
        """
        valid_questions = []
        
        for q_num in range(1, self.num_questions_per_article + 1):
            if result.get(f"q{q_num}_valid", 0) == 1:
                question_text = result.get(f"q{q_num}", "")
                if question_text:
                    valid_questions.append(question_text)
        
        if not valid_questions:
            return ""
        
        # Combine valid questions into a single text
        if self.use_freeq:
            # For free-form questions, combine them with proper numbering
            combined_text = ""
            for i, question in enumerate(valid_questions, 1):
                # Ensure each question is wrapped in q tags
                if not question.startswith("<q"):
                    question = f"<q{i}>\n{question}\n</q{i}>"
                combined_text += question + "\n\n"
            return combined_text.strip()
        else:
            # For MCQ questions, combine JSON blocks
            return "\n\n".join(valid_questions)

    def _prepare_choose_best_prompts(self, generated_texts: List[str], articles: List[Dict]) -> List[str]:
        """
        Prepare batch prompts for choosing the best question from the generated questions.
        """
        prompts = []
        for generated_text, article in zip(generated_texts, articles):
            prompt = self._format_choose_best_prompt(generated_text, article)
            prompts.append(prompt)
        return prompts
    
    def _format_choose_best_prompt(self, questions_text: str, article: Dict) -> str:
        """Format prompt for choosing the best question from the generated questions."""
        
        prompt = ""
        prompt += f"""
**Task:** You will be provided with a list of questions (possibly with size 1). Your job is to choose the best question from the list based on the following criteria or end your response with "NO GOOD QUESTION" if none of the questions meet the criteria.

**Instructions:**
GO THROUGH EACH QUESTION ONE BY ONE AND ANALYZE IT FOR THE FOLLOWING:
1. **Valid for forecasting**: Check if the WHOLE QUESTION is stated in a forward-looking manner. FROM THE PERSPECTIVE OF THE START DATE TO THE RESOLUTION DATE MENTIONED IN THE QUESTION, CHECK IF IT IS A VALID FORECASTING QUESTION. IF THE TIME HORIZON (START DATE TO RESOLUTION DATE) IN THE QUESTION IS AT LEAST A SINGLE DAY, THEN THE QUESTION SHOULD BE CONSIDERED VALID FOR FORECASTING. Go through each segment of the question (question title, background, resolution criteria) and check if each of them is valid and forward-looking.
2. **Tense**: The question SHOULD NOT BE STATED IN PAST TENSE. If the question covers an event, it should not imply as if the outcome of the event has already happened or occurred.
3. **Single Correct Answer**: ANALYZE WHETHER THE QUESTION CAN HAVE MULTIPLE OUTCOMES OR RIGHT ANSWERS. IF SO, THE QUESTION FAILS THIS CRITERIA. OTHERWISE, ENSURE THAT THE PROVIDED ANSWER IS THE SOLE CORRECT ANSWER TO THE QUESTION. IT SHOULD NOT BE THE CASE THAT THE QUESTION CAN HAVE MULTIPLE (DISTINCT) CORRECT ANSWERS.
4. **Impact**: How many people will the outcome of the question be relevant or interesting to? Consider on the basis of significant downstream impact or enabling meaningful action.
5. **Not Binary/Multiple Choice**: Question SHOULD NOT BE BINARY (yes/no, either ABC or XYZ, etc.) OR MULTIPLE CHOICE (SELECT FROM A LIST OF OPTIONS). It should be free-form (string -- name, date, place, etc.) or numerical (number, percentage, etc.). 
6. **Understandable**: THe question as a whole (title, background, resolution criteria) should have sufficient details to understand the premise of the question. Every detail should be crystal clear and the question should not be under or over specified. 
7. **Definite Answer**: EXTRACT THE ACTUAL ANSWER TO THE QUESTION PROVIDED IN ITS <answer> </answer> TAG. The extracted answer should be short, definite, well-defined and not uncertain or vague. It SHOULD NOT BE A PHRASE OR A RANGE like "between XYZ and ABC" or "above XYZ" or "below PQR". 



ANALYZE EACH QUESTION BASED ON THE ABOVE CRITERIA ONE BY ONE AND CHOOSE THE ONE WHICH PASSES ALL THE ABOVE CRITERIA. IF MULTIPLE QUESTIONS SATISFY THE CRITERIA, CHOOSE THE ONE WHICH WILL HAVE THE HIGHEST IMPACT (AFFECTS OR IS RELEVANT TO THE MOST NUMBER OF PEOPLE). IF NO QUESTION MEETS THE CRITERIA, RETURN "NO GOOD QUESTION FOUND". OTHERWISE, RETURN THE BEST QUESTION IN THE SAME FORMAT AS THE INPUT. 

**Generated Questions:**
{questions_text}

**Output Format:**
<q1>
<question_id>0</question_id>
<question_title>[ORIGINAL Title of the best question]</question_title>
<background>[ORIGINAL Background of the best question]</background>
<resolution_criteria>
<ul>
    <li> <b>Source of Truth</b>: [ORIGINAL Source of Truth of the best question] </li>
    <li> <b>Resolution Date</b>: [ORIGINAL Date of the best question] </li>
    <li> <b>Accepted Answer Format</b>: [ORIGINAL Accepted Answer Format of the best question] </li>
</ul>
</resolution_criteria>
<answer>[ORIGINAL Answer of the best question]</answer>
<answer_type>[ORIGINAL Answer Type of the best question]</answer_type>
</q1>
""" # WHICH IS FREE-FORM, VALID, HAS CONCLUSIVE ANSWER 
        return prompt

    def _format_question_validation_prompt(self, questions_text: str, article: Dict) -> str:
        """Format prompt for validating the question."""
        source_article = f"Title: {article.get('title', '')}\n\n"
        
        if 'description' in article and article['description']:
            source_article += f"Description: {article['description']}\n\n"
            
        if 'maintext' in article and article['maintext']:
            source_article += f"Content: {article['maintext']}\n\n"
            
        return f"""
**Task:** You will be provided with a news article and a question WHOSE ANSWER IS SUPPOSED TO BE BASED ON THE ARTICLE. Your job is to validate whether the answer to the question is valid by being faithful to the article (content, title, or description).

GO THROUGH EACH SEGMENT OF THE QUESTION ONE BY ONE (TITLE, BACKGROUND, RESOLUTION CRITERIA, ANSWER) TO UNDERSTAND THE WHOLE QUESTION. THEN CHECK EACH OF THE FOLLOWING CRITERIA: 

1. **Tense and Details**: FIRST CHECK WHETHER THE QUESTION IS NOT UNDER SPECIFIED OR STATED IN PAST TENSE. IT IS FINE IF THE QUESTION IS STATED IN CURRENT OR FUTURE TENSE. 
2. **Definite resolution of the answer by the article**: CHECK WHETHER THE ANSWER TO THE QUESTION IS SOUND, CLEAR AND PRESENT IN OR CAN BE DERIVED FROM THE ARTICLE. THE ARTICLE SHOULD RESOLVE THE ANSWER DEFINITELY AND IN AN INDISPUTABLE MANNER (WITHOUT ANY AMBIGUITY). THIS IS THE MOST IMPORTANT CRITERIA.
3. **Well-defined Answer**: The answer to the question should be short (NOT MORE THAN 3 WORDS). IT SHOULD NOT BE A PHRASE AND SHOULD BE SOMETHING WHICH IS CONCRETE, SPECIFIC AND WELL-DEFINED.
4. **Non-Numeric**: THE *ANSWER TYPE* SHOULD NOT BE NUMERIC LIKE A PERCENTAGE, INTEGER, DECIMAL, OR A RANGE.
5. **Single Correct Answer**: ANALYZE WHETHER THE QUESTION CAN HAVE MULTIPLE OUTCOMES OR RIGHT ANSWERS. IF SO, THE QUESTION FAILS THIS CRITERIA. OTHERWISE, ENSURE THAT THE PROVIDED ANSWER IS THE SOLE CORRECT ANSWER TO THE QUESTION. IT SHOULD NOT BE THE CASE THAT THE QUESTION CAN HAVE MULTIPLE (DISTINCT) CORRECT ANSWERS.

If ALL the above criteria pass (question is stated as required, answer to the whole question is valid, well-defined, and it is the only correct answer to the question), ONLY THENreturn <answer>1</answer>. Otherwise, return <answer>0</answer>. ALWAYS END YOUR RESPONSE IN <answer> </answer> tags.

**Article:**
{source_article}

**Question:**
{questions_text}

**Output Format:**
<answer>0/1</answer>
""" 


    def _format_freeq_leakage_prompt(self, questions_text: str, article_content: str) -> str:
        """Format prompt for checking free-form question leakage."""
        return f"""
**Task:** You will be provided with a forecasting question. Your job is to ANALYZE whether the question's answer has obviously leaked in the content of the question. The question will have multiple segments -- question title, background, resolution criteria. EXCEPT THE QUESTION TITLE, GO THROUGH EACH SEGMENT STEP BY STEP and check if any part DIRECTLY leaks the actual answer. If leakage is found, ONLY THEN rephrase the problematic parts appropriately to remove the answer while maintaining the question's integrity and focus. DO NOT CHANGE ANY PART OF THE QUESTION UNNECESSARILY. 

USE THE SAME XML FORMAT IN YOUR RESPONSE AS IS IN THE INPUT.

**Generated Question:**
{questions_text}

**Instructions:**
1. **Keep the title unchanged**: DO NOT MAKE ANY CHANGE TO THE QUESTION TITLE.
2. **Keep the start date in the background unchanged**: DO NOT MAKE ANY CHANGE TO THE QUESTION'S START DATE IN THE BACKGROUND.
3. **Identify the answer**: First, extract the actual answer from the XML tags for the current question being processed.
4. **Identify Leakage**: Keeping the extracted answer in mind, check if the  background, or resolution criteria (each of them -- source of truth, resolution date, accepted answer format) contain information that reveals the answer.
5. **Types of leakage which can be ignored**: The following types of leakage are fine and don't need to be rephrased:
   - If the outcome (actual answer) of the question is binary (yes/no, either ABC or XYZ, etc.), then NO NEED TO CHANGE ANYTHING ANYWHERE.
   - If the resolution criteria is based on a list of specific options, then NO NEED TO CHANGE ANYTHING IN ANY SEGMENT (BACKGROUND, RESOLUTION CRITERIA, etc.). For example, if the accepted answer format states "answer must be either .." OR "answer must be one of the following terms..", then NO NEED TO CHANGE ANYTHING ANYWHERE.
6. **Types of Leakage to Check:** ONLY CONSIDER THE FOLLOWING KIND OF LEAKAGE:
   - DIRECT MENTIONS of the answer (either in word or number form) or part of the answer in the question/background/resolution
   - References to specific outcomes that ARE CLOSE TO (OR REVEAL)THE ACTUAL ANSWER
7. **Rephrase Strategy**: If leakage is found, rephrase the problematic part while:
   - Keeping the question's core intent
   - Maintaining forecasting nature
   - Preserving necessary context
   - Making the answer UNOBVIOUS by replacing with a FAKE ANSWER (FAKE NAME, DATE, NUMBER, PERCENTAGE, etc.) WHICH IS GENERIC AND NOT CLOSE TO THE ACTUAL ANSWER.
   - The rephrased part should not contain any information that is part of the actual answer. Neither should it indirectly hint or reveal the answer. 
8. **Check Accepted Answer Format**: IF THERE IS ANY EXAMPLE MENTIONED IN ACCEPTED ANSWER FORMAT ("e.g..."), MAKE SURE THE EXAMPLE IS GENERIC AND AS FAR AWAY FROM THE ACTUAL ANSWER AS POSSIBLE. DO NOT INCLUDE AN EXAMPLE IF NOT MENTIONED ALREADY. 
9. **Do not change the answer**: Do not change the actual answer to the question. 
10. **Do not change the answer_type**: DO NOT MAKE ANY CHANGE TO the answer_type.
11. **Each segment should be checked independently**: Go through each segment of the whole question one by one. Everything from the title of the question to the background information to the resolution criteria should be checked independently with reference to the answer of the question. In the resolution criteria, go through each <li> step by step. Do not change the other segments when rephrasing a problematic segment.
12. **Do not change anything unless leakage is found**: DO NOT UNNECESSARILY CHANGE ANY PART OF THE QUESTION UNLESS LEAKAGE IS FOUND.

IT IS ALSO POSSIBLE THAT MULTIPLE PARTS OF THE QUESTION HAVE LEAKAGE. YOU SHOULD CHECK EACH OF THEM INDEPENDENTLY AND ONLY IF LEAKAGE IS FOUND, REPHRASE THE PROBLEMATIC PARTS. DO NOT OVER-ANALYZE.

During your analysis, you should:
- Go through EACH SEGMENT OF THE QUESTION STEP BY STEP INDEPENDENTLY. First <background> and then inside <resolution_criteria>. Under the resolution criteria, go through the source of truth, resolution date, accepted answer format (each of them is a <li> tag) one by one. For each such segment, do the following:
    - Compare the content in the current segment with the actual answer. If ANY PART OF THE ANSWER is mentioned in the current segment, then consider that as a leakage UNLESS THE ACCEPTED ANSWER FORMAT IS BINARY (yes/no, either ABC or XYZ, etc.) OR A LIST OF SPECIFIC OPTIONS.
    - IF THE CURRENT SEGMENT IS BACKGROUND, DO NOT CHANGE THE QUESTION START DATE.
    - If the current segment is accepted answer format and there is a SPECIFIC EXAMPLE MENTIONED in it ("e.g. XYZ") which is close to the actual answer, then consider that as a leakage.
    - If leakage is found in the current segment, mention "Leakage found -- {{reason for leakage}}". Form the segment with the problematic parts rephrased and mention it as "Replacement -- {{rephrased_text}}." THE REPHRASED TEXT SHOULD BE AS FAR AWAY FROM THE ACTUAL ANSWER AS POSSIBLE. It should now be present in the final output (instead of the original text).
    - Otherwise, mention "No leakage found". In your final output after you finish the analysis, return this segment UNCHANGED.
    - These outputs should be in the same format as the original input. 
- Return the actual answer unchanged in the <answer> tag in your final output.
- Skip any other segments (question title, answer_type, etc.) in your analysis and output them unchanged (verbatim) in the final output.

Output your analysis step by step, and then end your response with the CORRECTED question in THE SAME XML FORMAT AS THE ORIGINAL. 

**Output Format**:
{{ analysis }}

<q1>
<question_id>0</question_id>
<question_title>[UNCHANGED Question Title]</question_title>
<background>[Corrected Background]</background>
<resolution_criteria>
<ul>
    <li> [UNCHANGED Question Start Date] [Corrected Source of Truth] </li>
    <li> [UNCHANGED Resolution Date] </li>
    <li> [Corrected Accepted Answer Format] </li>
</ul>
</resolution_criteria>
<answer>[UNCHANGED Answer]</answer>
<answer_type>[UNCHANGED Answer Type]</answer_type>
</q1>
"""
    
    def _format_mcq_leakage_prompt(self, questions_text: str, article_content: str) -> str:
        """Format prompt for checking MCQ leakage."""
        return f"""
**Task:** Analyze the provided multiple choice forecasting questions and check if they leak the actual answer. If leakage is found, rephrase the problematic content to remove the answer while maintaining the question's integrity.

**Generated Questions:**
{questions_text}

**Instructions:**
1. **Identify Leakage**: Check if the question title, background, or options contain information that directly or indirectly reveals the correct answer.
2. **Types of Leakage to Check:**
   - Direct mentions of the answer in question/background
   - Hints that point to the specific correct option
   - Context that makes the correct answer obvious
   - Biased language that favors the correct option
   - Options that are clearly wrong or impossible
3. **Rephrase Strategy**: If leakage is found, rephrase the problematic part while:
   - Keeping the question's core intent
   - Maintaining forecasting nature
   - Preserving necessary context
   - Making all options equally plausible
   - Not making the correct answer obvious

**Output Format:**
If no leakage is found, return the original questions unchanged.
If leakage is found, return the questions with the problematic parts rephrased. USE THE SAME XML FORMAT AS IN THE INPUT.

**Example of Good Rephrasing:**
Original: "Who will win the 2016 Nobel Prize in Literature, with Bob Dylan as the leading candidate?"
Rephrased: "Who will win the 2016 Nobel Prize in Literature?"

**Example of Bad Rephrasing:**
Original: "What will be the final Brexit vote result?"
Rephrased: "What will be the final Brexit vote result, considering the Leave campaign's strong position?"

Return the corrected questions in THE SAME XML FORMAT AS THE ORIGINAL.
""" 