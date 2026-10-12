"""Shared forecasting criteria. Wording preserved from the generation prompt."""

QUESTION_CRITERIA = (
    """**Fresh, consequential event**: The questions should be about high impact, recent (within a week of the publish date) events reported in the document provided. The question require predicting an important aspect of the event, not a small detail.""",
    """**Free-form, unambiguous without the document**: The questions created should be free-form and have a unique, specific answer (a single phrase, up to a few words) which is not numeric. The questions should be unambiguous and not refer to the document in its wordings as the document will not be provided to the predictor. Avoid creating binary questions (yes/no, either/or) or questions with a list of specific options (multiple choice), though the question implicitly having a finite set of possible answers is not a problem. It is okay if the answer can have paraphrases, our grading will handle it, but there should be only one semantically distinct possible answer.""",
    """**Grounded in the document**: The question should be based on information explicitly provided on the document, and have an indisputable ground-truth. This means the ground-truth should be not be uncertain or speculative anymore once given the document. It should be an event that has already been observed and reported in the document. Make sure that the question does not add any information not supported by the document just to force uniqueness/specificity.""",
    """**No references to document or future information**: DO NOT refer to the specific document, such as by saying "in the document". The forecaster does not have access to the document, its metadata or any information beyond the document publish date. Do not leak any information that became available within 1 week of the publish date in the question itself.""",
    """**Interesting forecasting question**. Above all, the question should be an interesting, natural forecasting question, that more than 10 people (ideally as many) would have been curious about even before the event happened. It should not be based on some post-hoc unpredictable detail, while still being hard to predict and requiring reasoning under uncertainty. It is okay if it is conditional, or premised on some other event you didn't know about yet.""",
)

ANSWER_CRITERIA = (
    """**Faithfulness to document**: The answer should be based on information explicitly stated in the document, and not inferences, speculations, or your own knowledge. Do not add any vagueness or extra details on your own. The answer MUST be something that has already happened according to the document, not something that will happen in the future according to the document. Note that our grading will handle paraphrasing and abbreviations, so do not worry about the answer provided being identical to the wording used in the document, making it clearer for e.g. expanding abbrevations is fine.""",
    """**Non-Numeric, Unique free-form**: The answer should not be a number or a percentage. It can be a word, phrase, location, etc. but short enough that the answer is unique (beyond paraphrases of the same semantic meaning).""",
    """**Specificity**: The answer should be specific enough to be unambiguous and precisely resolvable. It should not be vague or too general. The answer should be indisputable and not be open to multiple interpretations."""
)

BACKGROUND_CRITERIA = (
    """**Necessary Context, Premise**: The answerer does not have access to the document. So if necessary, you can include the minimal context required to unambiguously understand the question, or any premise that the question assumes, in the background field. Do not give extra details of the event from the document in the background. Define any uncommon/unpopular terms in the question title here.""",
    """**Start date**: The start_date is the date when the question was formed and its premise makes sense. Provide that date in the start_date field. Then, the background, resolution criteria and question title should only contain information known before the start_date. Ideally start_date should be significantly before the resolution_date to give a wide enough window for the question to be forecasted during.""",
    """**Should not leak answer**: Do not mention the answer, or any hints towards the answer in the question background. It should only contain the premise needed to understand the question. Do not mention any examples, or "such as" statements as they can leak or hint toward the final answer.""",
    """***Write from the perspective of the start_date***: The background, resolution criteria and question title should be written from the perspective of the start_date, not including any information after it.""",
)

RESOLUTION_CRITERIA = (
    """**Precise resolution**: State the EXACT conditions by which the outcome will be judged. Include the criteria which determines unambiguously how the question will be resolved. """,
    """**Date and Source of Resolution**: If necessary for precise unambiguous resolution, state the date and the source by which the question will be resolved in the resolution criteria, because the resolution_date you output will not be shown to the predictor. For example, resolution dates such as "by/in {month_name}, {year}" and potential source(s) of resolution such as "based on {news source}", "reports from {official/reliable authority}", etc. are fine, this one does not have to be too specific as it is shown to the predictor. The actual resolution_date field which is not shown to predictor must be specific, in YYYY-MM-DD format, which is the earliest you think the answer to this question would have become known, based on the content and publish date of the article (if content indicates it was resolved earlier, please set it to earlier). In fact, it should not leak or hint any information that would not be expected apriori by the start date. However, for some questions, where the answer can change based on the time about which they are asked, or the sources that are to be trusted, it is necessary to include. In a nutshell, the resolution criteria should make the validity of the answer indisputable.""",
    """**Format of Answer**: The resolution criteria should state what format the final answer should be, for e.g. is it a name of a person/organization etc. Note that our grading takes care of paraphrases, so that you don't need to worry about. DO NOT provide any example answers, as that can leak or hint towards the actual answer.""",
    """**Do not hint/leak the answer**: The resolution criteria must not directly help answer the forecasting question, nor provide any hints toward the answer. It should never mention the answer itself, or any information from the document that helps get to it. It should not contain any information that became known after the start_date.""",
)

CRITERIA_GROUPS = (
    ("A", "Title Question Guidelines", QUESTION_CRITERIA),
    ("B", "Answer Guidelines", ANSWER_CRITERIA),
    ("C", "Background Guidelines", BACKGROUND_CRITERIA),
    ("D", "Resolution Criteria", RESOLUTION_CRITERIA),
)


def format_criteria():
    return "\n\n".join(
        f"**{letter}. {heading}**\n" + "\n".join(
            f"{letter}{number}. {text}" for number, text in enumerate(criteria, 1)
        )
        for letter, heading, criteria in CRITERIA_GROUPS
    )
