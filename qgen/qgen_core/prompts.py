"""Guidelines and output format for free-form question generation."""

FREEFORM_GUIDELINES = """**Question**
- Focus on consequential events reported within a week of the publication date. Ask about an important aspect, rather than a small detail.
- Ask a natural forecasting question that people would have been curious about before the event happened. Avoid post-hoc, unpredictable details. Conditional questions are fine.
- The question must be understandable without the document and have one semantically distinct, non-numeric answer. Do not give answer options or ask binary questions.
- The document must report an observed event that resolves the answer indisputably. Do not invent details to force uniqueness or specificity.
- Do not refer to the document or its metadata. Do not disclose information that became available within a week of publication.

**Answer**
- Use information explicitly stated in the document, not speculation, inference, or outside knowledge. The event must already have happened according to the document.
- Give a specific word or short phrase, up to a few words, with one unambiguous meaning. Paraphrases and expanded abbreviations are fine; verbatim wording is unnecessary.
- Do not use numbers, percentages, vague terms, or ranges.

**Background and start date**
- Include only the context or premise needed to understand the question. Explain uncommon terms if necessary. The background may be empty.
- Put the date when the premise makes sense in start_date, separately from the background. Prefer a wide forecasting window before resolution.
- Write the title, background, and resolution criteria from the perspective of start_date, using only information available by then.
- Do not hint at the answer or provide example answers, including "such as" statements.

**Resolution**
- State precise conditions that make the answer unambiguous, and the expected answer format.
- Include dates and sources in the resolution criteria when needed for precise resolution. Avoid details that would not have been expected at start_date.
- Put the earliest date the answer became publicly known in resolution_date, using the document's content and publication date. This field is not shown to the forecaster, so include any necessary time bounds in the visible question or resolution criteria as well.
- Both dates must be valid YYYY-MM-DD dates, with start_date earlier than resolution_date.
- Do not include the answer, example answers, or hints in the resolution criteria.
- Use string (name), string (location), or another suitable string category for answer_type."""

FREEFORM_EXAMPLE = """**Example**:
<q1>
<question_id>0</question_id>
<question_title>Who will win the Nobel Prize in Literature in 2016?</question_title>
<start_date>2016-01-10</start_date>
<background>The Nobel Prize in Literature is awarded annually by the Swedish Academy for outstanding contributions to literature.</background>
<resolution_date>2016-10-13</resolution_date>
<resolution_criteria>Resolve to the author named by the Swedish Academy in its official 2016 Nobel Prize in Literature announcement. Answer with the author's full name.</resolution_criteria>
<answer>Bob Dylan</answer>
<answer_type>string (name)</answer_type>
</q1>"""


def freeform_output_format(num_questions):
    return "\n\n".join(
        f"""<q{number}>
<question_id>{number - 1}</question_id>
<question_title>[Question {number}]</question_title>
<start_date>[YYYY-MM-DD]</start_date>
<background>[Minimal context, if needed]</background>
<resolution_date>[YYYY-MM-DD]</resolution_date>
<resolution_criteria>[Precise resolution conditions and answer format]</resolution_criteria>
<answer>[Short answer]</answer>
<answer_type>[string category]</answer_type>
</q{number}>"""
        for number in range(1, num_questions + 1)
    )
