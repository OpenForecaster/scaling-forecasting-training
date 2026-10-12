"""Generate candidates and report violations of the shared criteria."""

from typing import Dict

from .criteria import format_criteria
from .verification import format_verification_prompt, parse_verification


ONE_SHOT_EXAMPLE = """**Example Format**:
<question id="1">
<question_title>Who will win the Nobel Prize in Literature in 2016?</question_title>
<start_date>2016-01-10</start_date>
<background>The Nobel Prize in Literature is awarded annually by the Swedish Academy to authors for their outstanding contributions to literature.</background>
<resolution_date>2016-10-13</resolution_date>
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
</question>"""


def question_output_format(num_questions):
    return "\n".join(
        f"""<question id="{number}">
<question_title>[Question {number}]</question_title>
<start_date>[YYYY-MM-DD]</start_date>
<background>[Background {number}, if needed]</background>
<resolution_date>[YYYY-MM-DD]</resolution_date>
<resolution_criteria>[Resolution Criteria {number}]</resolution_criteria>
<answer>[Answer {number}]</answer>
</question>"""
        for number in range(1, num_questions + 1)
    )

class FeedbackQuestionGenerator:
    def __init__(self, inference_engine, *, num_questions=1, criteria_engine=None, fewshot=True):
        self.inference_engine = inference_engine
        self.criteria_engine = criteria_engine or inference_engine
        self.num_questions_per_article = num_questions
        self.fewshot = fewshot

    def format_prompt(self, article):
        return self._format_freeq_prompt(article)

    def _format_freeq_prompt(self, article: Dict) -> str:
        """Format prompt for free-form short answer questions using XML format."""
        source_document = f"Title: {article.get('title', '')}\n\n"
        if article.get("description"):
            source_document += f"Description: {article['description']}\n\n"
        source_document += f"Published Date: {article['date_publish']}\n\nContent: {article['maintext']}\n"

        output_format = question_output_format(self.num_questions_per_article)

        prompt = f"""
Based on the provided document, generate between 1 and {self.num_questions_per_article} high-quality, natural, distinct forecasting questions which have a short answer, using the XML format specified below.
Each forecasting question should be posed in a way to predict an important future event. Note that we will use these questions for backtesting, so the predictor will have a knowledge cutoff and access to information only before the document is published and thus no access to the document itself. The answer itself should thus be resolved undisputably by the document.

{ONE_SHOT_EXAMPLE if self.fewshot else ""}

The question should follow the structured guidelines below.

### **Guidelines for Creating Short Answer Forecasting Questions**

{format_criteria()}

Generate between 1 and {self.num_questions_per_article} high-quality, DIVERSE short answer forecasting questions based on the provided document. Use one XML block per question, with consecutive IDs: <question id="1">, <question id="2">, etc., each closed with </question>. Within each block, output question_title, start_date, background, resolution_date, resolution_criteria, and answer in that order. Use YYYY-MM-DD for both dates. The background may be empty if no context is needed. DO NOT INCLUDE ANY ANALYSIS, RANKING, OR ADDITIONAL COMMENTARY.

Document:
{source_document}

**Required Output Format**:
{output_format}
"""


        return prompt


    def format_verification_prompt(self, question, article):
        source_document = f"Title: {article.get('title', '')}\n\n"
        if article.get("description"):
            source_document += f"Description: {article['description']}\n\n"
        source_document += f"Published Date: {article['date_publish']}\n\nContent: {article['maintext']}\n"
        if article.get("date_modify"):
            source_document += f"Modified Date: {article['date_modify']}\n"
        return format_verification_prompt(question, source_document)

    async def run_pipeline(self, articles, output_path, concurrency=64, revise=False):
        """Run independent document episodes with bounded concurrency."""
        import asyncio
        from pathlib import Path
        from .io import read_jsonl, write_jsonl, append_jsonl
        from .xml_utils import parse_generated_questions
        from .revision import revise_question

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {row['url']: row for row in read_jsonl(path)} if path.exists() else {}
        rows = [saved.get(article['url'], dict(article)) for article in articles]
        slots = asyncio.Semaphore(concurrency)

        async def episode(row):
            async with slots:
                try:
                    row.pop('episode_error', None)
                    if row.get('generation_error'):
                        return
                    if "generated_questions" not in row:
                        prompt = self.format_prompt(row)
                        try:
                            response = (await self.inference_engine.generate([prompt]))[0]["response"]
                        except Exception as error:
                            row.update(generation_error=f'{type(error).__name__}: {error}', candidates=[])
                            append_jsonl(path, row)
                            return
                        row.update(generation_prompt=prompt, generated_questions=response)
                        try:
                            row["candidates"] = parse_generated_questions(response, self.num_questions_per_article)
                        except ValueError as error:
                            row.update(generation_error=str(error), candidates=[])
                            append_jsonl(path, row)
                            return
                        append_jsonl(path, row)
                    # Sequential within an episode: at most one request per worker.
                    for candidate in row["candidates"]:
                        if "verification" in candidate or candidate.get('verification_error'):
                            continue
                        prompt = self.format_verification_prompt(candidate["question"], row)
                        try:
                            response = (await self.criteria_engine.generate([prompt]))[0]["response"]
                            candidate.update(verification=parse_verification(response),
                                             verification_prompt=prompt, verification_response=response)
                        except Exception as error:
                            candidate['verification_error'] = f'{type(error).__name__}: {error}'
                        append_jsonl(path, row)
                    if revise:
                        try:
                            await revise_question(row, self.inference_engine, self.criteria_engine,
                                                  on_update=lambda: append_jsonl(path, row))
                        except Exception as error:
                            row['revision_error'] = f'{type(error).__name__}: {error}'
                except Exception as error:
                    row['episode_error'] = f'{type(error).__name__}: {error}'
                finally:
                    append_jsonl(path, row)

        await asyncio.gather(*(episode(row) for row in rows))
        # Compact once after the run, including any failed episodes for resume.
        write_jsonl(path, rows)
        return rows
