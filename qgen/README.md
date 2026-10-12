# QGen - Forecasting Question Generation Pipeline

Comprehensive toolkit for generating, validating, and processing forecasting questions from news articles.

## 📁 Architecture

```
qgen/
├── run_pipeline.py         # End-to-end pipeline (Recommended)
├── qgen_core/              # Question generation logic
├── filters/                # Leakage detection and removal
├── processors/             # Date processing
├── inference/              # OpenRouter & VLLM engines
├── utils/                  # Reusable utilities
├── scripts/                # Individual CLI tools
└── jobs/                   # HTCondor job submission
```

## 🚀 Pipeline Usage

### Basic Usage

**Quality defaults always enabled:** Free-form questions, leakage checking, best selection, validation, date updates.

```bash
python run_pipeline.py \
    --article_path articles.jsonl \
    --output_dir ./output \
    --use_openrouter
```

### Full Example

```bash
python run_pipeline.py \
    --article_path /path/to/articles.jsonl \
    --output_dir /path/to/output \
    --use_openrouter \
    --creator_model deepseek/deepseek-chat-v3-0324 \
    --selector_model meta-llama/llama-4-maverick \
    --num_q_per_article 3 \
    --first_date 2025-01-01 \
    --explicit_filter \
    --seed 42
```

### What It Does

1. Generates free-form questions with quality checks
2. Converts to standardized format
3. Extracts and updates resolution/start dates
4. Filters by date and answer type
5. If first date provided, only keeps those questions whose resolution date is on or after the first date.

**Output:** `{name}_final_questions.jsonl` with all intermediate files saved.

### Free-form prompts and dates

Single- and multi-question generation use the same prompt, returning between one
and `--num_q_per_article` distinct questions. The guidelines emphasize recent,
consequential events, natural forecasting questions, explicitly reported answers,
and avoiding hindsight or answer hints. Answers may be short paraphrases rather
than exact quotations. Pass `--fewshot` to include the worked example; it is off
by default.

New questions include `<start_date>` and `<resolution_date>` in YYYY-MM-DD format,
with start before resolution. Conversion stores them as `question_start_date`
and `resolution_date`, so valid generated dates do not need a separate extraction
call. Generation uses `date_publish`, rather than substituting `date_modify`.
The existing binary validator, best-question selector, and leakage rewriter are
retained and preserve the explicit dates. Older questions without date tags still
use the existing date-processing path.

### Optional feedback generation workflow

An alternative command uses shared numbered generation criteria and a verifier
that returns all violated IDs plus a short explanation. With `--revise`, if every
candidate has violations, it revises the candidate with the fewest violations
(breaking ties by lowest ID) once. The generator can repair it or replace it
with a different question grounded in the same document. The verifier then
checks every criterion again. Each model continues its own original conversation.

```bash
python -m qgen.scripts.generate_with_feedback \
    --article_path articles.jsonl \
    --output_path feedback/documents.jsonl \
    --creator_model deepseek/deepseek-chat-v3-0324 \
    --verifier_model meta-llama/llama-4-maverick \
    --num_q_per_article 3 --revise
```

This workflow lives in `qgen/feedback/` and uses its original
`<question id="1">` XML schema, with explicit dates. The worked example is enabled
by default here; use `--no_fewshot` to omit it. It does not call the binary
validator, best-question selector, or leakage rewriter. The existing generation
command and end-to-end pipeline continue to use those stages.

The output contains document records with `candidates`, original questions and
verification reports, and a nested `revision` with both conversations, responses,
and status. A candidate passes when its verification has an empty `violations`
list; use the revision's question and report when its status is `completed`.
An accepted original candidate or an incomplete/failed initial verification
skips revision. Completed and failed attempts are not repeated on resume; an
interrupted attempt resumes from the saved revision before re-verification.
Errors remain recorded alongside the original question. Reuse the output path
only with the same articles, models, and settings; to enable revision later,
rerun the same command with `--revise`.

### Optional quality checks

These checks run separately on standardized questions, after generation and the
existing validation/selection/leakage stages. Enable any subset:

```bash
python -m qgen.scripts.check_questions \
    --input questions_with_retrieval.jsonl \
    --output_dir quality_checks \
    --model deepseek/deepseek-chat-v3-0324 \
    --date_model deepseek/deepseek-chat-v3-0324:online \
    --grounding --fix_dates --difficulty
```

- **`--grounding`** independently answers from `article_maintext` and
  `article_publish_date`, without receiving the reference answer. It keeps a
  question only when the top answer matches the reference and has probability
  strictly greater than 0.5. This supplements the existing binary faithfulness
  validator, which sees the supplied answer.
- **`--fix_dates`** researches the answer and both dates through OpenRouter web
  search (`--date_model` must end in `:online`). It includes the resolution
  criteria and seeks the earliest publicly certain answer date. The `:online`
  suffix enables [OpenRouter web search](https://openrouter.ai/docs/guides/features/plugins/web-search).
  An answer
  mismatch triggers one answer-informed follow-up in the same conversation.
  One additional correction turn is allowed for invalid date responses. A
  successful result replaces both dates and requires start before resolution;
  the source publication date does not cap the researched resolution date.
  This extends the earlier `late_resolve` date-estimation workflow.
- **`--difficulty`** forecasts seven days before the corrected resolution date
  and keeps questions with probability at most 0.5 on the reference answer.
  Matching accepts semantic equivalents. The recorded probability is the maximum
  across matching outcomes, or zero when a successful forecast has no match.
  Failed calls or invalid responses are errors, not zero-probability forecasts.
  This differs from the archived pass@k filter, which retains questions with
  successful attempts.

Difficulty uses existing embedding-retrieval results in
`relevant_articles_sorted_by_docs` (triples of score, document ID, and metadata)
or `relevant_docs`. Document metadata must include `max_date` as Unix seconds or
an ISO date/time, and `relevant_passage` or `summary`. Dictionary entries are also
accepted. Evidence without a date or after the cutoff is excluded, including
documents with a later `date_modify`. Retrieval is not rerun after date correction;
prepare the candidate evidence with sufficient coverage. An explicitly empty
retrieval list runs without evidence and records an evidence count of zero. A
missing retrieval field is an error.

Use models without built-in live search for `--model` and `--judge_model` (the
latter defaults to `--model`). Historical filtering constrains the supplied
evidence, not knowledge already present in model weights. `--cutoff_days`,
`--difficulty_threshold`, `--concurrency`, and `--max_tokens` are configurable.
All enabled checks run in order: grounding, dates, difficulty. A rejection or error
stops that question's remaining checks.

The runner writes accepted rows to `questions.jsonl`, per-question decisions and
raw model interactions to `checks.jsonl`, and settings to `config.json` under the
chosen output directory. Use a fresh directory for each run. Existing question IDs
are preserved; reports also include the zero-based input index. These checks use
the existing OpenRouter client and do not require a separate forecasting runtime.

Run the offline checks with:

```bash
python -m pytest qgen/tests -q
```

---

## 📋 Arguments

### Required
- `--article_path` - Input articles file
- `--output_dir` - Output directory

### Model Configuration
- `--use_openrouter` - Use OpenRouter API
- `--creator_model` - Model for generation (default: deepseek/deepseek-chat-v3-0324)
- `--selector_model` - Model for validation/selection (default: meta-llama/llama-4-maverick)
- `--model_path` - Local model for VLLM (if not using OpenRouter)

### Optional
- `--num_q_per_article N` - Questions per article (default: 1)
- `--first_date YYYY-MM-DD` - Filter questions on or after this date
- `--explicit_filter` - Apply strict answer type filtering
- `--seed N` - Random seed for reproducible IDs
- `--batch_size N` - Batch size (default: 1000)
- `--regenerate` - Force regeneration

---

## 💼 Job Submission (HTCondor)

```bash
# Submit via HTCondor
python jobs/jobs_qgen.py \
    --article_path /path/to/articles.jsonl \
    --output_dir /path/to/output \
    --job_memory 64 \
    --job_gpus 1 \
    --additional_args "--first_date 2025-01-01 --seed 42"
```

---

## 🔧 Individual Scripts (Advanced)

### Generate Questions Only
```bash
python scripts/generate_questions.py \
    --article_path articles.jsonl \
    --output_path questions.jsonl \
    --use_openrouter \
    --num_q_per_article 3
```

### Process Dates
```bash
python scripts/process_dates_cli.py \
    --input_path questions.jsonl \
    --extract_resolution
```

### Remove Leakage
```bash
python scripts/remove_leakage_cli.py \
    --mode llm \
    --input_file questions.jsonl
```

### Filter Articles
```bash
python scripts/filter_articles_cli.py \
    --mode select \
    --article_path /data/articles/ \
    --min_word_count 150 \
    --random_sample 500
```

---

## 📊 Data Formats

### Input (Articles)
```json
{
  "title": "Article Title",
  "maintext": "Full article content",
  "url": "https://...",
  "date_publish": "2024-01-15"
}
```

### Output (Questions)
```json
{
  "question_id": 482761,
  "question_title": "Who will win the Nobel Prize in 2024?",
  "background": "The Nobel Prize...",
  "resolution_criteria": "Source: Nobel Committee...",
  "answer": "John Doe",
  "answer_type": "Name",
  "resolution_date": "2024-10-10",
  "news_source": "cnn"
}
```

---

## 🔑 Key Modules

- **`run_pipeline.py`** - End-to-end pipeline with quality defaults
- **`qgen_core/question_generator.py`** - Question generation with validation
- **`filters/leakage_filter.py`** - Leakage detection and removal
- **`processors/date_processor.py`** - Date extraction and updates
- **`utils/`** - Reusable utilities (I/O, filtering, conversion, IDs)
- **`inference/`** - OpenRouter & VLLM engines

---

## 💡 Tips

1. **Use the pipeline** - `run_pipeline.py` handles everything automatically
2. **Intermediate files saved** - Inspect them if something goes wrong
3. **Reproducible IDs** - Use `--seed 42`
4. **Optional date filtering** - Omit `--first_date` to keep all questions
5. **HTCondor for scale** - Use job submission for large datasets

---
