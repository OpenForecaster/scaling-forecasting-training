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
