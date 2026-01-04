# Forecasting Data Preprocessing for VeRL

Convert forecasting datasets to VeRL training format.

## Files

- **`prompt_utils.py`** - Shared prompt formatting functions
- **`load_foresight.py`** - Load OpenForesight from HuggingFace
- **`prepare_custom_dataset.py`** - Convert custom JSONL questions to VERL format

---

## Usage

### Load OpenForesight Dataset

```bash
# Full train split
python load_foresight.py --split train --output_dir data/

# Subsample 1000 examples
python load_foresight.py --split train --subsample 1000 --output_dir data/
```

**Output:** `foresight_{split}_samples{subsample}.jsonl`

---

### Prepare Custom Questions

```bash
# Convert questions from qgen pipeline
python prepare_custom_dataset.py \
    --questions_file /path/to/questions.jsonl \
    --output_dir data/

# With subsampling
python prepare_custom_dataset.py \
    --questions_file /path/to/questions.jsonl \
    --subsample 500 \
    --split validation \
    --output_dir data/
```

**Output:** `{input_filename}_verl_samples{subsample}.jsonl`

---

## Arguments

Both scripts support:
- `--subsample N` - Randomly sample N examples
- `--seed 42` - Random seed for reproducibility
- `--output_dir data/` - Output directory

**`load_foresight.py`:**
- `--split {train,validation,test}` - Dataset split (default: train)

**`prepare_custom_dataset.py`:**
- `--questions_file` - Input JSONL file (REQUIRED)
- `--split {train,validation,test}` - Split designation (default: train)

---

## VERL Output Format

```json
{
  "data_source": "freeform/custom-train",
  "prompt": [{"role": "user", "content": "...formatted prompt..."}],
  "ability": "forecasting",
  "reward_model": {"style": "rule", "ground_truth": "answer"},
  "extra_info": {
    "split": "train",
    "index": 0,
    "answer": "answer",
    "question": "question text",
    ...
  }
}
```

---

## Input Format (Custom Questions)

Required fields:
- `question_title` or `question`
- `background`
- `answer`

Optional: `resolution_criteria`, `answer_type`, `resolution_date`, etc.

Scripts automatically filter out irrelevant questions (`question_relevant: 0`, `no_good_question: 1`).
