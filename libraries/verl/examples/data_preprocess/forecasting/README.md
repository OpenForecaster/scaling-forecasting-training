# Forecasting Data Preprocessing for VeRL

Convert forecasting datasets to VeRL training format.

## Files

- **`prompt_utils.py`** - Shared prompt formatting functions
- **`load_foresight.py`** - Load OpenForesight from HuggingFace
- **`prepare_custom_dataset.py`** - Convert custom JSONL questions (need to provide as `--questions_file` args) to parquet format required by VeRL.

---

## Usage

### Load OpenForesight Dataset

```bash
# Full train split
python load_foresight.py --split train --output_dir data/

# Subsample 1000 examples
python load_foresight.py --split train --subsample 1000 --output_dir data/
```

**Output:** `foresight_{split}_samples{subsample}.parquet`

---



### Prepare Custom Questions

Requires a questions file as argument. This can be gathered using the question generation pipeline in `qgen/` subdirectory. Skip if you don't have one. 

```bash
# Convert questions from qgen pipeline
python prepare_custom_dataset.py \
    --questions_file /path/to/questions.jsonl \
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

---

## Launch Training

After preparing the data, launch RL training using the scripts in `scripts/ours/testrun/`.

### Quick Start

```bash
# Basic training with OpenForesight data
../../../scripts/ours/testrun/run_training.sh \
    --train_files data/foresight_train_samples10000.parquet \
    --model_name Qwen3-8B \
    --model_path Qwen/Qwen3-8B
```

### Using Python Launcher Directly

```bash
python ../../../scripts/ours/testrun/launch_training.py \
    --train_files data/foresight_train_samples10000.parquet \
    --val_files data/foresight_validation_samples500.parquet \
    --model_name Qwen3-8B \
    --lr 5e-6 \
    --total_epochs 5
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | Qwen3-8B | Model name (looks in /fast/nchandak/models/) |
| `--model_path` | auto | Full path to model (overrides model_name) |
| `--train_files` | - | Training data JSONL file |
| `--val_files` | - | Validation data JSONL file |
| `--lr` | 5e-6 | Learning rate |
| `--kl_coeff` | 0.005 | KL divergence coefficient |
| `--total_epochs` | 7 | Number of training epochs |
| `--train_batch_size` | 256 | Training batch size |
| `--max_prompt_length` | 4096 | Maximum prompt tokens |
| `--max_response_length` | 4096 | Maximum response tokens |
| `--num_gpus` | 8 | Number of GPUs |
| `--dry_run` | False | Print command without executing |

### Full Example: Train on OpenForesight

```bash
# Step 1: Prepare data
cd libraries/verl/examples/data_preprocess/forecasting
python load_foresight.py --split train --subsample 10000 --output_dir data/
python load_foresight.py --split validation --subsample 500 --output_dir data/

# Step 2: Launch training
cd ../../..
./scripts/ours/testrun/run_training.sh \
    --train_files examples/data_preprocess/forecasting/data/foresight_train_samples10000.jsonl \
    --val_files examples/data_preprocess/forecasting/data/foresight_validation_samples500.jsonl \
    --model_name Qwen3-8B \
    --project_name openforesight-training \
    --total_epochs 5 \
    --save_freq 50
```

### Dry Run (Preview Command)

```bash
./scripts/ours/testrun/run_training.sh --dry_run \
    --train_files data/foresight_train.jsonl \
    --model_name Qwen3-8B
```
