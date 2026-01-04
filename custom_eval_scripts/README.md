# Forecasting Evaluation Scripts

Evaluation suite for forecasting models on various benchmarks.

## Quick Start

### Freeform Forecasting
```bash
python eval_freeform.py \
  --model_dir=/path/to/model \
  --questions_file=/path/to/questions.jsonl \
  --base_save_dir=/path/to/output \
  --num_generations=3
```

### Binary Forecasting
```bash
python eval_binary.py \
  --model_dir=/path/to/model \
  --questions_file=/path/to/questions.jsonl \
  --base_save_dir=/path/to/output \
  --num_generations=5
```

### Multiple Choice (MCQ)
```bash
python eval_forecasting_mcq.py \
  --model_dir=/path/to/model \
  --data=metaculus_mcq \
  --base_save_dir=/path/to/output
```

## Benchmark specific files

- **Binary**: Binary yes/no forecasting questions (Metaculus, Manifold)
- **Freeform**: Open-ended questions with text answers
- **MCQ**: Multiple choice forecasting questions (often from Metaculus/Manifold)
- **FutureX**: For evaluating FutureX (Past) questions
- **FutureBench**: FutureBench benchmark from Huggingface
- **MMLU-Pro**: Academic knowledge evaluation
- **MATH**: Mathematical reasoning
- **SimpleQA**: Factual knowledge and calibration eval 

## Common Arguments

- `--model_dir`: Path to model directory
- `--base_save_dir`: Output directory for results
- `--num_generations`: Number of generations per question (default: 3)
- `--max_new_tokens`: Maximum tokens to generate (default: 16384)
- `--questions_file`: Path to the jsonl (questions) file

## Output Format

Results are saved as JSONL files with:
- Model responses
- Extracted answers and probabilities
- Token counts
- Question metadata
