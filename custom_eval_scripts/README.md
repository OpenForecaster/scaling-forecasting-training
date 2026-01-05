# Forecasting Evaluation Scripts

Evaluation suite for forecasting models on various benchmarks.

## Evaluation Scripts

### Forecasting Tasks
- **`eval_openforesight.py`**: OpenForesight dataset (HuggingFace)
- **`eval_paleka.py`**: Consistency checks (dpaleka/ccflmf from HuggingFace or local)

### General Benchmarks
- **`eval_mmlu_pro.py`**: MMLU-Pro academic knowledge
- **`eval_simpleqa.py`**: Factual knowledge and calibration

## Quick Examples

```bash
# OpenForesight (with retrieval)
python eval_openforesight.py --model_dir=/path/to/model --data_split=test

# OpenForesight (without retrieval)
python eval_openforesight.py --model_dir=/path/to/model --data_split=test --without_retrieval
```

## Common Arguments

- `--model_dir`: Path to model directory
- `--base_save_dir`: Output directory for results
- `--num_generations`: Number of generations per question
- `--max_new_tokens`: Maximum tokens to generate (default: 16384)
- `--data_split`: Dataset split (train/validation/test)


#### Additional Files 

- **`eval_freeform_retrieval.py`**: Freeform questions optionally with retrieval context
- **`eval_binary_retrieval.py`**: Binary yes/no questions optionally with retrieval
- **`eval_futurex.py`**: FutureX benchmark