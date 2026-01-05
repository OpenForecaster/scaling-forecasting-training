# Answer Matching -- LLM-based Response Evaluation

Automated evaluation of model responses against ground truth using local LLM judges with VLLM.

## Quick Start

```bash
python llm_judge.py \
    --model_dir /path/to/judge/model \
    --input_file responses.jsonl \
    --output_dir ./results
```

## Arguments

### Required
- `--model_dir` - Path to judge model directory

### Input/Output
- `--input_file` - Path to specific file to judge
- `--input_dir` - Path to directory containing files (default: `/fast/nchandak/forecasting/evals/freeform/manual/theguardian_207/`)
- `--output_dir` - Output directory (defaults to input location)

### Optional Scoring Options
- `--logprobs` - Store normalized probabilities
- `--no_ground_truth` - Judge without ground truth (knowledge-based)

### Optional Generation Parameters
- `--max_tokens` - Max tokens to generate (default: 2048)
- `--gen_kwargs` - Custom generation params (e.g., `"temperature=0.7,top_p=0.9"`)
- `--thinking` - Use thinking mode parameters
- `--batch_size` - Batch size for processing 

## Input Format

JSONL file with:
```json
{
  "question": "Question text",
  "answer": "Ground truth answer",
  "extracted_answer": ["Response 1", "Response 2"]
}
```

Or standard format:
```json
{
  "question_title": "Question text",
  "answer": "Ground truth",
  "filtered_resps": ["Response"]
}
```

## Output Format

Adds judgment fields to input:
```json
{
  "score_model_name": [0, 1, 1],
  "prob_model_name": [0.95, 0.87, 0.92],
  "response_model_name": ["Full judge response..."]
}
```

## Examples

### Basic Usage
```bash
python llm_judge.py \
    --model_dir /fast/nchandak/models/Llama-4-Scout \
    --input_file samples.jsonl
```

### Process Entire Directory
```bash
python llm_judge.py \
    --model_dir /fast/nchandak/models/Llama-4-Scout \
    --input_dir /path/to/eval/data \
    --output_dir /path/to/results
```

## Notes

- Judgments are saved incrementally (overwrites input file)
- Already-judged samples are automatically skipped
- GPU memory is cleared after each file
- Supports both freeform and standard evaluation formats

