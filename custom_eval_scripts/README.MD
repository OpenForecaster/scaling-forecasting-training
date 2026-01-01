## How to run

#### To run evaluation 

##### To run locally (choose the right eval file depending on task like forecasting, mmlu-pro, math, etc.)

```bash
python eval_forecasting_vllm.py --base_save_dir=/fast/nchandak/forecasting/evals/sft/llama3.1-8b/lora256/outputs/metaculus_test/full/ --model_dir=/fast/nchandak/forecasting/sft/llama3.1-8b/full/checkpoint-1000/ --model="llama3.1-8b-ins-full-checkpoint-1000" --max_new_tokens=16384 --data="metaculus"
```

##### To run on cluster by launching as a job (specify which task you want to run on like forecasting, mmlu-pro, math, etc.)

```bash
python jobs_eval.py --base_save_dir=/fast/nchandak/forecasting/evals/sft/llama3.1-8b/lora256/outputs/metaculus_test/full/ --model_dir=/fast/nchandak/forecasting/merged_models/sft/llama3.1-8b/lora256/checkpoint-3500/ --model="llama3.1-8b-ins-lora256-checkpoint-3500" --max_new_tokens=16384 --data="metaculus" --task forecasting
```


