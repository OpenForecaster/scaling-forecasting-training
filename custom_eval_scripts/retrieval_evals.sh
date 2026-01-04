#! /bin/bash

# Shell script for running retrieval-based evaluations with varying article counts.
# Evaluates Qwen3-4B and Qwen3-8B models with 1-10 retrieved articles.
# Useful for ablation studies on retrieval-augmented generation.
# Submits HTCondor jobs via jobs_eval.py for each configuration.


for num_articles in {1..10}; do
    python jobs_eval.py --base_save_dir /fast/nchandak/forecasting/evals/freeform/manual --model_dir /fast/nchandak/models/Qwen3-4B --model Qwen3-4B --max_new_tokens 16384 --data_split test --num_generations 5 --data /fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian-retrieval-30_207_free_3_cleaned.jsonl --num_articles $num_articles --task retrieval
done

for num_articles in {1..10}; do
    python jobs_eval.py --base_save_dir /fast/nchandak/forecasting/evals/freeform/manual --model_dir /fast/nchandak/models/Qwen3-8B --model Qwen3-8B --max_new_tokens 16384 --data_split test --num_generations 5 --data /fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian-retrieval-30_207_free_3_cleaned.jsonl --num_articles $num_articles --task retrieval
done