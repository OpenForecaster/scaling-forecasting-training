#!/bin/bash

# Example usage of filter_articles.py
# This script shows how to run the article filtering with VLLM

# Set paths
ARTICLES_PATH="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.dw.com_2025-05_selected100.jsonl"
ARTICLES_PATH="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.reuters.com_2024-05_selected10000.jsonl"
MODEL_PATH="/fast/nchandak/models/Qwen3-32B"

# Check if files exist
if [ ! -f "$ARTICLES_PATH" ]; then
    echo "Error: Articles file not found at $ARTICLES_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found at $MODEL_PATH"
    exit 1
fi

echo "Starting article filtering..."
echo "Articles path: $ARTICLES_PATH"
echo "Model path: $MODEL_PATH"

# Run the filtering script
python filter_articles.py \
    --articles_path "$ARTICLES_PATH" \
    --model_path "$MODEL_PATH" \
    --output_path "${ARTICLES_PATH%.jsonl}_filtered.jsonl"

echo "Filtering completed!"
echo "Check the output file for results with 'relevant' field added." 