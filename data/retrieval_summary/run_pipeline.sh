#!/bin/bash

# Set default values
SOURCE_PATH="/fast/sgoel/forecasting/news/retrieval/metaculus-binary_reuters_7_365/"
OUTPUT_DIR="/fast/nchandak/forecasting/retrieval_summary/metaculus-binary_reuters_7_365"
MODEL_PATH="/fast/rolmedo/models/llama-3.3-70b-instruct"
MODEL_PATH="/fast/rolmedo/models/qwen2.5-14b-it"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --source_path)
      SOURCE_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=== News Article Summarization Pipeline ==="
echo "Source Path: $SOURCE_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Path: $MODEL_PATH"
echo

# Generate summaries
echo "=== Generating Summaries ==="
python generate_summaries.py \
  --source_path "$SOURCE_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --model_path "$MODEL_PATH"

# Check if summaries were generated successfully
if [ $? -ne 0 ]; then
  echo "Error generating summaries!"
  exit 1
fi

# Analyze summaries
echo
echo "=== Analyzing Summaries ==="
python analyze_summaries.py \
  --summaries_dir "$OUTPUT_DIR"

echo
echo "=== Pipeline Complete ==="
echo "Results saved to $OUTPUT_DIR" 