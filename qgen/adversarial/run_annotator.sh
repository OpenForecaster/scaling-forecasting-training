#!/bin/bash
# Quick start script for the annotation tool

# Default values
DATA_PATH="/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_manualFilter.jsonl"
PORT=5001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--data_path PATH] [--port PORT]"
            exit 1
            ;;
    esac
done

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

echo "Starting Question Annotation Tool..."
echo "Data file: $DATA_PATH"
echo "Port: $PORT"
echo ""
echo "Open your browser to: http://localhost:$PORT"
echo ""

# Run the annotator
python annotator.py --data_path "$DATA_PATH" --port "$PORT"

