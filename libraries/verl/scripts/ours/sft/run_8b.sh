#! /bin/bash

# Run the distill script with different learning rates
MODEL_NAME="Qwen3-8B"

# for lr in 2e-5 3e-5 5e-5 7e-5 1e-4; do
#     echo "Running distill script with learning rate $lr"
#     bash /home/nchandak/forecasting/libraries/verl/scripts/ours/sft/distill.sh $lr $MODEL_NAME
# done


for lr in 7e-5; do
    echo "Running distill script with learning rate $lr"
    bash /home/nchandak/forecasting/libraries/verl/scripts/ours/sft/distill.sh $lr $MODEL_NAME
done

