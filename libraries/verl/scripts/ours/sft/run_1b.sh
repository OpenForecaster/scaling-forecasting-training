#! /bin/bash

# # Run the distill script with different learning rates
# MODEL_NAME="Qwen3-1.7B"

# for lr in 2e-4 3e-4 5e-4 7e-4 8e-4 1e-3 2e-3; do
#     echo "Running distill script with learning rate $lr"
#     bash /home/nchandak/forecasting/libraries/verl/scripts/ours/sft/distill.sh $lr $MODEL_NAME
# done


# Run the distill script with different learning rates
# MODEL_NAME="Qwen3-1.7B"

# for lr in 2e-4; do
#     echo "Running distill script with learning rate $lr"
#     bash /home/nchandak/forecasting/libraries/verl/scripts/ours/sft/distill.sh $lr $MODEL_NAME
# done



# # Run the distill script with different learning rates
# MODEL_NAME="Qwen3-4B"

# for lr in 2e-5 3e-5 7e-5; do
#     echo "Running distill script with learning rate $lr"
#     bash /home/nchandak/forecasting/libraries/verl/scripts/ours/sft/distill.sh $lr $MODEL_NAME
# done


# Run the distill script with different learning rates
MODEL_NAME="Qwen3-4B"

for lr in 7e-5; do
    echo "Running distill script with learning rate $lr"
    bash /home/nchandak/forecasting/libraries/verl/scripts/ours/sft/distill.sh $lr $MODEL_NAME
done