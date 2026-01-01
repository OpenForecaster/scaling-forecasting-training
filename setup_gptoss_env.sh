#!/bin/bash

# Create gptoss environment with Python 3.12
uv venv gptoss --python 3.12 --seed

# Activate it
source gptoss/bin/activate

# Install vLLM with gpt-oss support (this is the special version)
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Install openai-harmony (required for gpt-oss models)
uv pip install openai-harmony

# Install openai-agents (for agent capabilities)
uv pip install openai-agents

# Install all other packages from forecast environment (excluding vllm since we installed special version)
grep -v "^vllm==" requirements-gptoss-temp.txt > /tmp/gptoss_requirements_temp.txt
uv pip install -r /tmp/gptoss_requirements_temp.txt

# Install verl in editable mode
uv pip install -e libraries/verl

echo "✅ gptoss environment setup complete!"
echo "To activate: source gptoss/bin/activate"
echo "To start gpt-oss-20b server: vllm serve openai/gpt-oss-20b"

