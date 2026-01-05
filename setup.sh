#!/bin/bash

# Setup script for forecasting-rl project
# This script will set up the UV environment and install all dependencies

set -e  # Exit on any error

echo "🚀 Setting up forecasting-rl environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo "✅ UV installed successfully"
else
    echo "✅ UV is already installed"
fi

# Create virtual environment
if [ ! -d "forecast" ]; then
    echo "📦 Creating virtual environment..."
    uv venv forecast
else
    echo "📦 Virtual environment 'forecast' already exists."
fi

# Activate the environment
echo "🔧 Activating virtual environment..."
source forecast/bin/activate

# Install PyTorch with CUDA 12.1 support
echo "🔥 Installing PyTorch with CUDA 12.1..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install build dependencies to support --no-build-isolation
echo "🛠️ Installing build dependencies..."
uv pip install hatchling setuptools wheel packaging ninja editables

# Install all project dependencies
echo "📚 Installing project dependencies..."
uv pip install -e . --no-build-isolation

# Install VeRL library if it exists
if [ -d "libraries/verl" ]; then
    echo "🔬 Installing VeRL library..."
    cd libraries/verl
    uv pip install -e .
    cd ../..
fi


# Configure OpenRouter API Key
KEY_FILE="qgen/config/openrouter_key.py"
if [ ! -f "$KEY_FILE" ]; then
    echo ""
    echo "🔑 OpenRouter API Key Configuration"
    echo "This project requires an OpenRouter API key for generating questions."
    read -p "Do you want to enter your OpenRouter API key now? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your API Key: " API_KEY
        echo "API_KEY = \"$API_KEY\"" > "$KEY_FILE"
        echo "✅ Key saved to $KEY_FILE"
    else
        echo "API_KEY = \"\"" > "$KEY_FILE"
        echo "⚠️  Created placeholder file at $KEY_FILE"
        echo "Please edit this file and add your actual API Key before running the pipeline."
    fi
else
    echo "✅ OpenRouter key configuration found at $KEY_FILE"
fi

echo "✅ Setup complete! Your environment is ready."
echo ""
echo "To activate the environment in the future, run:"
echo "  source forecast/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate" 