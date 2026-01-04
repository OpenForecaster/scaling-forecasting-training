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
echo "📦 Creating virtual environment..."
uv venv forecast

# Activate the environment
echo "🔧 Activating virtual environment..."
source forecast/bin/activate

# Install PyTorch with CUDA 12.1 support
echo "🔥 Installing PyTorch with CUDA 12.1..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all project dependencies
echo "📚 Installing project dependencies..."
uv pip install -e .

# Install VeRL library if it exists
if [ -d "libraries/verl" ]; then
    echo "🔬 Installing VeRL library..."
    cd libraries/verl
    uv pip install -e .
    cd ../..
fi

echo "✅ Setup complete! Your environment is ready."
echo ""
echo "To activate the environment in the future, run:"
echo "  source forecast/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate" 