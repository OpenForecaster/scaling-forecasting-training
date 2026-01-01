# Setup Guide for Collaborators

This guide will help you set up the forecasting-rl project environment using UV.

## Prerequisites

Before starting, ensure you have:
- Python 3.9 or higher
- Git
- CUDA 12.1 (for GPU support with PyTorch)
- Internet connection for downloading packages

## Quick Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nikhilchandak/forecasting-rl.git
   cd forecasting-rl
   ```

2. **Run the automated setup:**
   ```bash
   ./setup.sh
   ```

3. **Activate the environment:**
   ```bash
   source forecast/bin/activate
   ```

That's it! Your environment is ready to use.

## Manual Setup

If the automated setup doesn't work, follow these steps:

### Step 1: Install UV

```bash
# Install UV using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to your PATH (you may need to restart your terminal)
source $HOME/.cargo/env
```

### Step 2: Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/nikhilchandak/forecasting-rl.git
cd forecasting-rl

# Create virtual environment
uv venv forecast

# Activate the environment
source forecast/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all project dependencies
uv pip install -e .

# Install VeRL library (if needed)
cd libraries/verl
uv pip install -e .
cd ../..
```

## Environment Management

### Activating the Environment
```bash
source forecast/bin/activate
```

### Deactivating the Environment
```bash
deactivate
```

### Checking Installed Packages
```bash
# Activate environment first
source forecast/bin/activate

# List installed packages
uv pip list
```

### Updating Dependencies
```bash
# Activate environment first
source forecast/bin/activate

# Update all dependencies
uv pip install -e . --upgrade
```

## Troubleshooting

### Common Issues

1. **UV command not found**
   - Solution: Install UV using the installer above
   - Make sure to restart your terminal after installation

2. **Permission denied when running setup.sh**
   - Solution: Make the script executable: `chmod +x setup.sh`

3. **CUDA/GPU issues**
   - Ensure CUDA 12.1 is installed
   - Check GPU compatibility with PyTorch

4. **Package installation failures**
   - Try updating UV: `uv self update`
   - Check your internet connection
   - Try installing packages one by one to identify problematic ones

5. **Python version issues**
   - Ensure you're using Python 3.9 or higher
   - Check with: `python --version`

### Getting Help

If you encounter issues:
1. Check the error messages carefully
2. Ensure all prerequisites are met
3. Try the manual setup steps
4. Check the project's issue tracker
5. Contact the project maintainer

## Alternative Setup Methods

### Using Traditional pip (Not Recommended)

If you prefer using pip instead of UV:

```bash
# Create a virtual environment
python -m venv forecast
source forecast/bin/activate

# Install dependencies
pip install -r requirements_uv.txt

# Install PyTorch separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Using Conda (Legacy)

The project previously used conda environments. If you need to use conda:

```bash
conda create -n forecast python=3.10
conda activate forecast
pip install -r requirements_uv.txt
```

## Project Structure

After setup, your project structure should look like:

```
forecasting-rl/
├── forecast/           # UV virtual environment
├── libraries/          # Custom libraries (including VeRL)
├── data/              # Data files
├── notebooks/         # Jupyter notebooks
├── trainingTRL/       # Training scripts
├── sft/              # Supervised fine-tuning
├── pyproject.toml    # UV project configuration
├── setup.sh          # Automated setup script
└── README.md         # Project documentation
```

## Next Steps

After successful setup:
1. Explore the project structure
2. Check out the notebooks in the `notebooks/` directory
3. Review the training scripts in `trainingTRL/` and `sft/`
4. Read the main README.md for project-specific information 