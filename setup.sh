#!/bin/bash

# Basic settings
PYTHON_VERSION="3.12"
REQUIREMENTS_FILE="requirements.txt"
VENV_DIR=".venv"

echo "=== Setting up Python environment ==="

# check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Please install uv first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv --python $PYTHON_VERSION --seed
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        exit 1
    fi
else
    echo "Virtual environment already exists. Reusing it."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Check Python version
echo "Using Python $(python --version 2>&1)"

# Install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from ${REQUIREMENTS_FILE}..."
    uv pip install -r $REQUIREMENTS_FILE
    if [ $? -ne 0 ]; then
        echo "Failed to install requirements"
        exit 1
    fi
fi

# Install additional packages
echo "Installing additional packages..."
uv pip install --upgrade pip setuptools wheel ninja

# Install flash-attn
echo "Installing flash-attn..."
if ! uv pip install flash-attn==2.8.3 --no-build-isolation; then
    echo "flash-attn installation failed. CUDA toolkit may be required."
    echo "Continuing without flash-attn..."
fi

echo "=== Environment setup complete ==="
echo "To activate environment: source $VENV_DIR/bin/activate"
