#!/bin/bash
# Setup script for He_NN Trading Backend

set -e

echo "==== He_NN Trading Backend Setup ===="
echo

# Check Python version
echo "Checking Python version..."
python3 --version
echo

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi
echo

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo

# Install dependencies
echo "Installing backend dependencies..."
pip install -r requirements-backend.txt
echo

echo "==== Backend Setup Complete ===="
echo
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo
echo "To start the backend server, run:"
echo "  python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"
echo
