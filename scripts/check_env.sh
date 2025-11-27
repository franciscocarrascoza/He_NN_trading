#!/usr/bin/env bash
# FIX: Script to verify relocated virtualenv is working
export ENV_PATH="${ENV_PATH:-$HOME/software/he_nn_env}"  # FIX: default to relocated env path
echo "Checking virtual environment at: $ENV_PATH"  # FIX: display path
# FIX: Check if environment exists
if [ -f "$ENV_PATH/bin/python" ]; then
  echo "✓ Virtual environment found"  # FIX: confirm found
  $ENV_PATH/bin/python -V  # FIX: show Python version
  echo ""  # FIX: blank line
  echo "Installed packages:"  # FIX: header
  $ENV_PATH/bin/pip list | head -20  # FIX: show first 20 packages
else
  echo "✗ Virtual environment not found at $ENV_PATH" >&2  # FIX: error message
  exit 1
fi
