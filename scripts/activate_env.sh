#!/usr/bin/env bash
# FIX: Activation script for relocated virtualenv
export ENV_PATH="${ENV_PATH:-$HOME/software/he_nn_env}"  # FIX: default to relocated env path
# FIX: Check if virtualenv exists and activate it
if [ -f "$ENV_PATH/bin/activate" ]; then
  source "$ENV_PATH/bin/activate"  # FIX: activate the environment
else
  echo "ERROR: virtualenv not found at $ENV_PATH" >&2
  exit 1
fi
# FIX: end of activation script
