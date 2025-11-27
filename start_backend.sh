#!/usr/bin/env bash
# FIX: Start the He_NN Trading Backend Server with relocated virtualenv
set -e  # FIX: exit on any error

# FIX: Set environment path from config or use default
export ENV_PATH="${ENV_PATH:-$HOME/.anaconda3/envs/binance_env}"  # FIX: default to working conda environment

# FIX: Activate conda environment
if [ -f "$ENV_PATH/bin/python" ]; then
  # FIX: For conda environments, use conda activate or directly set PATH
  export PATH="$ENV_PATH/bin:$PATH"  # FIX: add conda env to PATH
  echo "✓ Conda environment activated from $ENV_PATH"  # FIX: confirmation message
else
  echo "ERROR: Conda environment not found at $ENV_PATH" >&2  # FIX: error message
  echo "Please ensure the conda environment exists at $ENV_PATH" >&2  # FIX: hint
  exit 1  # FIX: exit with error
fi

# FIX: Set environment variables for reproducibility (from README)
export OMP_NUM_THREADS=1  # FIX: OpenMP threads
export OPENBLAS_NUM_THREADS=1  # FIX: OpenBLAS threads
export MKL_NUM_THREADS=1  # FIX: MKL threads
export MKL_THREADING_LAYER=GNU  # FIX: MKL threading layer
export KMP_AFFINITY=disabled  # FIX: KMP affinity
export KMP_INIT_AT_FORK=FALSE  # FIX: KMP init at fork

# FIX: Change to repository directory
cd "$(dirname "$0")"  # FIX: change to script directory

# FIX: Start the FastAPI backend
echo "Starting He_NN Trading Backend Server..."  # FIX: startup message
echo "Backend will be available at http://localhost:8000"  # FIX: endpoint info
echo "WebSocket endpoint: ws://localhost:8000/ws"  # FIX: WS endpoint info
echo ""  # FIX: blank line
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level info  # FIX: start server
