#!/usr/bin/env bash
# FIX: Start the He_NN Trading Backend Server with conda environment
set -e  # FIX: exit on any error

# FIX: Initialize conda for bash shell
eval "$(conda shell.bash hook)"  # FIX: initialize conda

# FIX: Set environment name from config or use default
export ENV_NAME="${ENV_NAME:-binance_env}"  # FIX: default to binance_env

# FIX: Activate conda environment
echo "Activating conda environment: $ENV_NAME"  # FIX: startup message
conda activate "$ENV_NAME"  # FIX: activate conda environment

if [ $? -eq 0 ]; then
  echo "✓ Conda environment '$ENV_NAME' activated successfully"  # FIX: confirmation message
else
  echo "ERROR: Failed to activate conda environment '$ENV_NAME'" >&2  # FIX: error message
  echo "Please ensure the environment exists. Run: conda env list" >&2  # FIX: hint
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
