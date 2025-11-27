#!/usr/bin/env bash
# FIX: Comprehensive setup verification script
echo "=== He_NN Trading Setup Verification ==="  # FIX: header
echo ""  # FIX: blank line

# FIX: Check virtual environment
export ENV_PATH="${ENV_PATH:-$HOME/software/he_nn_env}"  # FIX: default env path
echo "1. Checking virtual environment..."  # FIX: section header
if [ -f "$ENV_PATH/bin/python" ]; then
  echo "   ✓ Virtual environment found at: $ENV_PATH"  # FIX: success message
  $ENV_PATH/bin/python --version  # FIX: show Python version
else
  echo "   ✗ Virtual environment NOT found at: $ENV_PATH" >&2  # FIX: error message
  echo "   Please ensure environment has been moved correctly." >&2  # FIX: hint
  exit 1  # FIX: exit with error
fi
echo ""  # FIX: blank line

# FIX: Check data directory
DATA_PATH="${DATA_PATH:-$HOME/trading/he_nn_data}"  # FIX: default data path
echo "2. Checking data directory..."  # FIX: section header
if [ -d "$DATA_PATH" ]; then
  echo "   ✓ Data directory found at: $DATA_PATH"  # FIX: success message
  echo "   Size: $(du -sh "$DATA_PATH" | cut -f1)"  # FIX: show size
else
  echo "   ✗ Data directory NOT found at: $DATA_PATH" >&2  # FIX: error message
  echo "   Please ensure data has been moved correctly." >&2  # FIX: hint
  exit 1  # FIX: exit with error
fi
echo ""  # FIX: blank line

# FIX: Check config file
echo "3. Checking config/app.yaml..."  # FIX: section header
if [ -f "config/app.yaml" ]; then
  echo "   ✓ Config file exists"  # FIX: success message
  # FIX: Verify config has required keys
  if grep -q "env_path:" config/app.yaml && grep -q "storage_path:" config/app.yaml; then
    echo "   ✓ Config contains required keys (env_path, storage_path)"  # FIX: success
  else
    echo "   ⚠ Config may be missing required keys" >&2  # FIX: warning
  fi
else
  echo "   ✗ config/app.yaml NOT found" >&2  # FIX: error
  exit 1  # FIX: exit with error
fi
echo ""  # FIX: blank line

# FIX: Check startup scripts
echo "4. Checking startup scripts..."  # FIX: section header
for script in start_backend.sh start_gui.sh; do
  if [ -x "$script" ]; then
    echo "   ✓ $script is executable"  # FIX: success
  else
    echo "   ✗ $script is NOT executable or missing" >&2  # FIX: error
    exit 1  # FIX: exit with error
  fi
done
echo ""  # FIX: blank line

# FIX: Check backend module
echo "5. Checking backend module..."  # FIX: section header
if [ -f "backend/app.py" ]; then
  echo "   ✓ backend/app.py exists"  # FIX: success
else
  echo "   ✗ backend/app.py NOT found" >&2  # FIX: error
  exit 1  # FIX: exit with error
fi
echo ""  # FIX: blank line

# FIX: Check GUI executable
echo "6. Checking GUI executable..."  # FIX: section header
if [ -f "ui/desktop/build/HeNNTradingDesktop" ]; then
  echo "   ✓ GUI executable exists"  # FIX: success
else
  echo "   ⚠ GUI executable not found (may need to build)"  # FIX: warning
  echo "   Run: cd ui/desktop && mkdir build && cd build && cmake .. && cmake --build ."  # FIX: hint
fi
echo ""  # FIX: blank line

# FIX: Final summary
echo "=== Setup Verification Complete ==="  # FIX: footer
echo ""  # FIX: blank line
echo "To start the backend:"  # FIX: instructions
echo "  ./start_backend.sh"  # FIX: command
echo ""  # FIX: blank line
echo "To start the GUI:"  # FIX: instructions
echo "  ./start_gui.sh"  # FIX: command
echo ""  # FIX: blank line
