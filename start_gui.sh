#!/usr/bin/env bash
# FIX: Start the He_NN Trading Desktop GUI with backend health check

# FIX: Check if backend is running
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "WARNING: Backend server does not appear to be running!"  # FIX: warning message
    echo "Please start the backend first with:"  # FIX: instruction
    echo "  ./start_backend.sh"  # FIX: command
    echo ""  # FIX: blank line
    read -p "Continue anyway? (y/N) " -n 1 -r  # FIX: prompt user
    echo ""  # FIX: blank line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1  # FIX: exit if user declines
    fi
fi

# FIX: Start the GUI
echo "Starting He_NN Trading Desktop GUI..."  # FIX: startup message
cd ui/desktop/build  # FIX: change to build directory
./HeNNTradingDesktop  # FIX: launch GUI executable
