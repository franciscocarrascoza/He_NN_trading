#!/bin/bash
# Setup script for He_NN Trading Desktop GUI (Qt6)

set -e

echo "==== He_NN Trading Desktop GUI Setup ===="
echo

# Check if Qt6 development packages are installed
echo "Checking for Qt6 development packages..."
if ! dpkg -l | grep -q "qt6-base-dev"; then
    echo "ERROR: Qt6 development packages not found!"
    echo
    echo "Please install the required Qt6 packages:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y qt6-base-dev qt6-websockets-dev qt6-charts-dev cmake build-essential"
    echo
    exit 1
else
    echo "Qt6 development packages found."
fi
echo

# Navigate to desktop UI directory
cd ui/desktop

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Navigate to build directory
cd build

# Run CMake
echo "Running CMake configuration..."
cmake ..
echo

# Build the application
echo "Building the application..."
make -j$(nproc)
echo

echo "==== GUI Build Complete ===="
echo
echo "The executable is located at:"
echo "  ui/desktop/build/HeNNTradingDesktop"
echo
echo "To run the GUI:"
echo "  cd ui/desktop/build"
echo "  ./HeNNTradingDesktop"
echo
