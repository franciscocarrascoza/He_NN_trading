# He_NN Trading - Setup Instructions

This document provides instructions for fixing and setting up the He_NN Trading desktop application.

## Issues Found and Fixed

### 1. Qt6 Development Packages Missing
**Error:** CMake could not find Qt6Config.cmake
```
CMake Error at CMakeLists.txt:13 (find_package):
  By not providing "FindQt6.cmake" in CMAKE_MODULE_PATH this project has
  asked CMake to find a package configuration file provided by "Qt6", but
  CMake did not find one.
```
**Solution:** Install Qt6 development packages (`qt6-base-dev`, `qt6-websockets-dev`, `qt6-charts-dev`)

### 2. Python venv Package Missing
**Error:** Cannot create virtual environment
```
The virtual environment was not created successfully because ensurepip is not
available.
```
**Solution:** Install `python3.12-venv` package

### 3. Python Backend Dependencies Missing
**Error:** FastAPI, uvicorn, and websockets not installed
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution:** Install Python dependencies via requirements-backend.txt

### 4. Missing Setup and Startup Scripts
**Solution:** Created setup and startup scripts for both backend and GUI

## Setup Instructions

### Prerequisites

- Ubuntu/Debian-based Linux system
- Python 3.12+ (already installed)
- CMake 3.16+ (already installed)
- sudo access (for installing Qt6 packages)

### Step 1: Install System Dependencies

You need sudo access to install the required system packages:

```bash
sudo apt-get update
sudo apt-get install -y \
    qt6-base-dev \
    qt6-websockets-dev \
    qt6-charts-dev \
    cmake \
    build-essential \
    python3.12-venv
```

These packages provide:
- `qt6-base-dev`: Qt6 Core, Widgets, Network modules
- `qt6-websockets-dev`: Qt6 WebSockets module for backend communication
- `qt6-charts-dev`: Qt6 Charts module for candlestick visualization
- `cmake`: Build system
- `build-essential`: C++ compiler and build tools
- `python3.12-venv`: Python virtual environment support

### Step 2: Setup Python Backend

Run the backend setup script:

```bash
chmod +x setup_backend.sh
./setup_backend.sh
```

This script will:
1. Create a Python virtual environment (`.venv`)
2. Install all required Python dependencies
3. Prepare the backend for running

### Step 3: Build the Qt6 GUI

Run the GUI setup script:

```bash
chmod +x setup_gui.sh
./setup_gui.sh
```

This script will:
1. Verify Qt6 packages are installed
2. Run CMake to configure the build
3. Compile the Qt6 application
4. Create the executable at `ui/desktop/build/HeNNTradingDesktop`

## Running the Application

### Option 1: Run Backend and GUI Separately

**Terminal 1 - Start the backend:**
```bash
chmod +x start_backend.sh
./start_backend.sh
```

The backend will start on `http://localhost:8000`

**Terminal 2 - Start the GUI:**
```bash
chmod +x start_gui.sh
./start_gui.sh
```

### Option 2: Manual Execution

**Start backend manually:**
```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export KMP_AFFINITY=disabled
export KMP_INIT_AT_FORK=FALSE
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Start GUI manually:**
```bash
cd ui/desktop/build
./HeNNTradingDesktop
```

## Verification

### Check Backend is Running

```bash
curl http://localhost:8000/
```

Expected output:
```json
{"status":"ok","service":"He_NN Trading Backend"}
```

### Check GUI Compilation

```bash
ls -lh ui/desktop/build/HeNNTradingDesktop
```

The executable should be present and have execute permissions.

## API Endpoints

Once the backend is running, you can access:

- **Health check:** `GET http://localhost:8000/`
- **Start training:** `POST http://localhost:8000/start_training`
- **Stop training:** `POST http://localhost:8000/stop_training`
- **Get status:** `GET http://localhost:8000/status`
- **Sync data:** `POST http://localhost:8000/sync_data`
- **Get predictions:** `GET http://localhost:8000/get_predictions?fold=0`
- **WebSocket:** `ws://localhost:8000/ws`

## Troubleshooting

### Qt6 CMake Error
If CMake still can't find Qt6 after installing packages, try:
```bash
export CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake
```

### Backend Import Errors
Make sure you're in the virtual environment:
```bash
source .venv/bin/activate
```

### GUI Won't Connect to Backend
1. Verify backend is running: `curl http://localhost:8000/`
2. Check firewall settings
3. Ensure WebSocket port 8000 is not blocked

### Permission Denied on Scripts
Make all scripts executable:
```bash
chmod +x setup_backend.sh setup_gui.sh start_backend.sh start_gui.sh
```

## Architecture

The application consists of two components:

1. **FastAPI Backend** (`backend/app.py`)
   - REST API for training control and data sync
   - WebSocket endpoint for real-time metric streaming
   - Binance data downloader
   - Training worker with CV support

2. **Qt6 Desktop GUI** (`ui/desktop/src/`)
   - Main window with chart and control panels
   - WebSocket client for backend communication
   - Real-time metrics display
   - Candlestick chart with prediction overlays

## Next Steps

1. Install Qt6 packages (requires sudo)
2. Run `./setup_backend.sh`
3. Run `./setup_gui.sh`
4. Start backend with `./start_backend.sh`
5. Start GUI with `./start_gui.sh`

For more information about the trading pipeline, see the main [README.md](README.md).
