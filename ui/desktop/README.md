# He_NN Trading Desktop GUI (Qt6)

Qt6-based desktop application for controlling the He_NN trading backend and visualizing predictions.

## Components

- **MainWindow** - Main application window with layout management
- **ChartWidget** - Candlestick chart with prediction overlays (Qt Charts)
- **MetricsWidget** - Real-time metrics display (AUC, DirAcc, Brier, etc.)
- **ControlWidget** - Training control panel (start/stop, parameters)
- **WebSocketClient** - Backend communication via WebSocket

## Building

From repository root:

```bash
./setup_gui.sh
```

Or manually:

```bash
cd ui/desktop
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Running

From repository root:

```bash
./start_gui.sh
```

Or manually:

```bash
cd ui/desktop/build
./HeNNTradingDesktop
```

## Requirements

- Qt 6.4+ with modules: Core, Widgets, Network, WebSockets, Charts
- CMake 3.16+
- C++17 compiler

Install on Ubuntu:

```bash
sudo apt-get install -y qt6-base-dev qt6-websockets-dev qt6-charts-dev cmake build-essential
```

## Architecture

```
MainWindow
├── ChartWidget (candlestick + predictions)
├── MetricsWidget (real-time metrics)
├── ControlWidget (training controls)
└── WebSocketClient (backend connection)
```

## Backend Communication

- **REST API:** `http://localhost:8000`
- **WebSocket:** `ws://localhost:8000/ws`

The GUI expects the FastAPI backend to be running. Start it with:

```bash
cd ../..  # Back to repo root
./start_backend.sh
```

## Features

### Chart
- OHLCV candlestick display (green=up, red=down)
- Prediction scatter overlay (orange)
- Conformal prediction bands (blue)
- Auto-scaling axes
- Datetime x-axis

### Metrics
Real-time display of:
- **Classification:** AUC, DirAcc, Brier, ECE
- **Probabilistic:** NLL, PIT KS p-value
- **Skill tests:** MZ intercept, slope, F p-value
- **Conformal:** Coverage, width

### Controls
- Timeframe selection (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- Forecast horizon (1-100)
- Calibration method (abs, std_gauss)
- p_up source (cdf, logit)
- Cross-validation toggle
- Start/stop training buttons

### Status Bar
- Connection status
- Last update timestamp
- Training progress

## Source Files

| File | Purpose |
|------|---------|
| `src/main.cpp` | Application entry point |
| `src/mainwindow.{h,cpp}` | Main window and layout |
| `src/chartwidget.{h,cpp}` | Candlestick chart |
| `src/metricswidget.{h,cpp}` | Metrics panel |
| `src/controlwidget.{h,cpp}` | Control panel |
| `src/websocketclient.{h,cpp}` | WebSocket client |
| `CMakeLists.txt` | Build configuration |

## License

Same as parent project.
