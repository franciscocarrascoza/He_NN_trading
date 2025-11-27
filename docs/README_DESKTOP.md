# He_NN Trading Desktop Application

**Prediction-First** desktop application for Hermite NN probabilistic forecasting with real-time training metrics streaming and candlestick chart visualization.

---

## Overview

This desktop application provides a native Qt6 GUI for training, monitoring, and visualizing Hermite NN model predictions on cryptocurrency market data. **No trading or order execution functionality is included** — this is strictly a prediction and model evaluation tool.

### Key Features

- **Native Qt6 Desktop GUI** (C++)
  - MT5-like dockable/resizable layout
  - Real-time OHLCV candlestick charts with prediction overlays
  - Prediction arrows (`μ`) and conformal interval bands
  - Live metrics panel with AUC, DirAcc, Brier, ECE, NLL, MZ, PIT, conformal coverage/width
  - Training control panel (start/stop, parameters, no trading UI)
  - Layout editor with percent-based sizing and JSON export

- **Python FastAPI Backend**
  - REST API endpoints for training control and status queries
  - WebSocket streaming for real-time epoch metrics
  - Incremental Binance historical data downloader with Parquet storage
  - Training worker with conformal calibration and isotonic regression
  - Prediction CSV exporter
  - HPO adapter skeleton (Optuna integration ready, not enabled by default)

- **Prediction-Focused Metrics**
  - AUC, directional accuracy, Brier score (+ decomposition)
  - Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
  - Negative Log-Likelihood (NLL)
  - Mincer–Zarnowitz regression (intercept, slope, F-test p-value)
  - Probability Integral Transform (PIT) z-scores with KS test
  - Conformal prediction coverage and interval width

---

## System Requirements

### Backend (Python)
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **FastAPI**: 0.100+
- **Uvicorn**: 0.20+
- **Pandas**, **NumPy**, **SciPy**, **scikit-learn**
- **Requests** (for Binance REST API)

### Frontend (Qt6)
- **Qt6**: 6.2+ (Core, Widgets, Network, WebSockets, Charts)
- **CMake**: 3.16+
- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+

### Platform Support
- **Linux**: Tested on Ubuntu 20.04+
- **macOS**: Should work (Qt6 cross-platform)
- **Windows**: Should work (Qt6 cross-platform)

---

## Installation & Build

### 1. Python Backend

```bash
# FIX: Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # FIX: Linux/macOS
# .venv\Scripts\activate  # FIX: Windows

# FIX: Install Python dependencies
pip install --upgrade pip
pip install torch numpy pandas scipy scikit-learn matplotlib requests pyyaml pytest fastapi uvicorn websockets pyarrow

# FIX: Verify installation
pytest tests/test_conformal_desktop.py tests/test_isotonic_calibration_desktop.py tests/test_pit_z_desktop.py
```

### 2. Qt6 Desktop GUI

```bash
# FIX: Install Qt6 (Linux example - Ubuntu/Debian)
sudo apt-get update
sudo apt-get install qt6-base-dev qt6-charts-dev qt6-websockets-dev cmake build-essential

# FIX: Build desktop application
cd ui/desktop
mkdir build && cd build
cmake ..
cmake --build .

# FIX: Executable will be in build/ directory
./HeNNTradingDesktop
```

**macOS** (Homebrew):
```bash
brew install qt@6 cmake
# FIX: Update CMakeLists.txt Qt6_DIR if needed
cd ui/desktop && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$(brew --prefix qt@6) ..
cmake --build .
```

**Windows** (Qt installer + Visual Studio):
- Install Qt6 from [qt.io](https://www.qt.io/download-qt-installer)
- Use Qt Creator or CMake with Visual Studio
- Build from `ui/desktop/` directory

---

## Running the Application

### Quick Start - Backend

**Single command to start the backend** (from repository root):

```bash
source ~/software/he_nn_env/bin/activate && cd /home/francisco/work/AI/He_NN_trading && ./start_backend.sh
```

Or simply:

```bash
./start_backend.sh
```

Backend will be available at `http://localhost:8000`. Check health: `curl http://localhost:8000/`

**Note**: The virtual environment has been relocated to `~/software/he_nn_env/` and historical data to `~/trading/he_nn_data/` to keep the repository clean.

### Quick Start - GUI

```bash
./start_gui.sh
```

Or manually:

```bash
cd ui/desktop/build && ./HeNNTradingDesktop
```

The GUI will automatically connect to WebSocket endpoint `ws://localhost:8000/ws`. The startup script checks if the backend is running before launching.

### Step 3: Sync Data (First Run)

Before training, you need historical candle data:

**Option A**: Use Desktop GUI
- Click **Data Sync** button in control panel (if implemented)

**Option B**: Use REST API
```bash
curl -X POST http://localhost:8000/sync_data \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "1h"}'
```

This will download up to 8 years of historical 1h candles from Binance and store them in `data/ohlcv/BTCUSDT/1h/` as Parquet partitions.

### Step 4: Start Training

**Option A**: Desktop GUI Control Panel
- Select timeframe, forecast horizon, calibration method, p_up source
- Enable/disable cross-validation
- Click **Start Training**

**Option B**: REST API
```bash
curl -X POST http://localhost:8000/start_training \
  -H "Content-Type: application/json" \
  -d '{"use_cv": true, "cv_folds": 5, "forecast_horizon": 1, "calibration_method": "std_gauss", "p_up_source": "cdf", "seed": 42}'
```

Training will stream real-time metrics via WebSocket to the desktop GUI. Epoch-level updates include AUC, DirAcc, Brier, ECE, NLL, MZ stats, PIT KS p-value, and conformal coverage/width.

### Step 5: View Results

- **Desktop GUI**: Real-time metrics panel updates during training
- **Chart**: Displays OHLCV candles + prediction arrows + conformal bands after training completes
- **CSV Export**: Download predictions via REST API:
  ```bash
  curl http://localhost:8000/get_predictions?fold=0 --output predictions_fold_0.csv
  ```
- **Summary Report**: Download `summary.json`:
  ```bash
  curl http://localhost:8000/download_report --output summary.json
  ```

---

## Configuration

### Backend Config: `config/app.yaml`

```yaml
data:
  storage_path: "./data"                # FIX: Parquet storage path
  poll_interval_seconds: 60             # FIX: REST polling interval for live updates
  max_backfill_years: 8                 # FIX: maximum historical backfill window
  min_samples_to_train: 8192            # FIX: minimum candles before training
  min_calib_size: 256                   # FIX: minimum calibration set size

training:
  batch_size: 512                       # FIX: training batch size
  val_frac: 0.2                         # FIX: validation fraction
  test_frac: 0.1                        # FIX: test fraction (not used in CV)
  seed: 42                              # FIX: reproducibility seed
  forecast_horizon: 1                   # FIX: default forecast horizon
  use_cv: true                          # FIX: enable cross-validation by default
  cv_folds: 5                           # FIX: number of CV folds

evaluation:
  conformal_alpha: 0.10                 # FIX: conformal miscoverage level (90% coverage)
  residual_kind: "std_gauss"            # FIX: conformal residual type (abs or std_gauss)
  p_up_source: "cdf"                    # FIX: probability source (cdf or logit)

ui:
  default_timeframe: "1h"               # FIX: default chart timeframe
  auto_refresh_seconds: 60              # FIX: auto-refresh interval
  layout_file: "config/ui_layout.json"  # FIX: UI layout JSON path

optuna:
  enabled: false                        # FIX: Optuna disabled by default (skeleton only)
  storage_url: "sqlite:///storage/optuna_studies.db"  # FIX: Optuna storage URL
```

### UI Layout Config: `config/ui_layout.json`

```json
{
  "version": "1.0",
  "panels": {
    "chart": {
      "percent_width": 60.0,
      "percent_height": 70.0,
      "position": "left"
    },
    "metrics": {
      "percent_width": 40.0,
      "percent_height": 30.0,
      "position": "right-top"
    },
    ...
  },
  "fonts": { ... },
  "colors": { ... }
}
```

You can edit this JSON manually or use the built-in **Layout Editor** in the GUI (if implemented) to adjust panel sizes.

---

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/start_training` | POST | Start training worker |
| `/stop_training` | POST | Stop training worker |
| `/status` | GET | Query training status |
| `/sync_data` | POST | Trigger data sync (Binance) |
| `/get_predictions?fold=N` | GET | Download predictions CSV for fold N |
| `/get_history` | GET | Get epoch-level metrics history |
| `/download_report` | GET | Download `summary.json` |
| `/ws` | WebSocket | Streaming training updates |

See `docs/ws_schema.md` for WebSocket message schema.

---

## WebSocket Message Types

The WebSocket endpoint `/ws` streams JSON messages. See `docs/ws_schema.md` for full schema details.

Example epoch metrics message:
```json
{
  "type": "epoch_metrics",
  "fold_id": 0,
  "epoch": 10,
  "train_loss": 0.1234,
  "val_loss": 0.2345,
  "val_auc": 0.6789,
  "val_brier": 0.1234,
  "DirAcc": 0.5678,
  "ECE": 0.0456,
  "MZ_intercept": 0.0012,
  "MZ_slope": 0.9987,
  "MZ_F_p": 0.3456,
  "PIT_KS_p": 0.4567,
  "conformal_coverage": 0.9012,
  "conformal_width": 0.1234
}
```

---

## Prediction CSV Format

Downloaded predictions CSV (`/get_predictions?fold=0`) contains:

```
timestamp,y,mu,sigma,logit,p_up_raw,p_up_cal,conformal_p,pit_z
2023-01-01T00:00:00Z,0.0123,-0.0045,0.0234,0.5678,0.4912,0.5123,0.8765,0.1234
...
```

- `timestamp`: Prediction timestamp (ISO 8601 UTC)
- `y`: Actual log-return
- `mu`: Predicted mean log-return
- `sigma`: Predicted standard deviation
- `logit`: Raw logits for classification head
- `p_up_raw`: Raw probability (CDF or logit-based)
- `p_up_cal`: Calibrated probability (isotonic)
- `conformal_p`: Conformal p-value
- `pit_z`: PIT z-score

---

## Testing

Run test suite:

```bash
# FIX: From repository root
pytest tests/test_conformal_desktop.py tests/test_isotonic_calibration_desktop.py tests/test_pit_z_desktop.py tests/test_data_sync_desktop.py -v
```

Expected results:
- **Conformal coverage tests**: Pass within ±2% tolerance
- **Isotonic calibration test**: ECE reduction confirmed
- **PIT z-score test**: KS p-value > 0.05
- **Data sync tests**: Index creation and deduplication verified

---

## Troubleshooting

### Backend Connection Issues

**Problem**: Desktop GUI shows "Disconnected"

**Solution**:
1. Verify backend is running: `curl http://localhost:8000/`
2. Check WebSocket endpoint: `wscat -c ws://localhost:8000/ws` (if `wscat` installed)
3. Check backend logs for errors

### Training Fails with "Insufficient Calibration Sample"

**Problem**: Training stops with calibration warning

**Solution**:
1. Increase data: Sync more historical candles
2. Reduce `min_calib_size` in `config/app.yaml` (not recommended below 128)
3. Reduce `cv_folds` to allow larger calibration sets per fold

### Qt6 Build Errors

**Problem**: CMake can't find Qt6

**Solution**:
1. Ensure Qt6 is installed: `apt-cache policy qt6-base-dev` (Linux)
2. Set `CMAKE_PREFIX_PATH`: `cmake -DCMAKE_PREFIX_PATH=/path/to/qt6 ..`
3. Check Qt6 version: `qmake6 --version` should show 6.2+

---

## Enabling Optuna (Future)

The HPO adapter skeleton is ready but disabled by default. To enable Optuna:

1. Install Optuna: `pip install optuna`
2. Set `optuna.enabled: true` in `config/app.yaml`
3. Implement trial loop in `backend/hpo/hpo_adapter.py` (see `docs/optuna_ready.md`)
4. Add REST endpoint `/hpo/suggest` handler to `backend/app.py`
5. Run hyperparameter search via API or CLI

See `docs/optuna_ready.md` for detailed integration guide.

---

## Folder Structure

```
He_NN_trading/
├── backend/                          # FIX: Python backend
│   ├── app.py                        # FIX: FastAPI main application
│   ├── api/                          # FIX: API utilities
│   │   └── prediction_exporter.py    # FIX: prediction CSV exporter
│   ├── data/                         # FIX: data ingestion
│   │   └── downloader.py             # FIX: Binance downloader with Parquet
│   ├── hpo/                          # FIX: HPO adapter skeleton
│   │   └── hpo_adapter.py            # FIX: Optuna integration skeleton
│   └── workers/                      # FIX: training workers
│       └── training_worker.py        # FIX: background training with streaming
├── config/                           # FIX: configuration files
│   ├── app.yaml                      # FIX: desktop app config
│   ├── defaults.yaml                 # FIX: model/training defaults
│   └── ui_layout.json                # FIX: UI layout percent-based config
├── data/                             # FIX: OHLCV Parquet storage
│   ├── index.sqlite                  # FIX: metadata index
│   └── ohlcv/<symbol>/<tf>/YYYY/MM/  # FIX: partitioned Parquet files
├── docs/                             # FIX: documentation
│   ├── README_DESKTOP.md             # FIX: this file
│   ├── ws_schema.md                  # FIX: WebSocket message schema
│   └── optuna_ready.md               # FIX: Optuna integration guide
├── reports/                          # FIX: training outputs
│   ├── predictions/                  # FIX: prediction CSVs
│   ├── calibrators/                  # FIX: saved isotonic calibrators
│   ├── plots/                        # FIX: diagnostics plots
│   └── summary.json                  # FIX: aggregated results
├── src/                              # FIX: existing Python pipeline
│   ├── config/                       # FIX: config loaders
│   ├── data/                         # FIX: dataset builders
│   ├── eval/                         # FIX: conformal, diagnostics
│   ├── features/                     # FIX: feature engineering
│   ├── models/                       # FIX: Hermite forecaster
│   ├── pipeline/                     # FIX: training pipeline
│   ├── reporting/                    # FIX: plots and exports
│   └── utils/                        # FIX: utilities (PIT, PT, seed)
├── storage/                          # FIX: Optuna storage (when enabled)
│   └── optuna_studies.db             # FIX: Optuna SQLite database
├── tests/                            # FIX: test suite
│   ├── test_conformal_desktop.py     # FIX: conformal coverage tests
│   ├── test_isotonic_calibration_desktop.py  # FIX: isotonic ECE reduction test
│   ├── test_pit_z_desktop.py         # FIX: PIT z normality test
│   └── test_data_sync_desktop.py     # FIX: data sync deduplication test
└── ui/desktop/                       # FIX: Qt6 desktop GUI
    ├── CMakeLists.txt                # FIX: CMake build config
    ├── config/                       # FIX: UI config (if separate from root)
    └── src/                          # FIX: C++ source
        ├── main.cpp                  # FIX: entry point
        ├── mainwindow.{h,cpp}        # FIX: main window
        ├── chartwidget.{h,cpp}       # FIX: chart with overlays
        ├── metricswidget.{h,cpp}     # FIX: metrics panel
        ├── controlwidget.{h,cpp}     # FIX: control panel (no trading)
        └── websocketclient.{h,cpp}   # FIX: WebSocket client
```

---

## License

See main repository `README.md` for license and contribution guidelines.

---

## Contact & Support

For questions or issues, open a GitHub issue or contact the maintainer.

**Note**: This is a prediction and model evaluation tool. **No trading or order execution features are included.** Trading agent integration is planned for future releases.
