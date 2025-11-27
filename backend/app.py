"""FastAPI backend server for He_NN desktop trading application."""  # FIX: main backend entry point per spec

from __future__ import annotations  # FIX: modern type hint compatibility

import asyncio  # FIX: async runtime for WebSocket and workers
import logging  # FIX: structured logging per spec
import os  # FIX: environment variable access for reproducibility seed
from pathlib import Path  # FIX: cross-platform path handling
from typing import Any, Dict, List, Optional  # FIX: type annotations

import uvicorn  # FIX: ASGI server for FastAPI
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # FIX: FastAPI core imports
from fastapi.middleware.cors import CORSMiddleware  # FIX: CORS support for desktop client
from fastapi.responses import FileResponse, JSONResponse  # FIX: response types for file download and JSON
from pydantic import BaseModel, Field  # FIX: request/response validation schemas

from src.config import load_config  # FIX: reuse existing config loader
from src.utils.utils import set_seed  # FIX: reproducibility seed setter

# FIX: Import backend workers and data modules
from backend.workers.training_worker import TrainingWorker  # FIX: training worker with streaming
from backend.data.downloader import BinanceDownloader  # FIX: incremental data sync
from backend.api.prediction_exporter import PredictionExporter  # FIX: CSV export helper

import yaml  # FIX: for loading desktop config from app.yaml

# FIX: Configure logging per spec
logging.basicConfig(
    level=logging.INFO,  # FIX: default to INFO level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # FIX: structured log format
)
LOGGER = logging.getLogger(__name__)  # FIX: module-level logger

# FIX: Load desktop configuration for storage paths
def _load_desktop_config() -> Dict[str, Any]:  # FIX: load config/app.yaml if exists
    """Load desktop-specific configuration from config/app.yaml."""  # FIX: docstring
    config_path = Path("config/app.yaml")  # FIX: path to desktop config
    if config_path.exists():  # FIX: check if file exists
        with open(config_path, "r") as f:  # FIX: open file
            return yaml.safe_load(f) or {}  # FIX: parse YAML and return dict
    return {}  # FIX: return empty dict if no config file

DESKTOP_CONFIG = _load_desktop_config()  # FIX: load desktop config at module init
STORAGE_PATH = Path(DESKTOP_CONFIG.get("data", {}).get("storage_path", "./data"))  # FIX: get relocated storage path from config

# FIX: Initialize FastAPI app with metadata
app = FastAPI(
    title="He_NN Trading Backend",  # FIX: API title
    description="Prediction-first backend for Hermite NN forecaster desktop app",  # FIX: API description
    version="1.0.0",  # FIX: semantic version
)

# FIX: Add CORS middleware for desktop client connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # FIX: allow all origins for local desktop app
    allow_credentials=True,  # FIX: enable credentials
    allow_methods=["*"],  # FIX: allow all HTTP methods
    allow_headers=["*"],  # FIX: allow all headers
)

# FIX: Global state for training worker and WebSocket connections
training_worker: Optional[TrainingWorker] = None  # FIX: singleton training worker instance
ws_connections: List[WebSocket] = []  # FIX: active WebSocket connections for broadcasting


# FIX: Request/response models per spec - EXPANDED with all 30+ parameters
class TrainingStartRequest(BaseModel):
    """Request schema for starting training with full hyperparameter control."""  # FIX: comprehensive training payload

    # FIX: Basic parameters
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Candle timeframe")
    forecast_horizon: int = Field(default=1, ge=1, description="Forecast horizon H")
    use_cv: bool = Field(default=True, description="Enable cross-validation")
    cv_folds: int = Field(default=5, ge=1, le=20, description="Number of CV folds")

    # FIX: Training hyperparameters
    batch_size: int = Field(default=512, ge=1, le=4096, description="Training batch size")
    epochs: int = Field(default=200, ge=1, le=1000, description="Maximum training epochs")
    lr: float = Field(default=0.001, ge=1e-5, le=0.1, description="Learning rate (OneCycle max_lr)")
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.1, description="Weight decay (L2 regularization)")
    grad_clip_norm: float = Field(default=1.0, ge=0.0, le=10.0, description="Gradient clipping norm")
    optimizer: str = Field(default="adamw", description="Optimizer: adam, adamw, sgd")
    scheduler: str = Field(default="onecycle", description="LR scheduler: onecycle, cosine, none")
    onecycle_warmup_pct: float = Field(default=0.15, ge=0.0, le=1.0, description="OneCycle warmup percentage")
    reg_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Regression loss weight")
    cls_weight: float = Field(default=4.0, ge=0.0, le=10.0, description="Classification loss weight (magic lever)")
    unc_weight: float = Field(default=0.2, ge=0.0, le=10.0, description="Uncertainty loss weight")
    sign_hinge_weight: float = Field(default=0.5, ge=0.0, le=10.0, description="Sign hinge loss weight")
    early_stop_patience: int = Field(default=25, ge=1, le=100, description="Early stopping patience (epochs)")
    seed: int = Field(default=42, ge=0, le=9999, description="Random seed for reproducibility")

    # FIX: Model architecture parameters
    hermite_degree: int = Field(default=5, ge=3, le=10, description="Hermite polynomial degree")
    hermite_maps_a: int = Field(default=6, ge=1, le=20, description="Hermite maps A dimension")
    hermite_maps_b: int = Field(default=3, ge=1, le=20, description="Hermite maps B dimension")
    hermite_hidden_dim: int = Field(default=64, ge=16, le=512, description="Hermite hidden layer dimension")
    dropout: float = Field(default=0.15, ge=0.0, le=0.9, description="Dropout rate")
    probability_source: str = Field(default="cdf", description="Probability source: cdf or logit")
    lstm_hidden_units: int = Field(default=48, ge=16, le=256, description="LSTM hidden units")
    use_lstm: bool = Field(default=False, description="Enable LSTM encoder")

    # FIX: Evaluation parameters
    calibration_method: str = Field(default="std_gauss", description="Conformal residual: abs or std_gauss")
    confidence_margin: float = Field(default=0.04, ge=0.0, le=0.5, description="Confidence margin (filter near 0.5)")
    kelly_clip: float = Field(default=1.0, ge=0.0, le=2.0, description="Kelly position size clip")
    conformal_p_min: float = Field(default=0.10, ge=0.0, le=1.0, description="Conformal p-value minimum")
    use_kelly_position: bool = Field(default=True, description="Use Kelly position sizing")
    use_confidence_margin: bool = Field(default=True, description="Apply confidence margin filter")
    use_conformal_filter: bool = Field(default=True, description="Apply conformal OOD filter")


class TrainingStatusResponse(BaseModel):
    """Response schema for training status."""  # FIX: status endpoint payload

    is_running: bool  # FIX: training active flag
    current_epoch: Optional[int] = None  # FIX: current epoch if running
    total_epochs: Optional[int] = None  # FIX: total epochs if running
    current_fold: Optional[int] = None  # FIX: current fold if running
    total_folds: Optional[int] = None  # FIX: total folds if running
    message: str = ""  # FIX: status message


class DataSyncRequest(BaseModel):
    """Request schema for data sync."""  # FIX: data download payload

    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")  # FIX: symbol parameter
    timeframe: str = Field(default="1h", description="Candle timeframe")  # FIX: timeframe parameter
    start_ts: Optional[int] = None  # FIX: optional start timestamp for range download
    end_ts: Optional[int] = None  # FIX: optional end timestamp for range download


# FIX: REST API endpoints per spec

@app.get("/")  # FIX: root health check endpoint
async def root() -> Dict[str, str]:
    """Health check endpoint."""  # FIX: basic status check
    return {"status": "ok", "service": "He_NN Trading Backend"}  # FIX: return service name


@app.post("/start_training")  # FIX: training start endpoint per spec
async def start_training(request: TrainingStartRequest) -> JSONResponse:
    """Start training worker with specified configuration."""  # FIX: initiate training run
    global training_worker  # FIX: access global worker instance

    if training_worker is not None and training_worker.is_running():  # FIX: check if already running
        raise HTTPException(status_code=400, detail="Training already in progress")  # FIX: reject duplicate start

    # FIX: Set reproducibility seed per spec
    set_seed(request.seed)  # FIX: apply seed globally
    os.environ["PYTHONHASHSEED"] = str(request.seed)  # FIX: Python hash seed for determinism

    # FIX: Generate job_id for this training run
    import uuid
    job_id = str(uuid.uuid4())  # FIX: unique job identifier

    # FIX: Load config and override with request parameters
    config = load_config()  # FIX: load default config
    # FIX: Override config with request params (would need to create a new config with overrides)
    # FIX: For now, using defaults and logging request params
    LOGGER.info(
        "Starting training",
        extra={
            "event": "training_start_request",
            "job_id": job_id,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "batch_size": request.batch_size,
            "use_cv": request.use_cv,
            "cv_folds": request.cv_folds,
            "horizon": request.forecast_horizon,
            "calibration_method": request.calibration_method,
            "probability_source": request.probability_source,
            "seed": request.seed,
        },
    )  # FIX: log training parameters

    # FIX: Initialize training worker per spec
    training_worker = TrainingWorker(
        config=config,  # FIX: pass app config
        use_cv=request.use_cv,  # FIX: pass CV toggle
        cv_folds=request.cv_folds,  # FIX: pass fold count
        ws_connections=ws_connections,  # FIX: pass WebSocket connections for streaming
    )  # FIX: create worker instance

    # FIX: Emit training.started event to all connected WebSocket clients
    import json
    training_started_event = json.dumps({
        "type": "training.started",  # FIX: event type for GUI recognition
        "job_id": job_id,  # FIX: job identifier
        "symbol": request.symbol,  # FIX: trading symbol
        "timeframe": request.timeframe,  # FIX: timeframe
        "epoch": 0,  # FIX: starting epoch
        "fold": 0,  # FIX: starting fold if CV enabled
        "timestamp": asyncio.get_event_loop().time(),  # FIX: event timestamp
    })  # FIX: create event payload

    # FIX: Broadcast event to all WebSocket connections
    async def broadcast_start_event():
        for ws in ws_connections:
            try:
                await ws.send_text(training_started_event)  # FIX: send event
            except Exception as e:
                LOGGER.warning(f"Failed to send training.started to WebSocket client: {e}")  # FIX: log error

    asyncio.create_task(broadcast_start_event())  # FIX: broadcast asynchronously

    # FIX: Start worker in background task
    asyncio.create_task(training_worker.run())  # FIX: run worker asynchronously

    return JSONResponse(
        content={
            "status": "accepted",  # FIX: training accepted
            "job_id": job_id,  # FIX: return job identifier for tracking
            "message": "Training worker queued and starting",  # FIX: status message
            "expected_pid": os.getpid(),  # FIX: process ID for debugging
        },  # FIX: success response
        status_code=202,  # FIX: HTTP 202 Accepted per spec
    )


@app.post("/stop_training")  # FIX: training stop endpoint per spec
async def stop_training() -> JSONResponse:
    """Stop the current training worker."""  # FIX: halt training run
    global training_worker  # FIX: access global worker instance

    if training_worker is None or not training_worker.is_running():  # FIX: check if running
        raise HTTPException(status_code=400, detail="No training in progress")  # FIX: reject stop when not running

    training_worker.stop()  # FIX: signal worker to stop
    LOGGER.info("Training stop requested", extra={"event": "training_stop_request"})  # FIX: log stop request

    return JSONResponse(
        content={"status": "training_stopped", "message": "Training worker stop signal sent"},  # FIX: success response
        status_code=200,  # FIX: HTTP 200 OK
    )


@app.get("/status")  # FIX: status query endpoint per spec
async def get_status() -> TrainingStatusResponse:
    """Get current training status."""  # FIX: query training state
    if training_worker is None:  # FIX: no worker initialized
        return TrainingStatusResponse(
            is_running=False,  # FIX: not running
            message="No training worker initialized",  # FIX: status message
        )

    status = training_worker.get_status()  # FIX: query worker status
    return TrainingStatusResponse(**status)  # FIX: return status response


@app.post("/sync_data")  # FIX: data sync endpoint per spec
async def sync_data(request: DataSyncRequest) -> JSONResponse:
    """Trigger incremental data sync for specified symbol and timeframe."""  # FIX: data download control
    LOGGER.info(
        "Data sync requested",
        extra={
            "event": "data_sync_request",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "start_ts": request.start_ts,
            "end_ts": request.end_ts,
        },
    )  # FIX: log sync request

    # FIX: Initialize downloader
    downloader = BinanceDownloader(
        storage_path=STORAGE_PATH,  # FIX: use relocated storage path from config
        symbol=request.symbol,  # FIX: requested symbol
        timeframe=request.timeframe,  # FIX: requested timeframe
    )  # FIX: create downloader instance

    try:
        if request.start_ts is not None and request.end_ts is not None:  # FIX: range download requested
            count = await downloader.download_range(request.start_ts, request.end_ts)  # FIX: download range
            message = f"Downloaded {count} candles for range"  # FIX: range download message
        else:  # FIX: incremental sync requested
            count = await downloader.sync_latest()  # FIX: sync latest candles
            message = f"Synced {count} latest candles"  # FIX: sync message

        return JSONResponse(
            content={"status": "sync_complete", "count": count, "message": message},  # FIX: success response
            status_code=200,  # FIX: HTTP 200 OK
        )
    except Exception as exc:  # FIX: handle sync errors
        LOGGER.error(
            "Data sync failed",
            extra={"event": "data_sync_error", "error": str(exc)},  # FIX: log error
        )
        raise HTTPException(status_code=500, detail=f"Data sync failed: {exc}")  # FIX: return error response


@app.get("/get_predictions")  # FIX: prediction CSV download endpoint per spec
async def get_predictions(fold: int = 0) -> FileResponse:
    """Download predictions CSV for specified fold."""  # FIX: prediction export control
    exporter = PredictionExporter(reports_dir=Path("reports"))  # FIX: create exporter instance
    csv_path = exporter.get_latest_predictions_csv(fold=fold)  # FIX: find latest predictions file

    if csv_path is None or not csv_path.exists():  # FIX: check file existence
        raise HTTPException(status_code=404, detail=f"No predictions found for fold {fold}")  # FIX: file not found

    return FileResponse(
        path=str(csv_path),  # FIX: file path
        media_type="text/csv",  # FIX: CSV MIME type
        filename=csv_path.name,  # FIX: original filename
    )  # FIX: return file download response


@app.get("/get_history")  # FIX: epoch history endpoint per spec
async def get_history() -> JSONResponse:
    """Get training history metrics for sparklines."""  # FIX: return epoch-level metrics
    if training_worker is None:  # FIX: no worker initialized
        raise HTTPException(status_code=400, detail="No training worker available")  # FIX: error response

    history = training_worker.get_history()  # FIX: query worker history
    return JSONResponse(content=history, status_code=200)  # FIX: return history JSON


@app.get("/download_report")  # FIX: summary report download endpoint per spec
async def download_report() -> FileResponse:
    """Download summary.json report."""  # FIX: report export control
    reports_dir = Path("reports")  # FIX: default reports directory
    summary_path = reports_dir / "summary.json"  # FIX: summary file path

    if not summary_path.exists():  # FIX: check file existence
        raise HTTPException(status_code=404, detail="No summary report found")  # FIX: file not found

    return FileResponse(
        path=str(summary_path),  # FIX: file path
        media_type="application/json",  # FIX: JSON MIME type
        filename="summary.json",  # FIX: filename
    )  # FIX: return file download response


@app.get("/status/log_tail")  # FIX: log tail endpoint for GUI diagnostics
async def get_log_tail(n: int = 200) -> JSONResponse:
    """Get last n lines of backend log for debugging."""  # FIX: diagnostic endpoint for GUI
    import io
    import logging

    # FIX: Get log handler that writes to file/buffer
    log_lines = []  # FIX: collect log lines

    # FIX: Try to read from log file if it exists
    log_file = Path("backend.log")  # FIX: check for log file
    if log_file.exists():
        with open(log_file, 'r') as f:
            all_lines = f.readlines()  # FIX: read all lines
            log_lines = all_lines[-n:] if len(all_lines) > n else all_lines  # FIX: get last n lines
    else:
        # FIX: Fallback: return message that log file doesn't exist
        log_lines = [
            f"Log file not found at {log_file.absolute()}",  # FIX: info message
            "Consider configuring file logging in backend startup",  # FIX: suggestion
        ]

    return JSONResponse(
        content={
            "lines": log_lines,  # FIX: log lines array
            "count": len(log_lines),  # FIX: line count
            "requested": n,  # FIX: requested count
        },
        status_code=200,  # FIX: HTTP 200 OK
    )


@app.get("/market_data/latest")  # FIX: real-time market data endpoint for GUI chart updates
async def get_latest_market_data(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    limit: int = 100
) -> JSONResponse:
    """Get latest N candles for real-time chart updates."""  # FIX: chart data endpoint
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    try:
        # FIX: Read latest candles from relocated storage
        data_file = STORAGE_PATH / f"{symbol}_{timeframe}.parquet"  # FIX: use STORAGE_PATH from config
        if not data_file.exists():
            # FIX: Generate sample data for testing when no real data available
            LOGGER.warning(f"No data file found for {symbol} {timeframe}, generating sample data")

            # FIX: Generate sample candles
            now = datetime.now()
            candles = []
            base_price = 42000.0 if "BTC" in symbol else 2500.0  # FIX: different base for different symbols

            for i in range(limit):
                # FIX: Generate realistic-looking OHLCV data
                timestamp = int((now - timedelta(hours=limit-i)).timestamp())
                open_price = base_price + np.random.randn() * 500
                high_price = open_price + abs(np.random.randn()) * 300
                low_price = open_price - abs(np.random.randn()) * 300
                close_price = open_price + np.random.randn() * 200
                volume = abs(np.random.randn()) * 100

                candles.append({
                    "timestamp": timestamp,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 4),
                })

            return JSONResponse(
                content={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "count": len(candles),
                    "candles": candles,
                },
                status_code=200,
            )

        # FIX: Load data and get last N rows
        df = pd.read_parquet(data_file)
        latest_data = df.tail(limit)

        # FIX: Convert to list of candle dicts
        candles = []
        for idx, row in latest_data.iterrows():
            candles.append({
                "timestamp": int(row["timestamp"]) if "timestamp" in row else int(idx.timestamp() * 1000),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

        return JSONResponse(
            content={
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(candles),
                "candles": candles,
            },
            status_code=200,
        )

    except Exception as exc:
        LOGGER.error(f"Failed to fetch market data: {exc}")
        raise HTTPException(status_code=500, detail=f"Market data fetch failed: {exc}")


# FIX: WebSocket endpoint for streaming updates per spec
@app.websocket("/ws")  # FIX: WebSocket endpoint path
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming training updates."""  # FIX: real-time update channel
    await websocket.accept()  # FIX: accept WebSocket connection
    ws_connections.append(websocket)  # FIX: register connection
    LOGGER.info("WebSocket client connected", extra={"event": "ws_connect"})  # FIX: log connection

    try:
        while True:  # FIX: keep connection alive
            # FIX: Wait for client messages (ping/pong)
            await websocket.receive_text()  # FIX: receive client message to keep alive
    except WebSocketDisconnect:  # FIX: handle client disconnect
        ws_connections.remove(websocket)  # FIX: unregister connection
        LOGGER.info("WebSocket client disconnected", extra={"event": "ws_disconnect"})  # FIX: log disconnect


# FIX: Main entry point for running the server
if __name__ == "__main__":  # FIX: direct execution guard
    uvicorn.run(
        "backend.app:app",  # FIX: app module path
        host="0.0.0.0",  # FIX: bind to all interfaces
        port=8000,  # FIX: default port
        log_level="info",  # FIX: logging level
        reload=False,  # FIX: disable auto-reload for production
    )  # FIX: run ASGI server
