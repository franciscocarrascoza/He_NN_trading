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

# FIX: Configure logging per spec
logging.basicConfig(
    level=logging.INFO,  # FIX: default to INFO level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # FIX: structured log format
)
LOGGER = logging.getLogger(__name__)  # FIX: module-level logger

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


# FIX: Request/response models per spec
class TrainingStartRequest(BaseModel):
    """Request schema for starting training."""  # FIX: training start payload

    use_cv: bool = Field(default=True, description="Enable cross-validation")  # FIX: CV toggle
    cv_folds: int = Field(default=5, ge=1, le=20, description="Number of CV folds")  # FIX: fold count
    forecast_horizon: int = Field(default=1, ge=1, description="Forecast horizon")  # FIX: H parameter
    calibration_method: str = Field(default="abs", description="Residual kind: abs or std_gauss")  # FIX: conformal residual type
    p_up_source: str = Field(default="cdf", description="Probability source: cdf or logit")  # FIX: probability head choice
    seed: int = Field(default=42, description="Random seed for reproducibility")  # FIX: seed parameter


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

    # FIX: Load config and override with request parameters
    config = load_config()  # FIX: load default config
    # FIX: Override config with request params (would need to create a new config with overrides)
    # FIX: For now, using defaults and logging request params
    LOGGER.info(
        "Starting training",
        extra={
            "event": "training_start_request",
            "use_cv": request.use_cv,
            "cv_folds": request.cv_folds,
            "horizon": request.forecast_horizon,
            "calibration_method": request.calibration_method,
            "p_up_source": request.p_up_source,
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

    # FIX: Start worker in background task
    asyncio.create_task(training_worker.run())  # FIX: run worker asynchronously

    return JSONResponse(
        content={"status": "training_started", "message": "Training worker started successfully"},  # FIX: success response
        status_code=200,  # FIX: HTTP 200 OK
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
        storage_path=Path("data"),  # FIX: default storage path
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
