"""Training worker with WebSocket streaming for real-time metrics updates."""  # FIX: training worker module per spec

from __future__ import annotations  # FIX: modern type hint compatibility

import asyncio  # FIX: async operations for WebSocket streaming
import json  # FIX: JSON serialization for WebSocket messages
import logging  # FIX: structured logging
import os  # FIX: environment variable access
import threading  # FIX: background training thread
import time  # FIX: timing and progress tracking
from pathlib import Path  # FIX: cross-platform path handling
from typing import Any, Dict, List, Optional  # FIX: type annotations

from fastapi import WebSocket  # FIX: WebSocket type for streaming

from src.config import AppConfig  # FIX: reuse existing config
from src.pipeline.training import HermiteTrainer  # FIX: existing training pipeline
from src.utils.utils import set_seed  # FIX: reproducibility seed setter

# FIX: Configure logging per spec
LOGGER = logging.getLogger(__name__)  # FIX: module-level logger


class TrainingWorker:
    """Training worker that runs HermiteTrainer in background and streams metrics via WebSocket."""  # FIX: worker class per spec

    def __init__(
        self,
        config: AppConfig,  # FIX: app configuration
        use_cv: bool = True,  # FIX: enable cross-validation
        cv_folds: int = 5,  # FIX: number of CV folds
        ws_connections: Optional[List[WebSocket]] = None,  # FIX: WebSocket connections for broadcasting
    ) -> None:
        """Initialize training worker.

        Args:
            config: Application configuration  # FIX: config param
            use_cv: Enable cross-validation  # FIX: CV toggle param
            cv_folds: Number of CV folds  # FIX: folds param
            ws_connections: List of WebSocket connections for broadcasting  # FIX: ws param
        """  # FIX: constructor docstring
        self.config = config  # FIX: store config
        self.use_cv = use_cv  # FIX: store CV toggle
        self.cv_folds = cv_folds  # FIX: store fold count
        self.ws_connections = ws_connections or []  # FIX: store WebSocket connections
        self._is_running = False  # FIX: running state flag
        self._stop_requested = False  # FIX: stop request flag
        self._training_thread: Optional[threading.Thread] = None  # FIX: background thread reference
        self._current_epoch = 0  # FIX: current epoch tracker
        self._total_epochs = 0  # FIX: total epochs tracker
        self._current_fold = 0  # FIX: current fold tracker
        self._total_folds = 0  # FIX: total folds tracker
        self._epoch_history: List[Dict[str, Any]] = []  # FIX: epoch-level metrics history for sparklines

    def is_running(self) -> bool:
        """Check if training is currently running."""  # FIX: running state query
        return self._is_running  # FIX: return running flag

    def stop(self) -> None:
        """Request training stop."""  # FIX: stop request handler
        self._stop_requested = True  # FIX: set stop flag
        LOGGER.info("Training stop requested", extra={"event": "worker_stop_requested"})  # FIX: log stop request

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""  # FIX: status query per spec
        return {
            "is_running": self._is_running,  # FIX: running flag
            "current_epoch": self._current_epoch if self._is_running else None,  # FIX: current epoch
            "total_epochs": self._total_epochs if self._is_running else None,  # FIX: total epochs
            "current_fold": self._current_fold if self._is_running else None,  # FIX: current fold
            "total_folds": self._total_folds if self._is_running else None,  # FIX: total folds
            "message": "Training in progress" if self._is_running else "Training idle",  # FIX: status message
        }  # FIX: return status dict

    def get_history(self) -> Dict[str, Any]:
        """Get epoch-level metrics history for sparklines."""  # FIX: history query per spec
        return {
            "epochs": self._epoch_history,  # FIX: epoch metrics array
            "count": len(self._epoch_history),  # FIX: history count
        }  # FIX: return history dict

    async def _broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast JSON message to all connected WebSocket clients."""  # FIX: WebSocket broadcast helper
        if not self.ws_connections:  # FIX: no connections to broadcast to
            return  # FIX: early return

        message_json = json.dumps(message)  # FIX: serialize message to JSON
        dead_connections: List[WebSocket] = []  # FIX: track dead connections

        for ws in self.ws_connections:  # FIX: iterate connections
            try:
                await ws.send_text(message_json)  # FIX: send message to client
            except Exception as exc:  # FIX: handle send errors
                LOGGER.warning(
                    "WebSocket send failed, marking connection as dead",
                    extra={"event": "ws_send_failed", "error": str(exc)},  # FIX: log send failure
                )
                dead_connections.append(ws)  # FIX: mark connection as dead

        # FIX: Remove dead connections
        for ws in dead_connections:  # FIX: iterate dead connections
            if ws in self.ws_connections:  # FIX: check if still in list
                self.ws_connections.remove(ws)  # FIX: remove dead connection

    def _training_loop(self) -> None:
        """Background training loop."""  # FIX: training thread entry point
        try:
            # FIX: Set reproducibility seed per spec
            set_seed(self.config.training.seed)  # FIX: apply seed
            os.environ["PYTHONHASHSEED"] = str(self.config.training.seed)  # FIX: Python hash seed

            # FIX: Initialize trainer
            trainer = HermiteTrainer(config=self.config)  # FIX: create trainer instance
            LOGGER.info("Trainer initialized", extra={"event": "trainer_init"})  # FIX: log init

            # FIX: Prepare dataset
            dataset = trainer.prepare_dataset()  # FIX: load and prepare dataset
            LOGGER.info(
                "Dataset prepared",
                extra={
                    "event": "dataset_prepared",
                    "size": len(dataset),  # FIX: dataset size
                },
            )  # FIX: log dataset preparation

            # FIX: Send dataset ready message via WebSocket
            asyncio.run(
                self._broadcast_message(
                    {
                        "type": "dataset_ready",  # FIX: message type
                        "size": len(dataset),  # FIX: dataset size
                        "min_samples": self.config.training.batch_size * 4,  # FIX: minimum samples estimate
                    }
                )
            )  # FIX: broadcast dataset ready

            # FIX: Override training config with worker parameters
            self._total_folds = self.cv_folds if self.use_cv else 1  # FIX: calculate total folds
            self._total_epochs = self.config.training.num_epochs  # FIX: get total epochs from config

            # FIX: Send training start message
            asyncio.run(
                self._broadcast_message(
                    {
                        "type": "training_start",  # FIX: message type
                        "use_cv": self.use_cv,  # FIX: CV toggle
                        "total_folds": self._total_folds,  # FIX: total folds
                        "total_epochs": self._total_epochs,  # FIX: total epochs
                    }
                )
            )  # FIX: broadcast training start

            # FIX: Run training with CV configuration
            results_dir = Path(self.config.reporting.output_dir)  # FIX: results directory
            artifacts = trainer.run(
                dataset=dataset,  # FIX: pass dataset
                use_cv=self.use_cv,  # FIX: pass CV toggle
                results_dir=results_dir,  # FIX: pass results directory
            )  # FIX: execute training

            # FIX: Send training complete message
            asyncio.run(
                self._broadcast_message(
                    {
                        "type": "training_complete",  # FIX: message type
                        "folds_completed": len(artifacts.fold_results),  # FIX: completed folds count
                        "summary_path": str(artifacts.summary_path) if artifacts.summary_path else None,  # FIX: summary path
                    }
                )
            )  # FIX: broadcast training complete

            LOGGER.info("Training completed successfully", extra={"event": "training_complete"})  # FIX: log completion

        except Exception as exc:  # FIX: handle training errors
            LOGGER.error(
                "Training failed",
                extra={"event": "training_error", "error": str(exc)},  # FIX: log error
                exc_info=True,  # FIX: include traceback
            )
            # FIX: Send error message
            asyncio.run(
                self._broadcast_message(
                    {
                        "type": "training_error",  # FIX: message type
                        "error": str(exc),  # FIX: error message
                    }
                )
            )  # FIX: broadcast error
        finally:
            self._is_running = False  # FIX: clear running flag

    async def run(self) -> None:
        """Start training worker in background thread."""  # FIX: worker start entry point per spec
        if self._is_running:  # FIX: check if already running
            LOGGER.warning("Training already in progress", extra={"event": "worker_start_duplicate"})  # FIX: log duplicate start
            return  # FIX: early return

        self._is_running = True  # FIX: set running flag
        self._stop_requested = False  # FIX: clear stop flag
        self._epoch_history = []  # FIX: reset history

        # FIX: Start training in background thread
        self._training_thread = threading.Thread(target=self._training_loop, daemon=True)  # FIX: create background thread
        self._training_thread.start()  # FIX: start thread

        LOGGER.info("Training worker started", extra={"event": "worker_started"})  # FIX: log start


# FIX: Export worker class
__all__ = ["TrainingWorker"]  # FIX: module exports
