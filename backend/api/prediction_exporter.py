"""Prediction CSV exporter for desktop app API."""  # FIX: exporter module per spec

from __future__ import annotations  # FIX: modern type hint compatibility

import logging  # FIX: structured logging
from pathlib import Path  # FIX: cross-platform path handling
from typing import Optional  # FIX: type annotations

# FIX: Configure logging per spec
LOGGER = logging.getLogger(__name__)  # FIX: module-level logger


class PredictionExporter:
    """Helper to locate and export prediction CSV files."""  # FIX: exporter class per spec

    def __init__(self, reports_dir: Path) -> None:
        """Initialize prediction exporter.

        Args:
            reports_dir: Reports directory containing predictions subdirectory  # FIX: reports dir param
        """  # FIX: constructor docstring
        self.reports_dir = Path(reports_dir)  # FIX: store reports directory
        self.predictions_dir = self.reports_dir / "predictions"  # FIX: predictions subdirectory

    def get_latest_predictions_csv(self, fold: int = 0) -> Optional[Path]:
        """Find latest predictions CSV for specified fold.

        Args:
            fold: Fold number to retrieve predictions for  # FIX: fold param

        Returns:
            Path to latest predictions CSV or None if not found  # FIX: return path or None
        """  # FIX: method docstring
        if not self.predictions_dir.exists():  # FIX: check predictions directory exists
            LOGGER.warning(
                "Predictions directory not found",
                extra={"event": "predictions_dir_missing", "path": str(self.predictions_dir)},  # FIX: log warning
            )
            return None  # FIX: return None if directory missing

        # FIX: Find all prediction files for specified fold
        pattern = f"predictions_fold_{fold}_*.csv"  # FIX: file pattern per spec
        matching_files = sorted(
            self.predictions_dir.glob(pattern),  # FIX: glob matching files
            key=lambda p: p.stat().st_mtime,  # FIX: sort by modification time
            reverse=True,  # FIX: most recent first
        )  # FIX: find and sort matching files

        if not matching_files:  # FIX: no matching files found
            LOGGER.warning(
                "No predictions found for fold",
                extra={"event": "predictions_not_found", "fold": fold, "pattern": pattern},  # FIX: log warning
            )
            return None  # FIX: return None if no files found

        latest_file = matching_files[0]  # FIX: select most recent file
        LOGGER.info(
            "Found latest predictions file",
            extra={"event": "predictions_found", "fold": fold, "path": str(latest_file)},  # FIX: log success
        )
        return latest_file  # FIX: return path to latest file


# FIX: Export exporter class
__all__ = ["PredictionExporter"]  # FIX: module exports
