from __future__ import annotations

import json
import logging
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("fold_id", "event", "index_hash"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    level_value = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level_value, handlers=[handler])


__all__ = ["configure_logging"]
