from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional


def _qualname(obj: Any) -> Optional[str]:
    try:
        module = inspect.getmodule(obj)
        if module is None or not hasattr(obj, "__name__"):
            return None
        return f"{module.__name__}.{obj.__name__}"
    except Exception:  # pragma: no cover - defensive
        return None


def label_factory_location() -> Optional[str]:
    """
    Return the import path to the canonical label construction function if available.
    """
    try:
        module = importlib.import_module("src.data.labels")
    except ModuleNotFoundError:
        return None
    func = getattr(module, "make_labels", None)
    qualname = _qualname(func) if func else None
    if qualname and hasattr(module, "__file__"):
        return f"{qualname} ({module.__file__})"
    if func:
        return qualname
    return getattr(module, "__file__", None)


def forecast_horizon_owner(config: Any) -> str:
    """
    Describe which config dataclass owns the forecast horizon attribute.
    """
    if hasattr(config, "data") and hasattr(config.data, "forecast_horizon"):
        return "data.forecast_horizon"
    if hasattr(config, "model") and hasattr(config.model, "forecast_horizon"):
        return "model.forecast_horizon"
    if hasattr(config, "training") and hasattr(config.training, "forecast_horizon"):
        return "training.forecast_horizon"
    return "unknown"


def describe_model_heads(model: Any) -> Dict[str, str]:
    """
    Provide a simple mapping of model output heads discovered on the module.
    """
    heads: Dict[str, str] = {}
    for attr in ("mu_head", "logvar_head", "logit_head", "prob_head"):
        if hasattr(model, attr):
            heads[attr] = model.__class__.__name__
    if hasattr(model, "forward"):
        signature = inspect.signature(model.forward)
        heads["forward_signature"] = str(signature)
    return heads


__all__ = [
    "label_factory_location",
    "forecast_horizon_owner",
    "describe_model_heads",
]
