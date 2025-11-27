from __future__ import annotations

from dataclasses import replace

from src.config import APP_CONFIG, DATA
from src.utils.repo import forecast_horizon_owner, label_factory_location


def test_forecast_horizon_owner_detects_data() -> None:
    config = APP_CONFIG
    owner = forecast_horizon_owner(config)
    assert owner == "data.forecast_horizon"


def test_label_factory_location_returns_str_or_none() -> None:
    location = label_factory_location()
    assert location is None or isinstance(location, str)
