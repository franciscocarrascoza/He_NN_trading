from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import replace

import pytest

SCRIPT_TEMPLATE = """
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import APP_CONFIG, DATA, EVALUATION, MODEL, REPORTING, TRAINING
from src.data.dataset import HermiteDataset
from src.pipeline import HermiteTrainer

tmp_dir = Path({tmp_dir!r})
tmp_dir.mkdir(parents=True, exist_ok=True)
count = 30
times = np.arange(count, dtype=np.int64) * 60_000
data = {{
    "open": 100.0 + np.sin(np.arange(count) / 5.0),
    "high": 100.5 + np.sin(np.arange(count) / 4.5),
    "low": 99.5 + np.sin(np.arange(count) / 4.0),
    "close": 100.2 + np.sin(np.arange(count) / 3.5),
    "volume": np.linspace(5.0, 6.5, count),
    "quote_asset_volume": np.linspace(500.0, 520.0, count),
    "number_of_trades": np.linspace(50, 55, count),
    "taker_buy_base": np.linspace(2.0, 2.5, count),
    "taker_buy_quote": np.linspace(200.0, 210.0, count),
    "close_time": times,
}}
df = pd.DataFrame(data)
data_cfg = replace(DATA, feature_window=4, forecast_horizon=1, validation_split=0.2)
training_cfg = replace(TRAINING, num_epochs=1, batch_size=2, seed=0, scheduler="none", enable_lr_range_test=False)
model_cfg = replace(MODEL, hermite_hidden_dim=2, dropout=0.0)
evaluation_cfg = replace(EVALUATION, save_markdown=True, alpha=0.1, threshold=0.55, cost_bps=0.0, val_block=6)
reporting_cfg = replace(REPORTING, output_dir=str(tmp_dir))
config = replace(APP_CONFIG, data=data_cfg, training=training_cfg, model=model_cfg, evaluation=evaluation_cfg, reporting=reporting_cfg)
trainer = HermiteTrainer(config=config)
dataset = HermiteDataset(df, data_config=data_cfg)
artifacts = trainer.run(dataset=dataset, use_cv=False, results_dir=tmp_dir)
payload = {{
    "columns": list(artifacts.results_table.columns),
    "markdown": str(artifacts.markdown_path) if artifacts.markdown_path else "",
    "fold_plots": [
        {{label: str(path) for label, path in f.reliability_paths.items()}}
        for f in artifacts.fold_results
    ],
    "calibration_methods": [f.calibration_method for f in artifacts.fold_results],
    "calibration_warnings": [f.calibration_warning for f in artifacts.fold_results],
    "probability_collapse": [f.probability_collapse for f in artifacts.fold_results],
}}
(tmp_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
"""


@pytest.mark.slow
def test_results_table_and_markdown(tmp_path) -> None:
    script_path = tmp_path / "run_pipeline.py"
    script_path.write_text(SCRIPT_TEMPLATE.format(tmp_dir=str(tmp_path)), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KMP_AFFINITY", "disabled")
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), existing_pythonpath]))

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        if "OMP: Error #179" in (result.stderr or ""):
            pytest.skip("OpenMP shared memory unavailable in sandbox environment.")
        raise AssertionError(f"Training script failed: {result.stderr}")

    payload_path = tmp_path / "results.json"
    data = json.loads(payload_path.read_text(encoding="utf-8"))

    required_columns = {
        "horizon",
        "MAE_return",
        "RMSE_return",
        "MAE_price",
        "sMAPE_price",
        "DirAcc",
    "Binom_p",
    "DM_p_SE",
    "DM_d_SE",
    "DM_p_AE",
    "DM_d_AE",
    "MZ_intercept",
    "MZ_slope",
    "MZ_F_p",
    "Runs_p",
    "LjungBox_p",
    "Brier",
    "Brier_raw",
    "Brier_uncertainty",
    "Brier_uncertainty_raw",
    "Brier_resolution",
    "Brier_resolution_raw",
    "Brier_reliability",
    "Brier_reliability_raw",
    "AUC",
    "AUC_raw",
    "ECE",
        "ECE_raw",
        "Sharpe_strategy",
        "MDD_strategy",
        "Turnover",
        "Sharpe_naive_long",
        "Sharpe_naive_flat",
    }
    table_columns = set(data["columns"])
    assert required_columns.issubset(table_columns)
    markdown_path = data["markdown"]
    if markdown_path:
        content = Path(markdown_path).read_text(encoding="utf-8")
        assert "Legend" in content
        assert "DM_p_SE" in content
    for plot_dict in data["fold_plots"]:
        for path_str in plot_dict.values():
            assert Path(path_str).exists()
    assert data["calibration_methods"], "Expected calibration methods to be reported."
    assert any(method in {"temperature", "isotonic", "temp_isotonic", "raw"} for method in data["calibration_methods"])
