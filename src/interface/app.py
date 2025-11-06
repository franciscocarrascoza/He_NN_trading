from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    APP_CONFIG,
    BINANCE,
    DATA,
    MODEL,
    TRAINING,
    AppConfig,
    BinanceAPIConfig,
)
from src.data import BinanceDataFetcher
from src.pipeline import HermiteTrainer

assert hasattr(DATA, "forecast_horizon"), "DataConfig must define forecast_horizon."


def _build_binance_config(
    *,
    symbol: str,
    interval: str,
    history_limit: int,
    order_book_limit: int,
    liquidation_bins: int,
    liquidation_price_range: float,
    long_short_period: str,
) -> BinanceAPIConfig:
    return replace(
        BINANCE,
        symbol=symbol,
        interval=interval,
        history_limit=history_limit,
        order_book_limit=order_book_limit,
        liquidation_bins=liquidation_bins,
        liquidation_price_range=liquidation_price_range,
        long_short_period=long_short_period,
    )


def _format_datetime(series: pd.Series) -> pd.Series:
    try:
        converted = pd.to_datetime(series, unit="ms", utc=True)
        return converted.dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OverflowError):
        return series.astype(str)


def _render_fold_section(fold) -> None:
    st.markdown(f"### Fold {fold.fold_id}")
    if fold.calibration_warning:
        st.warning(fold.calibration_warning)
    if fold.coverage_warning:
        st.warning(f"Conformal warning: {fold.coverage_warning}")

    metrics_df = pd.DataFrame([fold.metrics]).T.reset_index()
    metrics_df.columns = ["Metric", "Value"]
    st.dataframe(metrics_df, use_container_width=True)

    reliability_cols = st.columns(2)
    raw_path = fold.reliability_paths.get("raw")
    cal_path = fold.reliability_paths.get("calibrated")
    if raw_path and raw_path.exists():
        reliability_cols[0].image(str(raw_path), caption="Reliability (raw calibrated probabilities)")
    if cal_path and cal_path.exists():
        reliability_cols[1].image(str(cal_path), caption="Reliability (post calibration)")
    if fold.lr_range_path and fold.lr_range_path.exists():
        st.image(str(fold.lr_range_path), caption="LR range sweep (fold 0)")

    frame = fold.forecast_frame.copy()
    if "anchor_price" in frame and "pred_price" in frame and "true_price" in frame:
        frame["abs_error"] = np.abs(frame["pred_price"] - frame["true_price"])
        denominator = np.abs(frame["true_price"]).replace(0, np.nan)
        frame["ape_pct"] = 100.0 * frame["abs_error"] / denominator
    if "target_time" in frame:
        frame["target_time_str"] = _format_datetime(frame["target_time"])

    st.subheader("Price forecasts")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.get("target_time_str", frame.get("target_time")),
            y=frame["true_price"],
            name="Observed Future Close",
            mode="lines",
            line=dict(color="green", dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.get("target_time_str", frame.get("target_time")),
            y=frame["pred_price"],
            name="Predicted Close",
            mode="lines",
            line=dict(color="orange", dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title="Timestamp (UTC)",
        yaxis_title="Price (USDT)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    if "abs_error" in frame:
        st.subheader("Validation residuals")
        residual_fig = go.Figure()
        residual_fig.add_trace(
            go.Histogram(x=frame["pred_price"] - frame["true_price"], nbinsx=50)
        )
        residual_fig.update_layout(
            xaxis_title="Prediction error (USDT)",
            yaxis_title="Frequency",
            bargap=0.05,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(residual_fig, use_container_width=True)

    display_columns = [
        col
        for col in [
            "target_time_str",
            "anchor_price",
            "true_price",
            "pred_price",
            "abs_error",
            "ape_pct",
            "prob_up",
            "prob_up_raw",
            "conformal_lower_price",
            "conformal_upper_price",
            "conformal_p",
        ]
        if col in frame.columns
    ]
    if display_columns:
        renamed = {
            "target_time_str": "Target Time (UTC)",
            "anchor_price": "Anchor Price",
            "true_price": "True Price",
            "pred_price": "Predicted Price",
            "abs_error": "|Error|",
            "ape_pct": "APE %",
            "prob_up": "Prob Up (cal)",
            "prob_up_raw": "Prob Up (raw)",
            "conformal_lower_price": "Conf Lower (price)",
            "conformal_upper_price": "Conf Upper (price)",
            "conformal_p": "Conformal p-value",
        }
        st.subheader("Forecast table (head)")
        st.dataframe(frame[display_columns].rename(columns=renamed).head(100), use_container_width=True)


def run_app() -> None:
    st.set_page_config(page_title="Hermite NN Forecaster", layout="wide")
    st.title("Hermite Neural Network Forecaster")
    st.write(
        "Configure the Binance data source and Hermite NN hyper-parameters, then train the model to "
        "visualise historical and predicted BTC prices."
    )

    controls_col, chart_col = st.columns([0.8, 1.2])

    with controls_col:
        st.header("Data source")
        data_col1, data_col2 = st.columns(2)
        with data_col1:
            symbol = st.text_input("Symbol", value=BINANCE.symbol, label_visibility="visible")
            history_limit = st.number_input(
                "History candles", value=BINANCE.history_limit, min_value=256, max_value=5000
            )
            order_book_limit = st.number_input(
                "Order book depth", value=BINANCE.order_book_limit, min_value=10, max_value=500, step=5
            )
            liquidation_price_range = st.number_input(
                "Liquidity price range", value=float(BINANCE.liquidation_price_range), min_value=0.1, max_value=5.0, step=0.1
            )
        with data_col2:
            interval = st.text_input("Interval", value=BINANCE.interval, label_visibility="visible")
            forecast_horizon = st.slider(
                "Forecast horizon (hours)",
                min_value=1,
                max_value=24,
                value=int(DATA.forecast_horizon),
                help="Forecast horizon (hours) = steps ahead for sign/return targets.",
            )
            liquidation_bins = st.number_input(
                "Liquidity bins", value=BINANCE.liquidation_bins, min_value=50, max_value=500, step=10
            )
            long_short_period = st.text_input("Long/short ratio period", value=BINANCE.long_short_period)
        feature_window = st.number_input("Feature window", value=int(DATA.feature_window), min_value=8, max_value=512, step=4)
        validation_split = st.slider(
            "Validation split", min_value=0.05, max_value=0.5, value=float(DATA.validation_split), step=0.01
        )
        use_extras = st.checkbox("Include liquidity/order-book extras (if available)", value=DATA.use_extras)

        st.header("Model hyper-parameters")
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            hermite_degree = st.number_input("Hermite degree", value=int(MODEL.hermite_degree), min_value=1, max_value=12)
            hermite_maps_a = st.number_input("Hermite maps A", value=int(MODEL.hermite_maps_a), min_value=1, max_value=8)
            hermite_hidden_dim = st.number_input(
                "Hermite hidden dim", value=int(MODEL.hermite_hidden_dim), min_value=8, max_value=512, step=8
            )
        with model_col2:
            hermite_maps_b = st.number_input("Hermite maps B", value=int(MODEL.hermite_maps_b), min_value=1, max_value=8)
            dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=float(MODEL.dropout), step=0.01)

        st.header("Training hyper-parameters")
        train_col1, train_col2, train_col3 = st.columns(3)
        with train_col1:
            batch_size = st.number_input("Batch size", value=int(TRAINING.batch_size), min_value=8, max_value=1024, step=8)
            num_epochs = st.number_input("Epochs", value=int(TRAINING.num_epochs), min_value=1, max_value=500)
            learning_rate = st.number_input(
                "Learning rate", value=float(TRAINING.learning_rate), min_value=1e-6, max_value=1.0, format="%.6f"
            )
            weight_decay = st.number_input(
                "Weight decay",
                value=float(TRAINING.weight_decay),
                min_value=0.0,
                max_value=1e-2,
                step=1e-6,
                format="%.6f",
            )
        with train_col2:
            gradient_clip = st.number_input(
                "Gradient clip", value=float(TRAINING.gradient_clip), min_value=0.1, max_value=10.0, step=0.1
            )
            lambda_bce = st.number_input(
                "BCE weight λ", value=float(TRAINING.lambda_bce), min_value=0.0, max_value=1.0, step=0.01
            )
            optimizer = st.selectbox(
                "Optimizer",
                options=["adamw", "adam", "sgd"],
                index=["adamw", "adam", "sgd"].index(TRAINING.optimizer.lower()),
            )
            scheduler = st.selectbox(
                "Scheduler",
                options=["none", "onecycle", "cosine"],
                index=["none", "onecycle", "cosine"].index(TRAINING.scheduler.lower()),
            )
            scheduler_warmup_pct = st.slider(
                "Scheduler warmup pct",
                min_value=0.0,
                max_value=0.5,
                value=float(TRAINING.scheduler_warmup_pct),
                step=0.01,
            )
        with train_col3:
            classification_loss = st.selectbox(
                "Classification loss", options=["bce", "focal"], index=["bce", "focal"].index(TRAINING.classification_loss)
            )
            focal_gamma = st.number_input(
                "Focal γ",
                value=float(TRAINING.focal_gamma),
                min_value=0.5,
                max_value=5.0,
                step=0.1,
            )
            auto_pos_weight = st.checkbox("Auto class weighting", value=TRAINING.auto_pos_weight)
            min_pos_weight_samples = st.number_input(
                "Min samples for class weight", value=int(TRAINING.min_pos_weight_samples), min_value=1, step=1
            )
            early_stopping_patience = st.number_input(
                "Early stopping patience", value=int(TRAINING.early_stopping_patience), min_value=0, max_value=200, step=1
            )
            seed = st.number_input("Random seed", value=int(TRAINING.seed), min_value=0, max_value=2**31 - 1)
            device_preference = st.text_input("Preferred CUDA device", value=TRAINING.device_preference)

        with st.expander("Learning-rate range test", expanded=TRAINING.enable_lr_range_test):
            enable_lr_range_test = st.checkbox(
                "Enable LR range sweep", value=TRAINING.enable_lr_range_test, help="Produces lr_range_fold_0.png."
            )
            lr_range_min = st.number_input(
                "LR range min", value=float(TRAINING.lr_range_min), min_value=1e-7, max_value=1.0, format="%.6f"
            )
            lr_range_max = st.number_input(
                "LR range max", value=float(TRAINING.lr_range_max), min_value=1e-6, max_value=1.0, format="%.6f"
            )
            lr_range_steps = st.number_input(
                "LR range steps", value=int(TRAINING.lr_range_steps), min_value=5, max_value=200, step=5
            )

        use_cv = st.checkbox("Enable rolling-origin CV", value=False)
        results_dir_input = st.text_input("Results directory", value=str(APP_CONFIG.reporting.output_dir))
        train_button = st.button("Start training", type="primary")

    with chart_col:
        placeholder = st.empty()
        placeholder.info("Fill the parameters on the left and start training to see the forecast plot.")

    if train_button:
        if enable_lr_range_test and lr_range_min >= lr_range_max:
            placeholder.error("LR range min must be strictly less than LR range max.")
            st.stop()
        placeholder.warning("Downloading data and training the model. This may take a few minutes...")
        binance_config = _build_binance_config(
            symbol=symbol,
            interval=interval,
            history_limit=int(history_limit),
            order_book_limit=int(order_book_limit),
            liquidation_bins=int(liquidation_bins),
            liquidation_price_range=float(liquidation_price_range),
            long_short_period=long_short_period,
        )
        data_config = replace(
            DATA,
            forecast_horizon=int(forecast_horizon),
            feature_window=int(feature_window),
            validation_split=float(validation_split),
            use_extras=use_extras,
        )
        model_config = replace(
            MODEL,
            hermite_degree=int(hermite_degree),
            hermite_maps_a=int(hermite_maps_a),
            hermite_maps_b=int(hermite_maps_b),
            hermite_hidden_dim=int(hermite_hidden_dim),
            dropout=float(dropout),
        )
        training_config = replace(
            TRAINING,
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            num_epochs=int(num_epochs),
            device_preference=device_preference,
            weight_decay=float(weight_decay),
            gradient_clip=float(gradient_clip),
            lambda_bce=float(lambda_bce),
            scheduler=scheduler,
            scheduler_warmup_pct=float(scheduler_warmup_pct),
            optimizer=optimizer,
            classification_loss=classification_loss,
            focal_gamma=float(focal_gamma),
            auto_pos_weight=auto_pos_weight,
            min_pos_weight_samples=int(min_pos_weight_samples),
            early_stopping_patience=int(early_stopping_patience),
            seed=int(seed),
            enable_lr_range_test=enable_lr_range_test,
            lr_range_min=float(lr_range_min),
            lr_range_max=float(lr_range_max),
            lr_range_steps=int(lr_range_steps),
        )
        app_config: AppConfig = replace(
            APP_CONFIG,
            binance=binance_config,
            data=data_config,
            model=model_config,
            training=training_config,
        )
        fetcher = BinanceDataFetcher(binance_config)
        trainer = HermiteTrainer(config=app_config, fetcher=fetcher)

        try:
            results_dir = Path(results_dir_input) if results_dir_input else None
            with st.spinner("Training Hermite NN..."):
                artifacts = trainer.run(use_cv=use_cv, results_dir=results_dir)
        except Exception as error:  # pylint: disable=broad-except
            placeholder.error(f"Training failed: {error}")
            st.stop()

        placeholder.empty()
        st.success("Training complete.")
        st.subheader("Aggregate metrics")
        st.dataframe(artifacts.results_table)

        if artifacts.csv_path:
            st.info(f"Results CSV saved to {artifacts.csv_path}")
        if artifacts.markdown_path:
            st.info(f"Results Markdown saved to {artifacts.markdown_path}")

        fold_metrics_df = pd.DataFrame([fold.metrics for fold in artifacts.fold_results])
        st.subheader("Fold metrics")
        st.dataframe(fold_metrics_df)

        for fold in artifacts.fold_results:
            _render_fold_section(fold)
            st.markdown("---")


if __name__ == "__main__":
    run_app()
