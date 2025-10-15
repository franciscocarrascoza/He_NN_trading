from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import BINANCE, FEATURES, TRAINING, BinanceAPIConfig, TrainingConfig
from src.data import BinanceDataFetcher
from src.pipeline import HermiteTrainer


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


def _build_training_config(
    *,
    forecast_horizon: int,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    hermite_degree: int,
    hermite_maps_a: int,
    hermite_maps_b: int,
    hermite_hidden_dim: int,
    jacobian_mode: str,
    feature_window: int,
    validation_split: float,
    random_seed: int,
    device_preference: str,
    direction_loss_weight: float,
    weight_decay: float,
    early_stopping_patience: int,
) -> TrainingConfig:
    return replace(
        TRAINING,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        hermite_degree=hermite_degree,
        hermite_maps_a=hermite_maps_a,
        hermite_maps_b=hermite_maps_b,
        hermite_hidden_dim=hermite_hidden_dim,
        jacobian_mode=jacobian_mode,
        feature_window=feature_window,
        validation_split=validation_split,
        random_seed=random_seed,
        device_preference=device_preference,
        direction_loss_weight=direction_loss_weight,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
    )


def _render_results(trainer: HermiteTrainer, artifacts) -> None:
    forecast_frame = artifacts.forecast_frame.copy()
    next_price = trainer.predict_next_price(artifacts)
    metrics = artifacts.price_metrics

    if "direction_hit" in forecast_frame:
        directionality = forecast_frame["direction_hit"].astype(int)
        if "direction_prob" in forecast_frame:
            forecast_frame["direction_probability"] = forecast_frame["direction_prob"]
        if "direction_pred" in forecast_frame:
            forecast_frame["direction_prediction"] = forecast_frame["direction_pred"].astype(int)
    else:
        price_delta_true = forecast_frame["true_price"] - forecast_frame["anchor_price"]
        price_delta_pred = forecast_frame["pred_price"] - forecast_frame["anchor_price"]
        directionality = (np.sign(price_delta_true) == np.sign(price_delta_pred)).astype(int)
        forecast_frame["direction_hit"] = directionality
        forecast_frame["direction_probability"] = np.nan
        forecast_frame["direction_prediction"] = np.sign(price_delta_pred).clip(min=0).astype(int)
    forecast_frame["directionality"] = directionality
    total_predictions = len(directionality)
    correct_predictions = int(directionality.sum())
    directional_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions else 0.0
    if hasattr(metrics, "direction_hit_rate") and not np.isnan(metrics.direction_hit_rate):
        directional_accuracy = metrics.direction_hit_rate * 100

    with st.container():
        st.markdown(
            """
            <style>
            .results-section * {
                font-size: 24px !important;
            }
            </style>
            <div class="results-section">
            """,
            unsafe_allow_html=True,
        )

        st.success(
            f"Training complete on device **{artifacts.device}**. "
            f"Next {trainer.training_config.forecast_horizon}-hour price forecast: **{next_price:.2f} USDT**"
        )

        metric_columns = st.columns(4)
        metric_columns[0].metric("Val MAE", f"{metrics.mae:.4f}")
        metric_columns[1].metric("Val RMSE", f"{metrics.rmse:.4f}")
        metric_columns[2].metric("Val MAPE", f"{metrics.mape:.3f}%")
        metric_columns[3].metric("Median APE", f"{metrics.median_ape:.3f}%")

        st.write(
            "Directional Accuracy (DA): "
            f"{directional_accuracy:.2f}% — percentage of forecasts with correct up/down direction."
        )
        st.write(
            "Sign Accuracy (SA): "
            f"{directional_accuracy:.2f}% — fraction of matching sign predictions (target vs forecast)."
        )

        st.subheader("Historical vs forecasted close")
        target_times = forecast_frame["target_time"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=target_times,
                y=forecast_frame["true_price"],
                name="Observed Future Close",
                mode="lines",
                line=dict(color="green", dash="solid"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=target_times,
                y=forecast_frame["pred_price"],
                name="Predicted Close",
                mode="lines",
                line=dict(color="orange", dash="dash"),
            )
        )
        fig.add_annotation(
            text=(
                f"Avg abs err last 10: {metrics.avg_abs_err_last_10:.4f} USDT\n"
                f"Validation window: {forecast_frame['target_time'].iloc[0]} → "
                f"{forecast_frame['target_time'].iloc[-1]}"
            ),
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            showarrow=False,
            bgcolor="rgba(0, 0, 0, 0.35)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            align="left",
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

        st.subheader("Validation residuals")
        residual_fig = go.Figure()
        residual_fig.add_trace(
            go.Histogram(x=forecast_frame["pred_price"] - forecast_frame["true_price"], nbinsx=50)
        )
        residual_fig.update_layout(
            xaxis_title="Prediction error (USDT)",
            yaxis_title="Frequency",
            bargap=0.05,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(residual_fig, use_container_width=True)

        st.dataframe(
            forecast_frame[[
                "target_time",
                "anchor_price",
                "true_price",
                "pred_price",
                "abs_error",
                "ape_pct",
                "direction_prediction",
                "direction_probability",
                "directionality",
            ]].rename(columns={
                "target_time": "Target Time",
                "anchor_price": "Anchor Price",
                "true_price": "True Price",
                "pred_price": "Predicted Price",
                "abs_error": "Absolute Error",
                "ape_pct": "APE %",
                "direction_prediction": "Pred Direction",
                "direction_probability": "P(Up)",
                "directionality": "Directionality",
            })
        )
        st.markdown("</div>", unsafe_allow_html=True)


def run_app() -> None:
    st.set_page_config(page_title="Hermite NN Forecaster", layout="wide")
    st.title("Hermite Neural Network Forecaster")
    st.write(
        "Configure the Binance data source and Hermite NN hyper-parameters, then train the model to "
        "visualise historical and predicted BTC prices."
    )

    controls_col, chart_col = st.columns([0.7, 2])

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
                "Forecast horizon (hours)", min_value=1, max_value=15, value=TRAINING.forecast_horizon
            )
            liquidation_bins = st.number_input(
                "Liquidity bins", value=BINANCE.liquidation_bins, min_value=50, max_value=500, step=10
            )
            long_short_period = st.text_input("Long/short ratio period", value=BINANCE.long_short_period)

        st.header("Training hyper-parameters")
        train_col1, train_col2, train_col3 = st.columns(3)
        with train_col1:
            batch_size = st.number_input("Batch size", value=TRAINING.batch_size, min_value=8, max_value=1024, step=8)
            num_epochs = st.number_input("Epochs", value=TRAINING.num_epochs, min_value=1, max_value=500)
            hermite_degree = st.number_input("Hermite degree", value=TRAINING.hermite_degree, min_value=1, max_value=12)
            hermite_maps_a = st.number_input("Hermite maps A", value=TRAINING.hermite_maps_a, min_value=1, max_value=8)
        with train_col2:
            learning_rate = st.number_input(
                "Learning rate", value=float(TRAINING.learning_rate), min_value=1e-5, max_value=1e-1, format="%.5f"
            )
            hermite_maps_b = st.number_input("Hermite maps B", value=2, min_value=1, max_value=8)
            hermite_hidden_dim = st.number_input(
                "Hermite hidden dim", value=TRAINING.hermite_hidden_dim, min_value=8, max_value=512, step=8
            )
            feature_window = st.number_input("Feature window", value=TRAINING.feature_window, min_value=8, max_value=512, step=8)
            jacobian_mode = st.selectbox(
                "Jacobian mode", options=["summary", "full", "none"], index=["summary", "full", "none"].index("full")
            )
        with train_col3:
            validation_split = st.slider(
                "Validation split", min_value=0.05, max_value=0.5, value=float(TRAINING.validation_split)
            )
            direction_loss_weight = st.number_input(
                "Direction loss weight", value=float(TRAINING.direction_loss_weight), min_value=0.0, max_value=1.0, step=0.01
            )
            weight_decay = st.number_input(
                "Weight decay",
                value=float(TRAINING.weight_decay),
                min_value=0.0,
                max_value=1e-2,
                step=1e-6,
                format="%.6f",
            )
            early_stopping_patience = st.number_input(
                "Early stopping patience", value=TRAINING.early_stopping_patience, min_value=0, max_value=100, step=1
            )
            random_seed = st.number_input("Random seed", value=TRAINING.random_seed, min_value=0, max_value=2**31 - 1)
            device_preference = st.text_input("Preferred CUDA device", value=TRAINING.device_preference)

        train_button = st.button("Start training", type="primary")

    with chart_col:
        placeholder = st.empty()
        placeholder.info("Fill the parameters on the left and start training to see the forecast plot.")

    if train_button:
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
        training_config = _build_training_config(
            forecast_horizon=int(forecast_horizon),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            num_epochs=int(num_epochs),
            hermite_degree=int(hermite_degree),
            hermite_maps_a=int(hermite_maps_a),
            hermite_maps_b=int(hermite_maps_b),
            hermite_hidden_dim=int(hermite_hidden_dim),
            jacobian_mode=str(jacobian_mode),
            feature_window=int(feature_window),
            validation_split=float(validation_split),
            random_seed=int(random_seed),
            device_preference=device_preference,
            direction_loss_weight=float(direction_loss_weight),
            weight_decay=float(weight_decay),
            early_stopping_patience=int(early_stopping_patience),
        )

        fetcher = BinanceDataFetcher(binance_config)
        trainer = HermiteTrainer(fetcher, feature_config=FEATURES, training_config=training_config)

        try:
            with st.spinner("Training Hermite NN..."):
                artifacts = trainer.train()
        except Exception as error:  # pylint: disable=broad-except
            placeholder.error(f"Training failed: {error}")
            st.stop()

        placeholder.empty()
        _render_results(trainer, artifacts)


if __name__ == "__main__":
    run_app()
