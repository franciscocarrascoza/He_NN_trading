from __future__ import annotations

from dataclasses import replace

import plotly.graph_objects as go
import streamlit as st

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
    )


def _render_results(trainer: HermiteTrainer, artifacts) -> None:
    forecast_frame = artifacts.forecast_frame
    next_price = trainer.predict_next_price(artifacts)

    st.success(
        f"Training complete on device **{artifacts.device}**. "
        f"Next {trainer.training_config.forecast_horizon}-hour price forecast: **{next_price:.2f} USDT**"
    )

    st.subheader("Loss history")
    loss_columns = st.columns(2)
    loss_columns[0].line_chart({"Train": artifacts.training_losses})
    loss_columns[1].line_chart({"Validation": artifacts.validation_losses})

    st.subheader("Historical vs forecasted close")
    base_candles = artifacts.dataset.candles
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base_candles["close_time"],
            y=base_candles["close"],
            name="Historical Close",
            mode="lines",
            line=dict(color="#636EFA"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_frame["target_time"],
            y=forecast_frame["actual_price"],
            name="Observed Future Close",
            mode="lines",
            line=dict(color="#EF553B"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_frame["target_time"],
            y=forecast_frame["predicted_price"],
            name="Predicted Close",
            mode="lines",
            line=dict(color="#00CC96"),
        )
    )
    fig.update_layout(
        xaxis_title="Timestamp (UTC)",
        yaxis_title="Price (USDT)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def run_app() -> None:
    st.set_page_config(page_title="Hermite NN Forecaster", layout="wide")
    st.title("Hermite Neural Network Forecaster")
    st.write(
        "Configure the Binance data source and Hermite NN hyper-parameters, then train the model to "
        "visualise historical and predicted BTC prices."
    )

    controls_col, chart_col = st.columns([1, 2])

    with controls_col:
        st.header("Data source")
        symbol = st.text_input("Symbol", value=BINANCE.symbol)
        interval = st.text_input("Interval", value=BINANCE.interval)
        history_limit = st.number_input("History candles", value=BINANCE.history_limit, min_value=256, max_value=5000)
        forecast_horizon = st.slider("Forecast horizon (hours)", min_value=1, max_value=15, value=TRAINING.forecast_horizon)
        order_book_limit = st.number_input(
            "Order book depth", value=BINANCE.order_book_limit, min_value=10, max_value=500, step=5
        )
        liquidation_bins = st.number_input(
            "Liquidity bins", value=BINANCE.liquidation_bins, min_value=50, max_value=500, step=10
        )
        liquidation_price_range = st.number_input(
            "Liquidity price range", value=float(BINANCE.liquidation_price_range), min_value=0.1, max_value=5.0, step=0.1
        )
        long_short_period = st.text_input("Long/short ratio period", value=BINANCE.long_short_period)

        st.header("Training hyper-parameters")
        batch_size = st.number_input("Batch size", value=TRAINING.batch_size, min_value=8, max_value=1024, step=8)
        learning_rate = st.number_input(
            "Learning rate", value=float(TRAINING.learning_rate), min_value=1e-5, max_value=1e-1, format="%.5f"
        )
        num_epochs = st.number_input("Epochs", value=TRAINING.num_epochs, min_value=1, max_value=500)
        hermite_degree = st.number_input("Hermite degree", value=TRAINING.hermite_degree, min_value=1, max_value=12)
        hermite_maps_a = st.number_input("Hermite maps A", value=TRAINING.hermite_maps_a, min_value=1, max_value=8)
        hermite_maps_b = st.number_input("Hermite maps B", value=TRAINING.hermite_maps_b, min_value=1, max_value=8)
        hermite_hidden_dim = st.number_input(
            "Hermite hidden dim", value=TRAINING.hermite_hidden_dim, min_value=8, max_value=512, step=8
        )
        jacobian_mode = st.selectbox(
            "Jacobian mode", options=["summary", "full", "none"], index=["summary", "full", "none"].index(TRAINING.jacobian_mode)
        )
        feature_window = st.number_input("Feature window", value=TRAINING.feature_window, min_value=8, max_value=512, step=8)
        validation_split = st.slider(
            "Validation split", min_value=0.05, max_value=0.5, value=float(TRAINING.validation_split)
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
