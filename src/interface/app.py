from __future__ import annotations

"""Streamlit dashboard for configuring and running the Hermite NN forecaster."""

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
    STRATEGY,
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
    if fold.calibration_warning:  # FIX: enhance calibration warning messages for clarity
        if "ECE improvement below 20%" in fold.calibration_warning:  # FIX: provide diagnostic context
            raw_ece = fold.calibration_metrics_raw.ece if hasattr(fold, 'calibration_metrics_raw') else None  # FIX: extract raw ECE
            cal_ece = fold.calibration_metrics_calibrated.ece if hasattr(fold, 'calibration_metrics_calibrated') else None  # FIX: extract calibrated ECE
            if raw_ece is not None and cal_ece is not None:  # FIX: only show metrics when available
                improvement = ((raw_ece - cal_ece) / raw_ece * 100) if raw_ece > 0 else 0.0  # FIX: compute improvement percentage
                st.info(  # FIX: use info severity for diagnostic messages
                    f"Calibration ECE improvement small (raw: {raw_ece:.4f}, post-cal: {cal_ece:.4f}, "
                    f"improvement: {improvement:.1f}%). Probabilities may lack discrimination."
                )
            else:
                st.warning(fold.calibration_warning)  # FIX: fallback to original warning
        else:
            st.warning(fold.calibration_warning)  # FIX: show other calibration warnings as-is
    if fold.coverage_warning:  # FIX: enhance conformal coverage warning messages
        if fold.coverage_warning == "insufficient_calibration":  # FIX: detect insufficient calibration scenario
            calib_size = len(fold.calibration_residuals) if hasattr(fold, 'calibration_residuals') else 0  # FIX: extract calibration set size
            st.warning(  # FIX: provide actionable feedback
                f"Calibration set size {calib_size} < min required 256. "
                "Conformal intervals skipped. Consider increasing validation_split or history_limit."
            )
        elif fold.coverage_warning == "coverage_out_of_band":  # FIX: detect out-of-band coverage
            coverage = fold.coverage if hasattr(fold, 'coverage') else None  # FIX: extract empirical coverage
            target = 0.9  # FIX: assume 90% target from alpha=0.1
            if coverage is not None:  # FIX: only show coverage when available
                st.info(  # FIX: use info severity for coverage diagnostics
                    f"Conformal coverage {coverage*100:.1f}% deviates from target {target*100:.0f}% ±2%. "
                    "This is expected on small validation sets."
                )
            else:
                st.warning(f"Conformal warning: {fold.coverage_warning}")  # FIX: fallback message
        else:
            st.warning(f"Conformal warning: {fold.coverage_warning}")  # FIX: show other conformal warnings
    if getattr(fold, "predictions_path", None):
        st.info(f"Predictions CSV: {fold.predictions_path}")

    metrics_df = pd.DataFrame([fold.metrics]).T.reset_index()
    metrics_df.columns = ["Metric", "Value"]
    st.dataframe(metrics_df, width="stretch")

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
        st.dataframe(frame[display_columns].rename(columns=renamed).head(100), width="stretch")

    st.subheader("Strategy summary")
    strategy = getattr(fold, "strategy_metrics", None)
    if strategy and strategy.threshold is not None:
        active_msg = (
            f" | Active {strategy.active_fraction * 100:.1f}%" if strategy.active_fraction is not None else ""
        )
        st.markdown(
            f"Best threshold **{strategy.threshold:.2f}** | Sharpe **{strategy.sharpe:.3f}** | "
            f"Turnover **{strategy.turnover:.3f}** | Hit-rate **{strategy.hit_rate:.3f}**{active_msg}"
        )
    runs = getattr(fold, "strategy_runs", None)
    if runs:  # FIX: add threshold sweep distinctness diagnostic
        rows = []
        turnovers = []  # FIX: collect turnovers to verify distinctness
        for thr, metrics in sorted(runs.items()):
            rows.append(
                {
                    "Threshold": thr,
                    "Sharpe": metrics.sharpe,
                    "Turnover": metrics.turnover,
                    "Hit-rate": metrics.hit_rate,
                    "Active %": metrics.active_fraction * 100 if metrics.active_fraction is not None else None,
                }
            )
            turnovers.append(metrics.turnover)  # FIX: track turnover progression
        if len(set(turnovers)) > 1:  # FIX: verify thresholds produce distinct results
            st.info(  # FIX: confirm threshold sweep is working correctly
                f"Threshold sweep produced distinct trade patterns across {len(runs)} thresholds. "
                f"Turnover range: [{min(turnovers):.3f}, {max(turnovers):.3f}]"
            )
        elif len(runs) > 1:  # FIX: warn if all thresholds yield identical results
            st.warning(  # FIX: alert to possible threshold sweep bug
                "Threshold sweep produced identical turnover across all thresholds. "
                "Check probability distribution and confidence margins."
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")


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
        hermite_version_options = ["probabilist", "physicist"]
        try:
            hermite_version_idx = hermite_version_options.index(MODEL.hermite_version)
        except ValueError:
            hermite_version_idx = 0
        prob_source_options = ["cdf", "logit"]
        try:
            prob_source_idx = prob_source_options.index(MODEL.prob_source)
        except ValueError:
            prob_source_idx = 0
        with model_col1:
            hermite_version = st.selectbox(
                "Hermite series type",
                options=hermite_version_options,
                index=hermite_version_idx,
            )
            hermite_degree = st.number_input("Hermite degree", value=int(MODEL.hermite_degree), min_value=1, max_value=12)
            hermite_maps_a = st.number_input("Hermite maps A", value=int(MODEL.hermite_maps_a), min_value=1, max_value=8)
            hermite_hidden_dim = st.number_input(
                "Hermite hidden dim", value=int(MODEL.hermite_hidden_dim), min_value=8, max_value=512, step=8
            )
        with model_col2:
            hermite_maps_b = st.number_input("Hermite maps B", value=int(MODEL.hermite_maps_b), min_value=1, max_value=8)
            dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=float(MODEL.dropout), step=0.01)
            prob_source = st.selectbox(
                "Probability source",
                options=prob_source_options,
                index=prob_source_idx,
                help="Use the return CDF or the classification logits for downstream probabilities.",
            )
        use_lstm = st.checkbox("Use LSTM temporal encoder", value=MODEL.use_lstm)
        lstm_hidden = st.number_input(
            "LSTM hidden units", value=int(MODEL.lstm_hidden), min_value=8, max_value=512, step=8
        )

        st.header("Training hyper-parameters")
        train_col1, train_col2, train_col3 = st.columns(3)
        with train_col1:
            batch_size = st.number_input("Batch size", value=int(TRAINING.batch_size), min_value=8, max_value=1024, step=8)
            num_epochs = st.number_input("Epochs", value=int(TRAINING.num_epochs), min_value=1)  # FIX: remove hard 500 epoch cap
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
            early_stop_metric = st.selectbox(
                "Early stop metric",
                options=["AUC", "DirAcc"],
                index=["AUC", "DirAcc"].index(TRAINING.early_stop_metric.upper()),
            )
            patience = st.number_input(
                "Early stop patience", value=int(TRAINING.patience), min_value=0, max_value=500, step=1
            )
            min_delta = st.number_input(
                "Early stop min delta",
                value=float(TRAINING.min_delta),
                min_value=0.0,
                max_value=1.0,
                step=0.0001,
                format="%.5f",
            )
            seed = st.number_input("Random seed", value=int(TRAINING.seed), min_value=0, max_value=2**31 - 1)
            device_preference = st.text_input("Preferred CUDA device", value=TRAINING.device_preference)
        lambda_bce = TRAINING.lambda_bce

        weight_col1, weight_col2, weight_col3, weight_col4 = st.columns(4)
        with weight_col1:
            reg_weight = st.number_input(
                "Reg weight", value=float(TRAINING.reg_weight), min_value=0.0, max_value=10.0, step=0.1
            )
        with weight_col2:
            cls_weight = st.number_input(
                "Cls weight", value=float(TRAINING.cls_weight), min_value=0.0, max_value=10.0, step=0.1
            )
        with weight_col3:
            unc_weight = st.number_input(
                "Unc weight", value=float(TRAINING.unc_weight), min_value=0.0, max_value=10.0, step=0.1
            )
        with weight_col4:
            sign_hinge_weight = st.number_input(
                "Sign hinge weight", value=float(TRAINING.sign_hinge_weight), min_value=0.0, max_value=1.0, step=0.01
            )

        st.header("Strategy hyper-parameters")
        strat_col1, strat_col2 = st.columns(2)
        default_thresholds = ", ".join(f"{val:.2f}" for val in STRATEGY.thresholds)
        with strat_col1:
            thresholds_text = st.text_input(
                "Strategy thresholds (comma-separated)",
                value=default_thresholds,
                help="Enter probability cutoffs for entering long/short trades.",
            )
            strategy_confidence_margin = st.number_input(
                "Confidence margin",
                value=float(STRATEGY.confidence_margin),
                min_value=0.0,
                max_value=0.5,
                step=0.01,
            )
            strategy_kelly_clip = st.number_input(
                "Kelly clip",
                value=float(STRATEGY.kelly_clip),
                min_value=0.01,
                max_value=1.0,
                step=0.01,
            )
        with strat_col2:
            strategy_use_conformal_gate = st.checkbox(
                "Use conformal gate",
                value=STRATEGY.use_conformal_gate,
                help="Require conformal p-value to exceed a minimum before trading.",
            )
            strategy_conformal_p_min = st.number_input(
                "Conformal p min",
                value=float(STRATEGY.conformal_p_min),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )

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

        use_cv = st.checkbox("Enable rolling-origin CV", value=TRAINING.use_cv)
        training_cv_folds = st.number_input(
            "CV folds", value=int(TRAINING.cv_folds), min_value=1, max_value=20, step=1
        )
        results_dir_input = st.text_input("Results directory", value=str(APP_CONFIG.reporting.output_dir))
        train_button = st.button("Start training", type="primary")

    with chart_col:
        placeholder = st.empty()
        placeholder.info("Fill the parameters on the left and start training to see the forecast plot.")

    if train_button:
        if enable_lr_range_test and lr_range_min >= lr_range_max:
            placeholder.error("LR range min must be strictly less than LR range max.")
            st.stop()
        try:
            strategy_thresholds = tuple(
                sorted(
                    {
                        float(token.strip())
                        for token in thresholds_text.split(",")
                        if token.strip()
                    }
                )
            )
        except ValueError:
            placeholder.error("Invalid strategy thresholds. Use comma-separated floats.")
            st.stop()
        if not strategy_thresholds:
            placeholder.error("Please provide at least one strategy threshold.")
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
            hermite_version=hermite_version,
            use_lstm=use_lstm,
            lstm_hidden=int(lstm_hidden),
            prob_source=prob_source,
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
            seed=int(seed),
            enable_lr_range_test=enable_lr_range_test,
            lr_range_min=float(lr_range_min),
            lr_range_max=float(lr_range_max),
            lr_range_steps=int(lr_range_steps),
            reg_weight=float(reg_weight),
            cls_weight=float(cls_weight),
            unc_weight=float(unc_weight),
            sign_hinge_weight=float(sign_hinge_weight),
            use_cv=use_cv,
            cv_folds=int(training_cv_folds),
            early_stop_metric=early_stop_metric,
            patience=int(patience),
            min_delta=float(min_delta),
        )
        strategy_config = replace(
            STRATEGY,
            thresholds=strategy_thresholds,
            confidence_margin=float(strategy_confidence_margin),
            kelly_clip=float(strategy_kelly_clip),
            use_conformal_gate=strategy_use_conformal_gate,
            conformal_p_min=float(strategy_conformal_p_min),
        )
        evaluation_config = replace(
            APP_CONFIG.evaluation,
            cv_folds=int(training_cv_folds),
        )
        app_config: AppConfig = replace(
            APP_CONFIG,
            binance=binance_config,
            data=data_config,
            model=model_config,
            training=training_config,
            strategy=strategy_config,
            evaluation=evaluation_config,
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
