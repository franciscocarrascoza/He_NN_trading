# Hermite Neural Network Trading Toolkit

This project provides a modular pipeline to forecast the next Bitcoin hourly price using a Hermite-based neural network. The model ingests three sources of information:

1. **OHLCV history** – the latest 3,000 hourly futures candles from Binance.
2. **Synthetic liquidity map** – a Coinglass-inspired liquidation density estimator derived from Binance derivatives metrics.
3. **Order book snapshot** – aggregated bid/ask depth and imbalance statistics.

## Project layout

```
.
├── main.py                   # Command line entry-point
├── inference.py              # Standalone checkpoint inference helper
├── src/
│   ├── __init__.py
│   ├── config/               # Centralised configuration dataclasses
│   │   └── settings.py
│   ├── data/                 # Binance REST client
│   │   ├── binance_fetcher.py
│   │   └── dataset.py        # Causal sliding-window dataset
│   ├── features/             # Feature engineering modules
│   │   ├── liquidity.py
│   │   └── orderbook.py
│   ├── models/               # Hermite neural network definition
│   │   └── hermite.py
│   ├── eval/                 # Offline evaluation helpers
│   │   └── evaluate.py
│   └── pipeline/             # Training utilities
│       └── training.py
```

Configuration values such as symbol, intervals, network hyper-parameters, and feature engineering controls are defined in `src/config/settings.py` and can be adjusted from a single location.

## Environment setup

### Conda workflow (recommended)

```bash
conda create -n hermite-nn python=3.10 -y
conda activate hermite-nn
# Install PyTorch with CUDA support (includes drivers for RTX 2060-class GPUs)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install numpy pandas requests scipy plotly streamlit -y
```

If you already maintain a CUDA-enabled PyTorch installation, you only need to ensure that the environment exposes the `torch` package compiled with GPU support.

### pip workflow

```bash
pip install numpy pandas requests scipy torch plotly streamlit --extra-index-url https://download.pytorch.org/whl/cu118
```

The `--extra-index-url` flag fetches the GPU-enabled wheels (compatible with RTX 2060) from the official PyTorch repository.

### Troubleshooting PyTorch import errors

If importing PyTorch fails with an error similar to:

```
ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

the environment is missing Intel's JIT profiling symbols that ship with the
`intel-openmp` runtime. This typically occurs when a conda environment upgrades
`torch` without pulling the matching OpenMP dependency, leaving an older
`libiomp5.so` on the search path. Fix the issue by reinstalling the Intel
runtime in the active environment:

```bash
conda install intel-openmp -y
# or, if you prefer conda-forge packages
conda install -c conda-forge libiomp -y
```

After reinstalling, restart the shell so the loader picks up the refreshed
library. Alternatively, reinstalling PyTorch with matching channels (e.g.
`conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y`) brings the
correct dependency set back in place.

## Usage

Run the training pipeline with:

```bash
python main.py
```

The script will:

1. Download the latest 3,000 hourly BTCUSDT futures candles from Binance.
2. Build the liquidity-map and order-book feature sets.
3. Assemble a sliding-window dataset, split it into training/validation subsets, and train the Hermite neural network.
4. Print the predicted price for the next hourly close.

The pipeline automatically stores a reproducible checkpoint under
`artifacts/hermite_forecaster.pt` (configurable via `--save`). The checkpoint
includes the model weights, optimiser state, feature list, train-only scaler
statistics, and the train/validation timestamp ranges.

### Interactive control panel

To manipulate data-source parameters, training hyper-parameters, and plot the historical vs. predicted prices, launch the Streamlit interface:

```bash
streamlit run src/interface/app.py
```

The left-hand panel exposes Binance symbol, interval, look-back size, and
Hermite NN hyper-parameters. Set the forecast horizon slider to any integer
between 1 and 15 hours to control how far ahead the network predicts. After
clicking **Start training**, the right-hand side renders the loss curves
together with historical candles, observed futures closes, the model's
forecasts, and the running average absolute error across the last ten
validation predictions (also displayed directly on the chart).

## Training and validation process

The pipeline trains through `src/pipeline/training.py` and follows these steps:

1. **Dataset preparation** – constructs causal OHLCV windows and appends
   liquidity/order-book summaries without performing any in-place
   normalisation.
2. **Chronological split** – keeps the earliest samples for training and
   reserves the last `validation_split` fraction for validation. The train/val
   timestamp ranges are printed to stdout for every run.
3. **GPU-aware execution** – automatically selects an NVIDIA RTX 2060 if
   present (`TrainingConfig.device_preference`). When unavailable, it falls
   back to the next CUDA device or CPU.
4. **Robust optimisation** – trains with the Huber loss (`delta=1e-3`) and
   clips gradients to `max_norm=1.0` for stability.
5. **Price-space validation** – after each epoch, reports the validation Huber
   loss, and after training computes MAE, RMSE, MAPE, median APE, and the mean
   absolute error over the last 10 validation predictions in price units.

Example console snippet:

```
Setting deterministic seed: 42
Train period: 2023-01-01T00:00:00+00:00 -> 2023-04-01T23:00:00+00:00 (1672531200000 -> 1680390000000)
Validation period: 2023-04-02T00:00:00+00:00 -> 2023-04-30T23:00:00+00:00 (1680393600000 -> 1682895600000)
Epoch 1/10 - Train Huber: 0.00001234 - Val Huber: 0.00002157
...
Validation price metrics -> MAE: 15.238401, RMSE: 22.719076, MAPE: 0.4123% , Median APE: 0.3011%, Avg abs err last 10: 11.904200
Checkpoint saved to artifacts/hermite_forecaster.pt
```

These metrics are mirrored in the Streamlit dashboard and stored with the
checkpoint for downstream evaluation.

### Offline evaluation and inference

Use the dedicated scripts to analyse or deploy trained checkpoints:

```bash
# Evaluate the validation slice recorded in the checkpoint
python -m src.eval.evaluate artifacts/hermite_forecaster.pt --csv-output reports/val_predictions.csv

# Run inference on your own candle history
python inference.py --ckpt artifacts/hermite_forecaster.pt --csv path/to/latest_candles.csv
```

The evaluation utility reloads the saved scalers, reconstructs validation price
predictions, and prints the MAE/RMSE/MAPE/median APE metrics while optionally
writing a detailed CSV report. The inference helper expects a CSV with at least
`open,high,low,close,volume,close_time` columns and outputs both the predicted
log-return and price at the configured horizon.

## Extending the project

* Update `FeatureConfig` or `TrainingConfig` in `src/config/settings.py` to change look-back windows, Hermite polynomial degree, or optimisation hyper-parameters.
* Implement additional feature transforms by adding new modules under `src/features/` and combining them within the training pipeline.
* Swap the Binance REST client with a websocket-driven data source by extending `src/data/binance_fetcher.py`.

## Disclaimer

This repository is intended for research and educational purposes. Cryptocurrency markets are highly volatile; no warranty is provided regarding the performance of the forecasts.
