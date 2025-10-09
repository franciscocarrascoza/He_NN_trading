# Hermite Neural Network Trading Toolkit

This project provides a modular pipeline to forecast the next Bitcoin hourly price using a Hermite-based neural network. The model ingests three sources of information:

1. **OHLCV history** – the latest 3,000 hourly futures candles from Binance.
2. **Synthetic liquidity map** – a Coinglass-inspired liquidation density estimator derived from Binance derivatives metrics.
3. **Order book snapshot** – aggregated bid/ask depth and imbalance statistics.

## Project layout

```
.
├── main.py                   # Command line entry-point
├── src/
│   ├── __init__.py
│   ├── config/               # Centralised configuration dataclasses
│   │   └── settings.py
│   ├── data/                 # Binance REST client
│   │   └── binance_fetcher.py
│   ├── features/             # Feature engineering modules
│   │   ├── liquidity.py
│   │   └── orderbook.py
│   ├── models/               # Hermite neural network definition
│   │   └── hermite.py
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
conda install numpy pandas requests scipy -y
```

If you already maintain a CUDA-enabled PyTorch installation, you only need to ensure that the environment exposes the `torch` package compiled with GPU support.

### pip workflow

```bash
pip install numpy pandas requests scipy torch --extra-index-url https://download.pytorch.org/whl/cu118
```

The `--extra-index-url` flag fetches the GPU-enabled wheels (compatible with RTX 2060) from the official PyTorch repository.

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

Optionally save the trained artifacts for later inference:

```bash
python main.py --save artifacts.pt
```

The saved payload contains the model parameters alongside feature normalisation statistics for reproducible deployment.

## Training and validation process

The pipeline trains through `src/pipeline/training.py` and follows these steps:

1. **Dataset preparation** – combines normalised OHLCV windows with liquidity/order-book features and produces log-return targets.
2. **Deterministic split** – shuffles the dataset with a fixed seed and reserves 20% for validation (configurable via `TrainingConfig.validation_split`).
3. **GPU-aware execution** – automatically selects an NVIDIA RTX 2060 if present (`TrainingConfig.device_preference`). When unavailable, it falls back to the next CUDA device or CPU.
4. **Loss tracking** – optimises with Adam while recording mean-squared-error losses for both training and validation sets each epoch. These metrics are printed to stdout and saved with the artifacts for later inspection.

## Extending the project

* Update `FeatureConfig` or `TrainingConfig` in `src/config/settings.py` to change look-back windows, Hermite polynomial degree, or optimisation hyper-parameters.
* Implement additional feature transforms by adding new modules under `src/features/` and combining them within the training pipeline.
* Swap the Binance REST client with a websocket-driven data source by extending `src/data/binance_fetcher.py`.

## Disclaimer

This repository is intended for research and educational purposes. Cryptocurrency markets are highly volatile; no warranty is provided regarding the performance of the forecasts.
