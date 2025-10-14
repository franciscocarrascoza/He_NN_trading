from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.config import BINANCE, FEATURES, TRAINING, FeatureConfig, TrainingConfig
from src.data import BinanceDataFetcher, HermiteDataset
from src.features import compute_liquidity_features, compute_orderbook_features
from src.models import HermiteForecaster


@dataclass(frozen=True)
class PriceMetrics:
    """Price-space validation metrics computed on the out-of-sample slice."""

    mae: float
    rmse: float
    mape: float
    median_ape: float
    avg_abs_err_last_10: float


@dataclass
class TrainingArtifacts:
    model: HermiteForecaster
    dataset: HermiteDataset
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    training_losses: List[float]
    validation_losses: List[float]
    device: torch.device
    forecast_frame: pd.DataFrame
    price_metrics: PriceMetrics
    checkpoint_path: Path
    train_indices: torch.Tensor
    val_indices: torch.Tensor


class HermiteTrainer:
    """Train Hermite forecasters with chronological validation splits."""

    def __init__(
        self,
        fetcher: BinanceDataFetcher | None = None,
        *,
        feature_config: FeatureConfig = FEATURES,
        training_config: TrainingConfig = TRAINING,
        device: torch.device | None = None,
    ) -> None:
        self.fetcher = fetcher or BinanceDataFetcher(BINANCE)
        self.feature_config = feature_config
        self.training_config = training_config
        self.device = device or self._select_device()

    # ------------------------------------------------------------------
    # Utilities
    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            preference = self.training_config.device_preference.lower()
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                if preference and preference in name.lower():
                    print(f"Using preferred CUDA device '{name}' (index {index}).")
                    return torch.device(f"cuda:{index}")
            default_name = torch.cuda.get_device_name(0)
            print(
                "Preferred CUDA device not found. "
                f"Falling back to '{default_name}' (index 0)."
            )
            return torch.device("cuda:0")
        print("CUDA device unavailable. Falling back to CPU execution.")
        return torch.device("cpu")

    def _set_deterministic_seeds(self) -> None:
        seed = getattr(self.training_config, "seed", self.training_config.random_seed)
        print(f"Setting deterministic seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # torch < 1.8
            pass

    # ------------------------------------------------------------------
    # Dataset preparation
    def prepare_dataset(self, *, normalise: bool = False) -> HermiteDataset:
        candles = self.fetcher.get_historical_candles(limit=BINANCE.history_limit)
        liquidity_features = compute_liquidity_features(
            self.fetcher, feature_config=self.feature_config
        )
        orderbook_features = compute_orderbook_features(
            self.fetcher, feature_config=self.feature_config
        )
        dataset = HermiteDataset(
            candles,
            liquidity_features,
            orderbook_features,
            feature_window=self.training_config.feature_window,
            forecast_horizon=self.training_config.forecast_horizon,
            normalise=normalise,
        )
        return dataset

    # ------------------------------------------------------------------
    # Training pipeline
    def train(self, dataset: Optional[HermiteDataset] = None) -> TrainingArtifacts:
        self._set_deterministic_seeds()

        dataset = dataset or self.prepare_dataset(normalise=False)
        total_length = len(dataset)
        if total_length < 2:
            raise ValueError("Dataset must contain at least two samples for splitting")

        val_length = max(1, int(total_length * self.training_config.validation_split))
        train_length = total_length - val_length
        if train_length <= 0:
            raise ValueError(
                "Validation split too large for dataset size. Reduce validation_split."
            )

        train_idx = torch.arange(0, train_length)
        val_idx = torch.arange(train_length, total_length)

        # Log the chronological ranges
        train_start = dataset.timestamps[train_idx[0]].item()
        train_end = dataset.timestamps[train_idx[-1]].item()
        val_start = dataset.timestamps[val_idx[0]].item()
        val_end = dataset.timestamps[val_idx[-1]].item()

        def _fmt(ts: int) -> str:
            # Binance close timestamps are in milliseconds
            return pd.to_datetime(ts, unit="ms", utc=True).isoformat()

        print(
            "Train period: "
            f"{_fmt(train_start)} -> {_fmt(train_end)} ({train_start} -> {train_end})"
        )
        print(
            "Validation period: "
            f"{_fmt(val_start)} -> {_fmt(val_end)} ({val_start} -> {val_end})"
        )

        # Compute scalers on the training slice only
        train_features = dataset.features[train_idx]
        feature_mean = train_features.mean(dim=0)
        feature_std = train_features.std(dim=0, unbiased=False)
        feature_std = torch.clamp(feature_std, min=1e-6)
        dataset.features = (dataset.features - feature_mean) / feature_std
        dataset.feature_mean = feature_mean
        dataset.feature_std = feature_std

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

        model = HermiteForecaster(input_dim=dataset.features.shape[1], config=self.training_config)
        model.to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.training_config.learning_rate)
        loss_fn = torch.nn.HuberLoss(delta=1e-3)

        model.train()
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(self.training_config.num_epochs):
            running_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimiser.zero_grad()
                preds = model(batch_features)
                loss = loss_fn(preds, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                running_loss += loss.item() * batch_features.size(0)
            running_loss /= train_length
            train_losses.append(running_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_features, val_targets in val_loader:
                    val_features = val_features.to(self.device)
                    val_targets = val_targets.to(self.device)
                    preds = model(val_features)
                    loss = loss_fn(preds, val_targets)
                    val_loss += loss.item() * val_features.size(0)
            val_loss /= val_length
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch + 1}/{self.training_config.num_epochs} - "
                f"Train Huber: {running_loss:.8f} - Val Huber: {val_loss:.8f}"
            )
            model.train()

        model.eval()
        with torch.no_grad():
            val_features = dataset.features[val_idx].to(self.device)
            predicted_log_returns = model(val_features).squeeze(1).cpu().numpy()

        anchor_prices = dataset.anchor_prices[val_idx].cpu().numpy()
        actual_log_returns = dataset.targets[val_idx].squeeze(1).cpu().numpy()
        predicted_prices = anchor_prices * np.exp(predicted_log_returns)
        actual_prices = anchor_prices * np.exp(actual_log_returns)

        residuals = predicted_prices - actual_prices
        abs_err = np.abs(residuals)
        mae = float(abs_err.mean())
        rmse = float(np.sqrt(np.mean(residuals**2)))
        ape = abs_err / np.maximum(np.abs(actual_prices), 1e-6)
        mape = float(ape.mean() * 100.0)
        median_ape = float(np.median(ape) * 100.0)
        last_10 = abs_err[-10:]
        avg_last_10 = float(last_10.mean()) if last_10.size else float("nan")

        print(
            "Validation price metrics -> "
            f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.4f}% , "
            f"Median APE: {median_ape:.4f}%, Avg abs err last 10: {avg_last_10:.6f}"
        )

        forecast_frame = pd.DataFrame(
            {
                "anchor_time": pd.to_datetime(
                    dataset.anchor_times[val_idx].numpy(), unit="ms", utc=True
                ),
                "target_time": pd.to_datetime(
                    dataset.target_times[val_idx].numpy(), unit="ms", utc=True
                ),
                "anchor_price": anchor_prices,
                "true_log_return": actual_log_returns,
                "pred_log_return": predicted_log_returns,
                "true_price": actual_prices,
                "pred_price": predicted_prices,
                "abs_error": abs_err,
                "ape_pct": ape * 100.0,
            }
        )

        price_metrics = PriceMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            median_ape=median_ape,
            avg_abs_err_last_10=avg_last_10,
        )

        checkpoint_path = Path(self.training_config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimiser.state_dict(),
            "scheduler_state": None,
            "feature_mean": feature_mean.cpu(),
            "feature_std": feature_std.cpu(),
            "feature_names": dataset.feature_names,
            "base_feature_names": list(HermiteDataset.base_feature_names),
            "extra_features": dataset.extra_features.cpu(),
            "window": self.training_config.feature_window,
            "horizon": self.training_config.forecast_horizon,
            "train_time_range": (
                int(dataset.timestamps[0].item()),
                int(dataset.timestamps[train_length - 1].item()),
            ),
            "val_time_range": (
                int(dataset.timestamps[train_length].item()),
                int(dataset.timestamps[-1].item()),
            ),
            "seed": getattr(self.training_config, "seed", self.training_config.random_seed),
            "training_config": asdict(self.training_config),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        return TrainingArtifacts(
            model=model,
            dataset=dataset,
            feature_mean=feature_mean,
            feature_std=feature_std,
            training_losses=train_losses,
            validation_losses=val_losses,
            device=self.device,
            forecast_frame=forecast_frame,
            price_metrics=price_metrics,
            checkpoint_path=checkpoint_path,
            train_indices=train_idx,
            val_indices=val_idx,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    def predict_next_price(self, artifacts: TrainingArtifacts) -> float:
        dataset = artifacts.dataset
        last_features = dataset.features[artifacts.val_indices[-1]].unsqueeze(0).to(self.device)
        artifacts.model.eval()
        with torch.no_grad():
            predicted_log_return = artifacts.model(last_features).item()
        last_price = dataset.anchor_prices[artifacts.val_indices[-1]].item()
        return float(last_price * math.exp(predicted_log_return))


__all__ = ["HermiteTrainer", "TrainingArtifacts", "PriceMetrics"]
