from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import BINANCE, FEATURES, TRAINING, FeatureConfig, TrainingConfig
from src.data import BinanceDataFetcher
from src.features import compute_liquidity_features, compute_orderbook_features
from src.models import HermiteForecaster


class HermiteDataset(Dataset):
    """Windowed dataset combining OHLCV, liquidity and order book features."""

    def __init__(
        self,
        candles: pd.DataFrame,
        liquidity_features: Dict[str, float],
        orderbook_features: Dict[str, float],
        *,
        feature_window: int,
        forecast_horizon: int,
        normalise: bool = True,
    ) -> None:
        super().__init__()
        required_cols = ["open", "high", "low", "close", "volume"]
        self.feature_window = feature_window
        self.forecast_horizon = forecast_horizon
        self.candles = candles.reset_index(drop=True).copy()
        values = self.candles[required_cols].to_numpy(dtype=np.float32)
        feature_vectors = []
        targets = []
        anchor_prices = []
        target_times = []

        liquidity_vector = np.array(list(liquidity_features.values()), dtype=np.float32)
        orderbook_vector = np.array(list(orderbook_features.values()), dtype=np.float32)
        extra_features = np.concatenate([liquidity_vector, orderbook_vector])

        total_length = len(candles)
        max_index = total_length - forecast_horizon
        for end in range(feature_window, max_index):
            window = values[end - feature_window : end]
            current_close = values[end - 1, 3]
            future_index = end + forecast_horizon - 1
            future_close = values[future_index, 3]
            window_features = window.flatten()
            features = np.concatenate([window_features, extra_features])
            feature_vectors.append(features)
            anchor_prices.append(current_close)
            targets.append(np.log(future_close / current_close))
            target_times.append(self.candles.iloc[future_index]["close_time"])

        self.features = np.vstack(feature_vectors).astype(np.float32)
        self.targets = np.array(targets, dtype=np.float32)[:, None]
        self.anchor_prices = np.array(anchor_prices, dtype=np.float32)
        self.target_times = pd.Series(target_times, name="target_time")

        if normalise:
            self.feature_mean = self.features.mean(axis=0)
            self.feature_std = self.features.std(axis=0) + 1e-6
            self.features = (self.features - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = np.zeros(self.features.shape[1], dtype=np.float32)
            self.feature_std = np.ones(self.features.shape[1], dtype=np.float32)

        self.features = torch.from_numpy(self.features)
        self.targets = torch.from_numpy(self.targets)
        self.anchor_prices = torch.from_numpy(self.anchor_prices)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]


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


class HermiteTrainer:
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

    def prepare_dataset(self) -> HermiteDataset:
        candles = self.fetcher.get_historical_candles(limit=BINANCE.history_limit)
        liquidity_features = compute_liquidity_features(self.fetcher, feature_config=self.feature_config)
        orderbook_features = compute_orderbook_features(self.fetcher, feature_config=self.feature_config)
        dataset = HermiteDataset(
            candles,
            liquidity_features,
            orderbook_features,
            feature_window=self.training_config.feature_window,
            forecast_horizon=self.training_config.forecast_horizon,
        )
        return dataset

    def train(self) -> TrainingArtifacts:
        dataset = self.prepare_dataset()
        total_length = len(dataset)
        val_length = max(1, int(total_length * self.training_config.validation_split))
        train_length = total_length - val_length
        if train_length <= 0:
            raise ValueError("Validation split too large for dataset size. Reduce validation_split.")

        generator = torch.Generator().manual_seed(self.training_config.random_seed)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_length, val_length],
            generator=generator,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

        model = HermiteForecaster(input_dim=dataset.input_dim, config=self.training_config).to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.training_config.learning_rate)
        loss_fn = torch.nn.MSELoss()

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
                f"Train Loss: {running_loss:.6f} - Val Loss: {val_loss:.6f}"
            )
            model.train()

        model.eval()
        with torch.no_grad():
            full_features = dataset.features.to(self.device)
            predicted_log_returns = model(full_features).squeeze(1).cpu().numpy()

        anchor_prices = dataset.anchor_prices.cpu().numpy()
        actual_log_returns = dataset.targets.squeeze(1).cpu().numpy()
        predicted_prices = anchor_prices * np.exp(predicted_log_returns)
        actual_prices = anchor_prices * np.exp(actual_log_returns)
        forecast_frame = pd.DataFrame(
            {
                "target_time": dataset.target_times,
                "actual_price": actual_prices,
                "predicted_price": predicted_prices,
            }
        )

        return TrainingArtifacts(
            model=model,
            dataset=dataset,
            feature_mean=torch.from_numpy(dataset.feature_mean),
            feature_std=torch.from_numpy(dataset.feature_std),
            training_losses=train_losses,
            validation_losses=val_losses,
            device=self.device,
            forecast_frame=forecast_frame,
        )

    def predict_next_price(self, artifacts: TrainingArtifacts) -> float:
        dataset = artifacts.dataset
        last_features = dataset.features[-1].unsqueeze(0).to(self.device)
        artifacts.model.eval()
        with torch.no_grad():
            predicted_log_return = artifacts.model(last_features).item()
        last_price = dataset.anchor_prices[-1].item()
        return float(last_price * np.exp(predicted_log_return))


__all__ = [
    "HermiteDataset",
    "HermiteTrainer",
    "TrainingArtifacts",
]
