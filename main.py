from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.pipeline import HermiteTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hermite NN forecaster for BTCUSDT")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the trained model state.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = HermiteTrainer()
    artifacts = trainer.train()
    print(f"Training device: {artifacts.device}")
    if artifacts.training_losses:
        print(
            f"Final losses -> Train: {artifacts.training_losses[-1]:.6f}, "
            f"Validation: {artifacts.validation_losses[-1]:.6f}"
        )
    next_price = trainer.predict_next_price(artifacts)
    print(f"Predicted next price: {next_price:.2f} USDT")
    if args.save:
        payload = {
            "model_state": artifacts.model.state_dict(),
            "feature_mean": artifacts.feature_mean,
            "feature_std": artifacts.feature_std,
            "training_losses": artifacts.training_losses,
            "validation_losses": artifacts.validation_losses,
            "device": str(artifacts.device),
        }
        torch.save(payload, args.save)
        print(f"Artifacts saved to {args.save}")


if __name__ == "__main__":
    main()
