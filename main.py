from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

from src.config import APP_CONFIG, AppConfig, load_config
from src.pipeline import HermiteTrainer
from src.pipeline.split import RollingOriginSplitter
from src.utils.logging import configure_logging
from src.utils.repo import forecast_horizon_owner, label_factory_location
from src.utils.utils import set_seed  # FIX: access deterministic seeding helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train probabilistic Hermite forecaster with diagnostics.")
    parser.add_argument("--config", type=Path, default=None, help="Optional path to YAML config overrides.")
    parser.add_argument(
        "--cv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable rolling-origin cross-validation (defaults to config).",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override conformal alpha (e.g. 0.1).")
    parser.add_argument("--threshold", type=float, default=None, help="Override trading threshold τ.")
    parser.add_argument("--cost-bps", type=float, default=None, help="Override transaction cost per trade in basis points.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--prob-source",
        choices=["cdf", "logit"],
        default=None,
        help="Select probability head output (CDF of Gaussian or sigmoid logits).",
    )
    parser.add_argument(
        "--use-lstm",
        dest="use_lstm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the temporal LSTM encoder regardless of config.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override the number of rolling-origin CV folds.",
    )
    parser.add_argument(
        "--save-md",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable saving Markdown results alongside CSV.",
    )
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory to store results table outputs.")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the effective configuration and data split indices, then exit.",
    )
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    cfg = config
    if args.seed is not None:
        cfg = replace(cfg, training=replace(cfg.training, seed=args.seed))
    if args.prob_source is not None:
        cfg = replace(cfg, model=replace(cfg.model, prob_source=args.prob_source))
    if args.use_lstm is not None:
        cfg = replace(cfg, model=replace(cfg.model, use_lstm=args.use_lstm))
    if args.cv_folds is not None:
        cfg = replace(cfg, training=replace(cfg.training, cv_folds=args.cv_folds))
    if args.alpha is not None:
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, alpha=args.alpha))
    if args.threshold is not None:
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, threshold=args.threshold))
    if args.cost_bps is not None:
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, cost_bps=args.cost_bps))
    if args.save_md is not None:
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, save_markdown=args.save_md))
    return cfg


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config) if args.config else APP_CONFIG
    config = apply_overrides(base_config, args)
    configure_logging(config.logging.level)
    set_seed(config.training.seed)  # FIX: initialise reproducible seed state

    if args.print_config:
        trainer = HermiteTrainer(config=config)
        dataset = trainer.prepare_dataset()
        splitter = RollingOriginSplitter(
            dataset_length=len(dataset),
            data_config=config.data,
            evaluation_config=config.evaluation,
        )
        effective_use_cv = args.cv if args.cv is not None else config.training.use_cv
        folds = splitter.split(use_cv=effective_use_cv)
        split_payload = [
            {
                "fold_id": fold.fold_id,
                "train_idx": fold.train_idx.tolist(),
                "calibration_idx": fold.calibration_idx.tolist(),
                "val_idx": fold.val_idx.tolist(),
                "scaler_idx": fold.scaler_idx.tolist(),
            }
            for fold in folds
        ]
        payload = {
            "config": asdict(config),
            "forecast_horizon_owner": forecast_horizon_owner(config),
            "label_factory": label_factory_location(),
            "num_samples": len(dataset),
            "folds": split_payload,
        }
        print(json.dumps(payload, indent=2, default=_json_default))
        return

    trainer = HermiteTrainer(config=config)
    artifacts = trainer.run(use_cv=args.cv, results_dir=args.results_dir)

    print("Folds trained:", len(artifacts.fold_results))
    if artifacts.csv_path:
        print(f"Results CSV: {artifacts.csv_path}")
    if artifacts.markdown_path:
        print(f"Results Markdown: {artifacts.markdown_path}")
    if artifacts.summary_path:
        print(f"Summary JSON: {artifacts.summary_path}")


if __name__ == "__main__":
    main()
