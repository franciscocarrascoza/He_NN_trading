from __future__ import annotations

import copy
import itertools
import logging
import os
import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_CREATE_SHM", "FALSE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_FORKJOIN_BARRIER", "plain")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False

from src.config import (
    APP_CONFIG,
    AppConfig,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    ReportingConfig,
    TrainingConfig,
    load_config,
)
from src.data import BinanceDataFetcher, HermiteDataset
from src.eval.conformal import conformal_interval, conformal_p_value, compute_conformal_quantile
from src.eval.diagnostics import (
    CalibrationMetrics,
    binomial_test_pvalue,
    diebold_mariano,
    ljung_box,
    mincer_zarnowitz,
    probability_calibration_metrics,
    runs_test,
)
from src.eval.strategy import StrategyMetrics, baseline_strategies, evaluate_strategy
from src.models import HermiteForecaster
from src.pipeline.scaler import LeakageGuardScaler, ScalerStats
from src.pipeline.split import FoldIndices, RollingOriginSplitter
from src.reporting.plots import save_lr_range_plot, save_reliability_diagram
from src.features import compute_liquidity_features_series, compute_orderbook_features_series

LOGGER = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "set_num_threads"):
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def _select_device(preference: str) -> torch.device:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        preference = preference.lower()
        for index in range(device_count):
            name = torch.cuda.get_device_name(index).lower()
            if preference in name:
                LOGGER.info("Using CUDA device", extra={"event": "device_select", "device": name})
                return torch.device(f"cuda:{index}")
        LOGGER.info("Preferred CUDA device not found, using cuda:0", extra={"event": "device_fallback"})
        return torch.device("cuda:0")
    LOGGER.info("CUDA unavailable, using CPU", extra={"event": "device_cpu"})
    return torch.device("cpu")


def _seconds_per_step(anchor_times: torch.Tensor) -> float:
    if anchor_times.numel() < 2:
        return 3600.0
    diffs = anchor_times[1:] - anchor_times[:-1]
    median_ms = torch.median(diffs).item()
    return max(1.0, median_ms / 1000.0)


def _build_dataloader(
    features: torch.Tensor,
    targets: torch.Tensor,
    direction_labels: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    subset_features = features.index_select(0, indices)
    subset_targets = targets.index_select(0, indices)
    subset_labels = direction_labels.index_select(0, indices)
    dataset = TensorDataset(subset_features, subset_targets, subset_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _sigmoid_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    reduction: str = "mean",
) -> torch.Tensor:
    if gamma < 0:
        raise ValueError("Focal loss gamma must be non-negative.")
    logit_targets = targets.to(logits.dtype)
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, logit_targets, reduction="none")
    p_t = torch.where(logit_targets >= 0.5, prob, 1.0 - prob)
    modulating = torch.pow(1.0 - p_t, gamma)
    loss = modulating * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("Unsupported reduction for focal loss.")


def _build_optimizer(parameters: Iterable[torch.nn.Parameter], cfg_train: TrainingConfig) -> torch.optim.Optimizer:
    name = cfg_train.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg_train.learning_rate, weight_decay=cfg_train.weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=cfg_train.learning_rate, weight_decay=cfg_train.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=cfg_train.learning_rate,
            weight_decay=cfg_train.weight_decay,
            momentum=0.9,
            nesterov=False,
        )
    raise ValueError(f"Unsupported optimizer: {cfg_train.optimizer}.")


def _build_scheduler(
    optimiser: torch.optim.Optimizer,
    cfg_train: TrainingConfig,
    steps_per_epoch: int,
) -> tuple[Optional[torch.optim.lr_scheduler._LRScheduler], bool]:
    name = cfg_train.scheduler.lower()
    if name in {"", "none"}:
        return None, False
    if name == "onecycle":
        pct_start = max(0.0, min(cfg_train.scheduler_warmup_pct, 0.9))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=cfg_train.learning_rate,
            epochs=max(1, cfg_train.num_epochs),
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=pct_start,
            anneal_strategy="cos",
        )
        return scheduler, True
    if name in {"cosine", "cosineannealing"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=max(1, cfg_train.num_epochs),
        )
        return scheduler, False
    raise ValueError(f"Unsupported scheduler: {cfg_train.scheduler}.")


def _fit_temperature_scale(logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    logits = np.asarray(logits_np, dtype=float).reshape(-1)
    labels = np.asarray(labels_np, dtype=float).reshape(-1)
    if logits.size == 0 or np.all(labels == labels[0]):
        return 1.0

    def _loss(temp: float) -> float:
        temp = max(temp, 1e-3)
        scaled = logits / temp
        positive = np.maximum(scaled, 0.0)
        loss = positive - scaled * labels + np.log1p(np.exp(-np.abs(scaled)))
        return float(np.mean(loss))

    candidate_temps = np.concatenate(
        [
            np.linspace(0.1, 5.0, num=25),
            np.linspace(0.5, 10.0, num=25),
            np.array([1.0], dtype=float),
        ]
    )
    best_temp = 1.0
    best_loss = float("inf")
    for temp in candidate_temps:
        loss = _loss(temp)
        if loss < best_loss:
            best_loss = loss
            best_temp = temp
    refine_low = max(0.05, best_temp / 2.0)
    refine_high = min(20.0, best_temp * 2.0)
    for temp in np.linspace(refine_low, refine_high, num=40):
        loss = _loss(temp)
        if loss < best_loss:
            best_loss = loss
            best_temp = temp
    return float(np.clip(best_temp, 0.05, 20.0))


def _apply_temperature_scale(logits_np: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        temperature = 1.0
    scaled_logits = logits_np / temperature
    return 1.0 / (1.0 + np.exp(-scaled_logits))


def _fit_isotonic_regression(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if probabilities.size <= 1:
        return probabilities.copy(), labels.copy()
    order = np.argsort(probabilities)
    x_sorted = probabilities[order]
    y_sorted = labels[order]

    unique_x: List[float] = []
    unique_y: List[float] = []
    unique_w: List[float] = []
    current_x = x_sorted[0]
    sum_y = y_sorted[0]
    weight = 1.0
    for xi, yi in zip(x_sorted[1:], y_sorted[1:]):
        if abs(xi - current_x) < 1e-12:
            sum_y += yi
            weight += 1.0
        else:
            unique_x.append(current_x)
            unique_y.append(sum_y / weight)
            unique_w.append(weight)
            current_x = xi
            sum_y = yi
            weight = 1.0
    unique_x.append(current_x)
    unique_y.append(sum_y / weight)
    unique_w.append(weight)

    x = np.asarray(unique_x, dtype=float)
    y = np.asarray(unique_y, dtype=float)
    w = np.asarray(unique_w, dtype=float)
    n = x.size
    if n <= 1:
        return x, y

    y_iso = y.copy()
    w_iso = w.copy()
    i = 0
    while i < n - 1:
        if y_iso[i] > y_iso[i + 1] + 1e-12:
            total_weight = w_iso[i] + w_iso[i + 1]
            avg = (w_iso[i] * y_iso[i] + w_iso[i + 1] * y_iso[i + 1]) / total_weight
            y_iso[i] = avg
            y_iso[i + 1] = avg
            w_iso[i] = total_weight
            w_iso[i + 1] = total_weight
            j = i - 1
            while j >= 0 and y_iso[j] > y_iso[j + 1] + 1e-12:
                total_weight = w_iso[j] + w_iso[j + 1]
                avg = (w_iso[j] * y_iso[j] + w_iso[j + 1] * y_iso[j + 1]) / total_weight
                y_iso[j] = avg
                y_iso[j + 1] = avg
                w_iso[j] = total_weight
                w_iso[j + 1] = total_weight
                j -= 1
        i += 1
    return x, y_iso


def _apply_isotonic_regression(x_fit: np.ndarray, y_fit: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    if x_fit.size == 0:
        default = float(np.mean(y_fit)) if y_fit.size else 0.5
        return np.full_like(probabilities, default, dtype=float)
    return np.interp(probabilities, x_fit, y_fit, left=y_fit[0], right=y_fit[-1])


@dataclass
class FoldResult:
    fold_id: int
    metrics: Dict[str, float]
    calibration_metrics_raw: CalibrationMetrics
    calibration_metrics_calibrated: CalibrationMetrics
    calibration_method: str
    calibration_warning: Optional[str]
    calibration_params: Dict[str, Any]
    probability_collapse: bool
    coverage_warning: Optional[str]
    strategy_metrics: StrategyMetrics
    baseline_metrics: Dict[str, StrategyMetrics]
    quantile: float
    coverage: float
    interval_width: float
    conformal_p_values: np.ndarray
    calibration_residuals: np.ndarray
    forecast_frame: pd.DataFrame
    reliability_paths: Dict[str, Path]
    scaler_stats: ScalerStats
    training_losses: List[float]
    validation_losses: List[float]
    validation_brier_scores: List[float]
    training_nll_scores: List[float]
    training_bce_scores: List[float]
    training_brier_scores: List[float]
    validation_nll_scores: List[float]
    validation_bce_scores: List[float]
    learning_rates: List[float]
    lr_range_path: Optional[Path]


@dataclass
class TrainingArtifacts:
    config: AppConfig
    fold_results: List[FoldResult]
    results_table: pd.DataFrame
    csv_path: Optional[Path]
    markdown_path: Optional[Path]


class HermiteTrainer:
    def __init__(
        self,
        *,
        config: AppConfig = APP_CONFIG,
        fetcher: Optional[BinanceDataFetcher] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.fetcher = fetcher
        self.device = device or _select_device(config.training.device_preference)

    def prepare_dataset(self, *, use_extras: Optional[bool] = None) -> HermiteDataset:
        binance_cfg = self.config.binance
        fetcher = self.fetcher or BinanceDataFetcher(binance_cfg)
        candles = fetcher.get_historical_candles(limit=binance_cfg.history_limit)
        use_extras = self.config.data.use_extras if use_extras is None else use_extras
        feature_cfg = self.config.features
        liquidity_series = (
            compute_liquidity_features_series(candles, feature_config=feature_cfg) if use_extras else None
        )
        orderbook_series = (
            compute_orderbook_features_series(candles, feature_config=feature_cfg) if use_extras else None
        )
        dataset = HermiteDataset(
            candles,
            data_config=self.config.data,
            liquidity_features=liquidity_series,
            orderbook_features=orderbook_series,
        )
        return dataset

    def run(
        self,
        dataset: Optional[HermiteDataset] = None,
        *,
        use_cv: bool,
        results_dir: Optional[Path] = None,
    ) -> TrainingArtifacts:
        cfg = self.config
        dataset = dataset or self.prepare_dataset()
        seconds_step = _seconds_per_step(dataset.anchor_times)
        splitter = RollingOriginSplitter(
            dataset_length=len(dataset),
            data_config=cfg.data,
            evaluation_config=cfg.evaluation,
        )
        folds = splitter.split(use_cv=use_cv)
        LOGGER.info(
            "Prepared folds",
            extra={
                "event": "split_ready",
                "folds": len(folds),
                "use_cv": use_cv,
            },
        )
        fold_results: List[FoldResult] = []
        alpha = cfg.evaluation.alpha
        threshold = cfg.evaluation.threshold
        cost_bps = cfg.evaluation.cost_bps

        output_dir = (results_dir or Path(cfg.reporting.output_dir)).resolve()
        plot_dir = output_dir / "plots"

        for fold in folds:
            fold_result = self._run_fold(
                dataset=dataset,
                fold=fold,
                seconds_per_step=seconds_step,
                alpha=alpha,
                threshold=threshold,
                cost_bps=cost_bps,
                plot_dir=plot_dir,
            )
            fold_results.append(fold_result)

        results_table, csv_path, markdown_path = self._build_results(
            fold_results,
            cfg.reporting,
            cfg.data.forecast_horizon,
            alpha,
            cfg.evaluation.save_markdown,
            results_dir=output_dir,
        )

        return TrainingArtifacts(
            config=cfg,
            fold_results=fold_results,
            results_table=results_table,
            csv_path=csv_path,
            markdown_path=markdown_path,
        )

    def _lr_range_test(
        self,
        train_loader: DataLoader,
        cfg_train: TrainingConfig,
        feature_dim: int,
        pos_weight_tensor: Optional[torch.Tensor],
        plot_path: Path,
    ) -> Optional[Path]:
        min_lr = max(cfg_train.lr_range_min, 1e-7)
        max_lr = max(min_lr * 1.01, cfg_train.lr_range_max)
        steps = max(5, cfg_train.lr_range_steps)
        try:
            lr_values = np.logspace(np.log10(min_lr), np.log10(max_lr), num=steps)
        except ValueError:
            return None
        model = HermiteForecaster(
            input_dim=feature_dim,
            model_config=self.config.model,
        ).to(self.device)
        model.train()
        temp_cfg = replace(cfg_train, learning_rate=min_lr)
        optimiser = _build_optimizer(model.parameters(), temp_cfg)
        losses: List[float] = []
        data_iter = itertools.cycle(train_loader)
        classification_mode = cfg_train.classification_loss.lower()
        for lr in lr_values:
            batch_features, batch_targets, batch_labels = next(data_iter)
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            batch_labels = batch_labels.to(self.device)
            for param_group in optimiser.param_groups:
                param_group["lr"] = float(lr)
            optimiser.zero_grad()
            mu, logvar, _, logits = model(batch_features)
            nll = 0.5 * ((batch_targets - mu) ** 2 * torch.exp(-logvar) + logvar)
            labels_float = batch_labels.to(batch_targets.dtype)
            if classification_mode == "bce":
                bce = F.binary_cross_entropy_with_logits(
                    logits,
                    labels_float,
                    reduction="none",
                    pos_weight=pos_weight_tensor,
                )
            else:
                bce = _sigmoid_focal_loss_with_logits(
                    logits,
                    labels_float,
                    gamma=cfg_train.focal_gamma,
                    reduction="none",
                )
            loss = nll.mean() + cfg_train.lambda_bce * bce.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.gradient_clip)
            optimiser.step()
            losses.append(float(loss.item()))
        save_lr_range_plot(lr_values, losses, output_path=plot_path)
        return plot_path

    def _run_fold(
        self,
        *,
        dataset: HermiteDataset,
        fold: FoldIndices,
        seconds_per_step: float,
        alpha: float,
        threshold: float,
        cost_bps: float,
        plot_dir: Path,
    ) -> FoldResult:
        cfg_train = self.config.training
        cfg_model = self.config.model
        _set_global_seed(cfg_train.seed + fold.fold_id)

        scaler = LeakageGuardScaler(max_index=int(fold.scaler_idx[-1].item()))
        scaled_features, stats = scaler.fit_transform(dataset.features, fold.scaler_idx)
        scaled_targets = dataset.targets
        direction_labels_int = dataset.direction_labels.view(-1)
        direction_labels = dataset.direction_labels.to(dtype=torch.float32)

        def _report_class_balance(split: str, indices: torch.Tensor) -> tuple[int, int, float]:
            labels_subset = direction_labels_int.index_select(0, indices)
            total = int(labels_subset.numel())
            pos = int(labels_subset.sum().item())
            neg = total - pos
            frac = float(pos / total) if total > 0 else float("nan")
            LOGGER.info(
                "Class balance",
                extra={
                    "event": "class_balance",
                    "fold_id": fold.fold_id,
                    "split": split,
                    "total": total,
                    "pos": pos,
                    "neg": neg,
                    "p_pos": frac,
                },
            )
            return total, pos, frac

        train_indices = fold.train_idx
        train_total, train_pos, train_frac = _report_class_balance("train", train_indices)
        cal_total, cal_pos, cal_frac = _report_class_balance("calibration", fold.calibration_idx)
        val_total, val_pos, val_frac = _report_class_balance("validation", fold.val_idx)
        _ = (cal_total, cal_pos, cal_frac, val_total, val_pos, val_frac)  # quiet linters when unused

        train_loader = _build_dataloader(
            scaled_features,
            scaled_targets,
            direction_labels,
            train_indices,
            cfg_train.batch_size,
            shuffle=True,
        )
        val_loader = _build_dataloader(
            scaled_features,
            scaled_targets,
            direction_labels,
            fold.val_idx,
            cfg_train.batch_size,
            shuffle=False,
        )

        classification_mode = cfg_train.classification_loss.lower()
        if classification_mode not in {"bce", "focal"}:
            raise ValueError("training.classification_loss must be either 'bce' or 'focal'.")

        pos_weight_tensor: Optional[torch.Tensor] = None
        imbalance = abs(train_frac - 0.5) if not math.isnan(train_frac) else 0.0
        class_balance = min(train_frac, 1.0 - train_frac) if not math.isnan(train_frac) else 0.5
        if train_pos == 0 or train_pos == train_total:
            raise ValueError(
                f"Fold {fold.fold_id} has all-same-sign labels. "
                "Adjust forecast horizon or enable class balancing."
            )

        needs_mitigation = class_balance < 0.10
        imbalance_handled = False

        if classification_mode == "focal":
            imbalance_handled = True

        if (
            classification_mode == "bce"
            and cfg_train.auto_pos_weight
            and 0 < train_pos < train_total
        ):
            ratio = (train_total - train_pos) / max(train_pos, 1)
            pos_weight_tensor = torch.tensor(float(ratio), dtype=torch.float32, device=self.device)
            LOGGER.info(
                "Applying automatic positive class weighting",
                extra={
                    "event": "class_weight",
                    "fold_id": fold.fold_id,
                    "ratio": float(ratio),
                    "imbalance": imbalance,
                },
            )
            imbalance_handled = True

        if needs_mitigation and cfg_train.enable_class_downsample and not imbalance_handled:
            rng = np.random.default_rng(cfg_train.seed + fold.fold_id)
            train_idx_np = train_indices.cpu().numpy()
            train_labels_np = direction_labels_int.index_select(0, train_indices).cpu().numpy()
            majority_class = 1 if train_pos >= (train_total - train_pos) else 0
            minority_class = 1 - majority_class
            minority_indices = train_idx_np[train_labels_np == minority_class]
            majority_indices = train_idx_np[train_labels_np == majority_class]
            if minority_indices.size == 0:
                raise ValueError(
                    f"Fold {fold.fold_id} lacks minority class samples and cannot be downsampled."
                )
            rng.shuffle(majority_indices)
            keep_majority = majority_indices[: minority_indices.size]
            balanced_indices = np.concatenate([minority_indices, keep_majority])
            balanced_indices.sort()
            train_indices = torch.from_numpy(balanced_indices.astype(np.int64)).to(fold.train_idx.device)
            train_total = int(train_indices.numel())
            train_labels_after = direction_labels_int.index_select(0, train_indices)
            train_pos = int(train_labels_after.sum().item())
            train_frac = float(train_pos / max(train_total, 1))
            class_balance = min(train_frac, 1.0 - train_frac)
            imbalance_handled = True
            LOGGER.info(
                "Downsampled majority class for training fold",
                extra={
                    "event": "class_downsample",
                    "fold_id": fold.fold_id,
                    "new_train_size": train_total,
                    "class_balance": class_balance,
                },
            )
            _report_class_balance("train_downsampled", train_indices)

        if needs_mitigation and not imbalance_handled:
            raise ValueError(
                f"Fold {fold.fold_id} class balance {class_balance:.3f} requires mitigation. "
                "Enable focal loss, auto_pos_weight, or class downsampling."
            )

        # rebuild train loader if indices changed due to downsampling
        if not torch.equal(train_indices, fold.train_idx):
            train_loader = _build_dataloader(
                scaled_features,
                scaled_targets,
                direction_labels,
                train_indices,
                cfg_train.batch_size,
                shuffle=True,
            )

        lr_range_path: Optional[Path] = None
        if cfg_train.enable_lr_range_test and fold.fold_id == 0:
            try:
                lr_range_path = self._lr_range_test(
                    train_loader,
                    cfg_train,
                    scaled_features.shape[1],
                    pos_weight_tensor,
                    plot_dir / f"lr_range_fold_{fold.fold_id}.png",
                )
            except Exception as exc:  # pragma: no cover - diagnostic safeguard
                LOGGER.warning(
                    "LR range test failed",
                    extra={
                        "event": "lr_range_failure",
                        "fold_id": fold.fold_id,
                        "error": str(exc),
                    },
                )

        model = HermiteForecaster(
            input_dim=scaled_features.shape[1],
            model_config=cfg_model,
        ).to(self.device)
        optimiser = _build_optimizer(model.parameters(), cfg_train)
        steps_per_epoch = max(1, len(train_loader))
        scheduler, scheduler_batch_step = _build_scheduler(optimiser, cfg_train, steps_per_epoch)

        best_state = None
        best_val_loss = float("inf")
        best_val_brier = float("inf")
        patience = max(0, cfg_train.early_stopping_patience)
        patience_counter = 0
        training_losses: List[float] = []
        validation_losses: List[float] = []
        validation_brier_scores: List[float] = []
        training_nll_scores: List[float] = []
        training_bce_scores: List[float] = []
        training_brier_scores: List[float] = []
        validation_nll_scores: List[float] = []
        validation_bce_scores: List[float] = []
        learning_rates: List[float] = []

        for epoch in range(cfg_train.num_epochs):
            model.train()
            running_loss = 0.0
            running_nll = 0.0
            running_bce = 0.0
            train_brier_sum = 0.0
            count = 0
            for batch_features, batch_targets, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_labels = batch_labels.to(self.device)
                target_sign = (batch_targets.view(-1) > 0).to(torch.int64)
                label_sign = batch_labels.view(-1).to(torch.int64)
                if not torch.equal(target_sign, label_sign):
                    raise RuntimeError("Direction labels mismatch regression target sign within batch.")
                optimiser.zero_grad()
                mu, logvar, _, logits = model(batch_features)
                nll = 0.5 * ((batch_targets - mu) ** 2 * torch.exp(-logvar) + logvar)
                labels_float = batch_labels.to(batch_targets.dtype)
                if classification_mode == "bce":
                    bce = F.binary_cross_entropy_with_logits(
                        logits,
                        labels_float,
                        reduction="none",
                        pos_weight=pos_weight_tensor,
                    )
                else:
                    bce = _sigmoid_focal_loss_with_logits(
                        logits,
                        labels_float,
                        gamma=cfg_train.focal_gamma,
                        reduction="none",
                    )
                bce_mean = bce.mean()
                nll_mean = nll.mean()
                loss = nll_mean + cfg_train.lambda_bce * bce_mean
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.gradient_clip)
                optimiser.step()
                if scheduler is not None and scheduler_batch_step:
                    scheduler.step()
                batch_size = batch_features.size(0)
                running_loss += loss.item() * batch_size
                running_nll += float(nll_mean.item()) * batch_size
                running_bce += float(bce_mean.item()) * batch_size
                prob = torch.sigmoid(logits)
                train_brier_sum += float(((prob - labels_float) ** 2).sum().item())
                count += batch_size
            epoch_train_loss = running_loss / max(1, count)
            epoch_train_nll = running_nll / max(1, count)
            epoch_train_bce = running_bce / max(1, count)
            epoch_train_brier = train_brier_sum / max(1, count)
            training_losses.append(epoch_train_loss)
            training_nll_scores.append(epoch_train_nll)
            training_bce_scores.append(epoch_train_bce)
            training_brier_scores.append(epoch_train_brier)

            model.eval()
            val_loss = 0.0
            val_count = 0
            val_brier_sum = 0.0
            val_nll_sum = 0.0
            val_bce_sum = 0.0
            with torch.no_grad():
                for batch_features, batch_targets, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    target_sign = (batch_targets.view(-1) > 0).to(torch.int64)
                    label_sign = batch_labels.view(-1).to(torch.int64)
                    if not torch.equal(target_sign, label_sign):
                        raise RuntimeError("Direction labels mismatch regression target sign within validation batch.")
                    mu, logvar, _, logits = model(batch_features)
                    nll = 0.5 * ((batch_targets - mu) ** 2 * torch.exp(-logvar) + logvar)
                    labels_float = batch_labels.to(batch_targets.dtype)
                    if classification_mode == "bce":
                        bce = F.binary_cross_entropy_with_logits(
                            logits,
                            labels_float,
                            reduction="none",
                            pos_weight=pos_weight_tensor,
                        )
                    else:
                        bce = _sigmoid_focal_loss_with_logits(
                            logits,
                            labels_float,
                            gamma=cfg_train.focal_gamma,
                            reduction="none",
                        )
                    bce_mean = bce.mean()
                    nll_mean = nll.mean()
                    loss = nll_mean + cfg_train.lambda_bce * bce_mean
                    batch_size = batch_features.size(0)
                    val_loss += loss.item() * batch_size
                    val_count += batch_size
                    val_nll_sum += float(nll_mean.item()) * batch_size
                    val_bce_sum += float(bce_mean.item()) * batch_size
                    prob = torch.sigmoid(logits)
                    brier_batch = ((prob - labels_float) ** 2).sum()
                    val_brier_sum += float(brier_batch.item())
            epoch_val_loss = val_loss / max(1, val_count)
            epoch_val_brier = val_brier_sum / max(1, val_count)
            epoch_val_nll = val_nll_sum / max(1, val_count)
            epoch_val_bce = val_bce_sum / max(1, val_count)
            validation_losses.append(epoch_val_loss)
            validation_brier_scores.append(epoch_val_brier)
            validation_nll_scores.append(epoch_val_nll)
            validation_bce_scores.append(epoch_val_bce)

            current_lr = float(optimiser.param_groups[0]["lr"])
            learning_rates.append(current_lr)

            LOGGER.info(
                "Epoch metrics",
                extra={
                    "event": "epoch_metrics",
                    "fold_id": fold.fold_id,
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_nll": epoch_train_nll,
                    "train_bce": epoch_train_bce,
                    "train_brier": epoch_train_brier,
                    "val_loss": epoch_val_loss,
                    "val_nll": epoch_val_nll,
                    "val_bce": epoch_val_bce,
                    "val_brier": epoch_val_brier,
                    "lr": current_lr,
                },
            )

            if not math.isfinite(epoch_train_loss) or not math.isfinite(epoch_val_loss):
                LOGGER.warning(
                    "Non-finite loss encountered; stopping training",
                    extra={"event": "loss_divergence", "fold_id": fold.fold_id, "epoch": epoch},
                )
                break
            if epoch_train_loss > 1e6 or epoch_val_loss > 1e6:
                LOGGER.warning(
                    "Loss exceeded stability threshold; stopping training",
                    extra={"event": "loss_divergence", "fold_id": fold.fold_id, "epoch": epoch},
                )
                break

            if scheduler is not None and not scheduler_batch_step:
                scheduler.step()

            if epoch_val_loss + 1e-8 < best_val_loss or epoch_val_brier + 1e-8 < best_val_brier:
                best_val_loss = epoch_val_loss
                best_val_brier = min(best_val_brier, epoch_val_brier)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            val_features = scaled_features.index_select(0, fold.val_idx).to(self.device)
            val_targets = scaled_targets.index_select(0, fold.val_idx).to(self.device)
            mu_val, logvar_val, p_up_val, logits_val = model(val_features)
            cal_features = scaled_features.index_select(0, fold.calibration_idx).to(self.device)
            cal_targets = scaled_targets.index_select(0, fold.calibration_idx).to(self.device)
            mu_cal, logvar_cal, p_up_cal, logits_cal = model(cal_features)

        mu_val_np = mu_val.squeeze(1).cpu().numpy()
        logvar_val_np = logvar_val.squeeze(1).cpu().numpy()
        sigma_val_np = np.exp(0.5 * logvar_val_np)
        sigma_val_np = np.clip(sigma_val_np, 1e-6, None)
        p_up_val_np = p_up_val.squeeze(1).cpu().numpy()
        logits_val_np = logits_val.squeeze(1).cpu().numpy()
        y_val_np = val_targets.squeeze(1).cpu().numpy()
        mu_cal_np = mu_cal.squeeze(1).cpu().numpy()
        logvar_cal_np = logvar_cal.squeeze(1).cpu().numpy()
        p_up_cal_np = p_up_cal.squeeze(1).cpu().numpy()
        logits_cal_np = logits_cal.squeeze(1).cpu().numpy()
        y_cal_np = cal_targets.squeeze(1).cpu().numpy()
        sigma_cal_np = np.exp(0.5 * logvar_cal_np)
        sigma_cal_np = np.clip(sigma_cal_np, 1e-6, None)
        cal_residuals = np.abs(y_cal_np - mu_cal_np) / sigma_cal_np
        direction_cal_np = direction_labels_int.index_select(0, fold.calibration_idx).cpu().numpy().astype(int)
        direction_val_np = direction_labels_int.index_select(0, fold.val_idx).cpu().numpy().astype(int)

        hist_bins = np.linspace(0.0, 1.0, 21)
        hist_counts, _ = np.histogram(p_up_val_np, bins=hist_bins)
        collapse_low = float((p_up_val_np < 0.05).mean())
        collapse_high = float((p_up_val_np > 0.95).mean())
        probability_collapse = collapse_low > 0.9 or collapse_high > 0.9
        LOGGER.info(
            "Probability head histogram",
            extra={
                "event": "probability_histogram",
                "fold_id": fold.fold_id,
                "counts": hist_counts.tolist(),
                "bins": hist_bins.tolist(),
                "collapse_low": collapse_low,
                "collapse_high": collapse_high,
            },
        )
        if probability_collapse:
            LOGGER.warning(
                "Probability head collapse detected",
                extra={
                    "event": "probability_collapse",
                    "fold_id": fold.fold_id,
                    "collapse_low": collapse_low,
                    "collapse_high": collapse_high,
                },
            )

        def _make_candidate(name: str, prob_cal: np.ndarray, prob_val: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
            prob_cal = np.clip(np.asarray(prob_cal, dtype=float), 1e-6, 1.0 - 1e-6)
            prob_val = np.clip(np.asarray(prob_val, dtype=float), 1e-6, 1.0 - 1e-6)
            return {
                "name": name,
                "prob_cal": prob_cal,
                "prob_val": prob_val,
                "metrics_cal": probability_calibration_metrics(y_cal_np, prob_cal),
                "metrics_val": probability_calibration_metrics(y_val_np, prob_val),
                "params": params,
            }

        calibration_candidates: List[Dict[str, Any]] = []
        calibration_candidates.append(_make_candidate("raw", p_up_cal_np, p_up_val_np, {"temperature": 1.0}))

        temperature = _fit_temperature_scale(logits_cal_np, direction_cal_np)
        temp_prob_cal = _apply_temperature_scale(logits_cal_np, temperature)
        temp_prob_val = _apply_temperature_scale(logits_val_np, temperature)
        calibration_candidates.append(
            _make_candidate("temperature", temp_prob_cal, temp_prob_val, {"temperature": temperature})
        )

        iso_x, iso_y = _fit_isotonic_regression(p_up_cal_np, direction_cal_np)
        iso_prob_cal = _apply_isotonic_regression(iso_x, iso_y, p_up_cal_np)
        iso_prob_val = _apply_isotonic_regression(iso_x, iso_y, p_up_val_np)
        calibration_candidates.append(
            _make_candidate("isotonic", iso_prob_cal, iso_prob_val, {"isotonic_x": iso_x, "isotonic_y": iso_y})
        )

        iso_temp_x, iso_temp_y = _fit_isotonic_regression(temp_prob_cal, direction_cal_np)
        iso_temp_prob_cal = _apply_isotonic_regression(iso_temp_x, iso_temp_y, temp_prob_cal)
        iso_temp_prob_val = _apply_isotonic_regression(iso_temp_x, iso_temp_y, temp_prob_val)
        calibration_candidates.append(
            _make_candidate(
                "temp_isotonic",
                iso_temp_prob_cal,
                iso_temp_prob_val,
                {"temperature": temperature, "isotonic_x": iso_temp_x, "isotonic_y": iso_temp_y},
            )
        )

        def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple[float, float]:
            metrics_val = candidate["metrics_val"]
            return (math.inf if math.isnan(metrics_val.brier) else metrics_val.brier,
                    math.inf if math.isnan(metrics_val.ece) else metrics_val.ece)

        best_candidate = min(calibration_candidates, key=_candidate_sort_key)
        raw_candidate = next(candidate for candidate in calibration_candidates if candidate["name"] == "raw")

        calibration_method = best_candidate["name"]
        calibration_params = best_candidate["params"]
        calibration_metrics_raw = raw_candidate["metrics_val"]
        calibration_metrics_calibrated = best_candidate["metrics_val"]
        p_up_calibrated_val = best_candidate["prob_val"]

        calibration_warning: Optional[str] = None
        raw_ece = calibration_metrics_raw.ece
        best_ece = calibration_metrics_calibrated.ece
        if not math.isnan(raw_ece) and raw_ece > 0.0 and not math.isnan(best_ece):
            improvement = (raw_ece - best_ece) / raw_ece
            if improvement < 0.2:
                calibration_warning = "Calibration ECE improvement below 20%."
                LOGGER.warning(
                    calibration_warning,
                    extra={
                        "event": "calibration_warning",
                        "fold_id": fold.fold_id,
                        "raw_ece": raw_ece,
                        "calibrated_ece": best_ece,
                        "method": calibration_method,
                    },
                )
        quantile = compute_conformal_quantile(cal_residuals, alpha)
        realised_error = y_val_np - mu_val_np
        normalized_error = np.abs(realised_error) / sigma_val_np
        lower, upper = conformal_interval(mu_val_np, quantile * sigma_val_np)
        p_values = conformal_p_value(cal_residuals, normalized_error)

        anchor_prices = dataset.anchor_prices.index_select(0, fold.val_idx).numpy()
        anchor_times = dataset.anchor_times.index_select(0, fold.val_idx).numpy()
        target_times = dataset.target_times.index_select(0, fold.val_idx).numpy()
        actual_prices = anchor_prices * np.exp(y_val_np)
        predicted_prices = anchor_prices * np.exp(mu_val_np)
        interval_lower_price = anchor_prices * np.exp(lower)
        interval_upper_price = anchor_prices * np.exp(upper)

        residual_price = predicted_prices - actual_prices
        mae_price = float(np.abs(residual_price).mean())
        rmse_price = float(np.sqrt(np.mean(residual_price ** 2)))
        smape_price = float(200.0 * np.mean(np.abs(predicted_prices - actual_prices) / (np.abs(predicted_prices) + np.abs(actual_prices) + 1e-9)))

        mae_return = float(np.mean(np.abs(realised_error)))
        rmse_return = float(np.sqrt(np.mean(realised_error ** 2)))

        direction_pred = (p_up_calibrated_val >= 0.5).astype(int)
        direction_true = direction_val_np
        dir_acc = float(np.mean(direction_pred == direction_true))
        binom_p = binomial_test_pvalue(int((direction_pred == direction_true).sum()), direction_pred.size)

        zero_benchmark = np.zeros_like(y_val_np)
        dm_mse = diebold_mariano(y_val_np, mu_val_np, zero_benchmark, horizon=self.config.data.forecast_horizon, loss="mse")
        dm_mae = diebold_mariano(y_val_np, mu_val_np, zero_benchmark, horizon=self.config.data.forecast_horizon, loss="mae")

        mz = mincer_zarnowitz(y_val_np, mu_val_np)
        runs_p = runs_test(realised_error)
        ljung_q, ljung_p = ljung_box(realised_error, max_lag=None)

        calibration_raw_metrics = calibration_metrics_raw
        calibration_calibrated_metrics = calibration_metrics_calibrated

        strategy = evaluate_strategy(
            y_val_np,
            p_up_calibrated_val,
            threshold=threshold,
            cost_bps=cost_bps,
            seconds_per_step=seconds_per_step,
        )
        baselines = baseline_strategies(
            y_val_np,
            cost_bps=cost_bps,
            seconds_per_step=seconds_per_step,
        )

        coverage = float(np.mean((y_val_np >= lower) & (y_val_np <= upper)))
        interval_width = float(np.mean(upper - lower))
        coverage_warning: Optional[str] = None
        target_coverage = 1.0 - alpha
        if coverage < 0.85:
            coverage_warning = "under-dispersion"
            LOGGER.warning(
                "Conformal under-dispersion detected",
                extra={
                    "event": "conformal_warning",
                    "fold_id": fold.fold_id,
                    "coverage": coverage,
                    "target": target_coverage,
                },
            )
        elif coverage > 0.95:
            coverage_warning = "over-dispersion"
            LOGGER.warning(
                "Conformal over-dispersion detected",
                extra={
                    "event": "conformal_warning",
                    "fold_id": fold.fold_id,
                    "coverage": coverage,
                    "target": target_coverage,
                },
            )

        forecast_frame = pd.DataFrame(
            {
                "fold": fold.fold_id,
                "anchor_time": anchor_times,
                "target_time": target_times,
                "anchor_price": anchor_prices,
                "true_log_return": y_val_np,
                "pred_mean": mu_val_np,
                "pred_logvar": logvar_val_np,
                "pred_sigma": sigma_val_np,
                "prob_up": p_up_calibrated_val,
                "prob_up_raw": p_up_val_np,
                "logits": logits_val_np,
                "pred_price": predicted_prices,
                "true_price": actual_prices,
                "conformal_lower": lower,
                "conformal_upper": upper,
                "conformal_lower_price": interval_lower_price,
                "conformal_upper_price": interval_upper_price,
                "conformal_p": p_values,
                "normalized_abs_error": normalized_error,
            }
        )

        reliability_paths = {
            "raw": plot_dir / f"reliability_raw_fold_{fold.fold_id}.png",
            "calibrated": plot_dir / f"reliability_calibrated_fold_{fold.fold_id}.png",
        }
        save_reliability_diagram(
            calibration_raw_metrics,
            output_path=reliability_paths["raw"],
            title=f"Reliability Diagram (Raw Fold {fold.fold_id})",
        )
        save_reliability_diagram(
            calibration_calibrated_metrics,
            output_path=reliability_paths["calibrated"],
            title=f"Reliability Diagram (Calibrated Fold {fold.fold_id})",
        )

        coverage_key = f"Conf_Coverage@{int(round((1 - alpha) * 100))}%"
        width_key = f"Conf_Width@{int(round((1 - alpha) * 100))}%"

        metrics = {
            "horizon": self.config.data.forecast_horizon,
            "MAE_return": mae_return,
            "RMSE_return": rmse_return,
            "MAE_price": mae_price,
            "sMAPE_price": smape_price,
            "DirAcc": dir_acc,
            "Binom_p": binom_p,
            "DM_p_SE": dm_mse.p_value,
            "DM_d_SE": dm_mse.mean_loss_diff,
            "DM_p_AE": dm_mae.p_value,
            "DM_d_AE": dm_mae.mean_loss_diff,
            "MZ_intercept": mz.intercept,
            "MZ_slope": mz.slope,
            "MZ_F_p": mz.p_value,
            "Runs_p": runs_p,
            "LjungBox_p": ljung_p,
            "Brier": calibration_calibrated_metrics.brier,
            "Brier_raw": calibration_raw_metrics.brier,
            "Brier_uncertainty": calibration_calibrated_metrics.brier_uncertainty,
            "Brier_uncertainty_raw": calibration_raw_metrics.brier_uncertainty,
            "Brier_resolution": calibration_calibrated_metrics.brier_resolution,
            "Brier_resolution_raw": calibration_raw_metrics.brier_resolution,
            "Brier_reliability": calibration_calibrated_metrics.brier_reliability,
            "Brier_reliability_raw": calibration_raw_metrics.brier_reliability,
            "AUC": calibration_calibrated_metrics.auc,
            "AUC_raw": calibration_raw_metrics.auc,
            "ECE": calibration_calibrated_metrics.ece,
            "ECE_raw": calibration_raw_metrics.ece,
            coverage_key: coverage,
            width_key: interval_width,
            "Sharpe_strategy": strategy.sharpe,
            "MDD_strategy": strategy.max_drawdown,
            "Turnover": strategy.turnover,
            "Sharpe_naive_long": baselines["always_long"].sharpe,
            "Sharpe_naive_flat": baselines["always_flat"].sharpe,
        }

        LOGGER.info(
            "Fold metrics",
            extra={
                "event": "fold_complete",
                "fold_id": fold.fold_id,
                "DirAcc": dir_acc,
                "coverage": coverage,
                "index_hash": stats.index_hash,
                "probability_collapse": probability_collapse,
                "calibration_method": calibration_method,
            },
        )

        return FoldResult(
            fold_id=fold.fold_id,
            metrics=metrics,
            calibration_metrics_raw=calibration_raw_metrics,
            calibration_metrics_calibrated=calibration_calibrated_metrics,
            calibration_method=calibration_method,
            calibration_warning=calibration_warning,
            calibration_params=calibration_params,
            probability_collapse=probability_collapse,
            coverage_warning=coverage_warning,
            strategy_metrics=strategy,
            baseline_metrics=baselines,
            quantile=quantile,
            coverage=coverage,
            interval_width=interval_width,
            conformal_p_values=p_values,
            calibration_residuals=cal_residuals,
            forecast_frame=forecast_frame,
            reliability_paths=reliability_paths,
            scaler_stats=stats,
            training_losses=training_losses,
            validation_losses=validation_losses,
            validation_brier_scores=validation_brier_scores,
            training_nll_scores=training_nll_scores,
            training_bce_scores=training_bce_scores,
            training_brier_scores=training_brier_scores,
            validation_nll_scores=validation_nll_scores,
            validation_bce_scores=validation_bce_scores,
            learning_rates=learning_rates,
            lr_range_path=lr_range_path,
        )

    def _build_results(
        self,
        fold_results: Sequence[FoldResult],
        reporting: ReportingConfig,
        horizon: int,
        alpha: float,
        save_markdown: bool,
        *,
        results_dir: Optional[Path],
    ) -> tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
        metrics_df = pd.DataFrame([fold.metrics for fold in fold_results])
        mean_series = metrics_df.mean()
        std_series = metrics_df.std(ddof=0)

        summary = {}
        for column in metrics_df.columns:
            mean_value = mean_series[column]
            std_value = std_series[column]
            if metrics_df.shape[0] > 1:
                summary[column] = f"{mean_value:.6f}±{std_value:.6f}"
            else:
                summary[column] = f"{mean_value:.6f}"
        summary["folds"] = metrics_df.shape[0]
        results_table = pd.DataFrame([summary])

        output_dir = results_dir or Path(reporting.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        results_table.to_csv(csv_path, index=False)

        markdown_path: Optional[Path] = None
        if save_markdown:
            markdown_path = output_dir / f"{csv_path.stem}.md"
            self._write_markdown(results_table, markdown_path, alpha)

        return results_table, csv_path, markdown_path

    def _write_markdown(self, table: pd.DataFrame, path: Path, alpha: float) -> None:
        headers = list(table.columns)
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in table.iterrows():
            values = [str(row[col]) for col in headers]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
        lines.append("Legend")
        lines.append("------")
        coverage_label = f"Conf_Coverage@{int(round((1 - alpha) * 100))}%"
        width_label = f"Conf_Width@{int(round((1 - alpha) * 100))}%"
        legend_items = {
            "p_up": "calibrated probability that next return > 0",
            "DirAcc": "directional accuracy on out-of-sample predictions",
            "Binom_p": "two-sided binomial test p-value vs 50% accuracy",
            "DM_p_SE": "Diebold–Mariano p-value using squared error loss vs zero-return benchmark",
            "DM_d_SE": "Mean squared-error loss difference (model − benchmark); negative favours the model",
            "DM_p_AE": "Diebold–Mariano p-value using absolute error loss vs zero-return benchmark",
            "DM_d_AE": "Mean absolute-error loss difference (model − benchmark); negative favours the model",
            "MZ_intercept": "Mincer–Zarnowitz regression intercept",
            "MZ_slope": "Mincer–Zarnowitz regression slope",
            "MZ_F_p": "Joint p-value for intercept=0 and slope=1",
            "Brier": "Calibrated Brier score (probability accuracy; lower is better)",
            "Brier_raw": "Uncalibrated Brier score before probability calibration",
            "Brier_uncertainty": "Uncertainty component of the calibrated Brier decomposition",
            "Brier_uncertainty_raw": "Uncertainty component before calibration",
            "Brier_resolution": "Resolution component of the calibrated Brier decomposition (higher is better)",
            "Brier_resolution_raw": "Resolution component before calibration",
            "Brier_reliability": "Reliability component (calibration error) after probability calibration",
            "Brier_reliability_raw": "Reliability component before probability calibration",
            "AUC": "Calibrated ROC AUC for probability head",
            "AUC_raw": "ROC AUC before probability calibration",
            "ECE": "Expected Calibration Error after probability calibration",
            "ECE_raw": "Expected Calibration Error before probability calibration",
            coverage_label: f"empirical coverage of conformal intervals at {(1 - alpha) * 100:.0f}%",
            width_label: f"mean width of conformal intervals at {(1 - alpha) * 100:.0f}%",
            "MDD_strategy": "maximum drawdown of threshold strategy",
            "Sharpe_strategy": "annualised Sharpe of threshold strategy",
            "Turnover": "average absolute position change per step",
            "Sharpe_naive_long": "annualised Sharpe of always-long baseline",
            "Sharpe_naive_flat": "annualised Sharpe of flat baseline",
        }
        for key, description in legend_items.items():
            lines.append(f"- {key}: {description}.")
        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")


__all__ = ["HermiteTrainer", "FoldResult", "TrainingArtifacts"]
