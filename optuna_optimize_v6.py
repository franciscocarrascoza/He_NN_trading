# -*- coding: utf-8 -*-
# optuna_optimize_v6.py
"""
STAGE 1: MODEL QUALITY OPTIMIZATION (Two-Stage Strategy)

Professional-grade Optuna optimization focusing on model architecture and training quality.
This stage optimizes for predictive performance (AUC + DirAcc), ignoring trading strategy metrics.

Key Features:
- Fixes strategy parameters at known-good values from v4 best trial
- Focuses on: model architecture, learning dynamics, loss balance, regularization
- Objective: 70% AUC + 30% DirAcc (pure prediction quality, no Sharpe)
- Search space: 13 critical parameters (down from 24)
- Target: 2500 trials for statistical coverage (~10x coverage per parameter dimension)
- Expanded ranges to include v4 optimal region

Strategy Parameters (FIXED):
- threshold = 0.624 (v4 best)
- confidence_margin = 0.105 (v4 best)
- kelly_clip = 0.940 (v4 best)
- batch_size = 512 (proven best for this dataset)

Next Stage (Stage 2):
- Fix model params at best from this stage
- Optimize strategy execution parameters for maximum Sharpe

Usage:
    /home/francisco/.anaconda3/envs/binance_env/bin/python optuna_optimize_v6.py

Expected Runtime: ~10-12 hours for 2500 trials (15-17s per trial)
"""
from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import replace
import optuna
from optuna.trial import TrialState
from tqdm import tqdm
import torch
import sys

# Replace these imports with your project's modules (assumed to exist in your environment)
from src.config import APP_CONFIG
from src.pipeline import HermiteTrainer
from src.utils.utils import set_seed

# --- Minimal stubs if running standalone for syntax-checking (remove if real modules exist) ---
# BEGIN STUBS (remove when running in your project)
try:
    APP_CONFIG  # type: ignore
except NameError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ModelConfig:
        hermite_version: str = "probabilist"
        hermite_degree: int = 7
        hermite_maps_a: int = 7
        hermite_maps_b: int = 2
        hermite_hidden_dim: int = 128
        dropout: float = 0.117
        use_lstm: bool = True
        lstm_hidden: int = 48
        prob_source: str = "pdf"

    @dataclass(frozen=True)
    class TrainingConfig:
        lr: float = 0.00012
        batch_size: int = 512
        weight_decay: float = 0.01
        grad_clip: float = 4.28
        reg_weight: float = 0.94
        cls_weight: float = 0.30
        unc_weight: float = 0.695
        sign_hinge_weight: float = 1.20
        classification_loss: str = "focal"
        focal_gamma: float = 2.7081

    @dataclass(frozen=True)
    class DataConfig:
        feature_window: int = 252

    @dataclass(frozen=True)
    class EvalConfig:
        threshold: float = 0.639
        confidence_margin: float = 0.079
        kelly_clip: float = 0.895
        use_kelly_position: bool = True
        use_confidence_margin: bool = True

    @dataclass(frozen=True)
    class AppConfig:
        model: ModelConfig = ModelConfig()
        training: TrainingConfig = TrainingConfig()
        data: DataConfig = DataConfig()
        eval: EvalConfig = EvalConfig()

    APP_CONFIG = AppConfig()  # template frozen config

    # Minimal trainer stub — replace with your real HermiteTrainer
    class HermiteTrainer:
        def __init__(self, config, use_cv=False, cv_folds=3):
            self.config = config
            self.use_cv = use_cv
            self.cv_folds = cv_folds

        def train(self):
            # Fake training result object with attributes used below.
            class R: pass
            r = R()
            # deterministic-ish fake metrics based on hermite_degree (for testing)
            deg = getattr(self.config.model, "hermite_degree", 5)
            r.sharpe = float(1.0 + 0.1 * (deg - 5))
            r.auc = 0.6
            r.dir_acc = 0.55
            r.turnover = 0.1
            return r

    def set_seed(s: int):
        import random, numpy as np
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
# END STUBS

# ----------------------------- USER CONFIGURATION -------------------------------- #
STUDY_NAME = "hermite_v6_stage1_model_quality"
STORAGE = "sqlite:///optuna_hermite_v6_stage1.db"
N_TRIALS = 2500  # Professional coverage: ~192 trials per parameter dimension
SEED = 42

# STAGE 1: Focus on prediction quality only (no Sharpe)
W_AUC = 0.70      # Primary: ROC AUC measures probability calibration quality
W_DIRACC = 0.30   # Secondary: Directional accuracy is the ultimate prediction goal
# Note: Sharpe is NOT used in Stage 1 - we optimize model quality first

# FIXED STRATEGY PARAMETERS (from v4 best trial, Sharpe=1.20)
FIXED_THRESHOLD = 0.6243501180239593
FIXED_CONFIDENCE_MARGIN = 0.10478522730539439
FIXED_KELLY_CLIP = 0.9399874126598519
FIXED_BATCH_SIZE = 512  # Proven optimal for this dataset size

# --- STAGE 1 SEARCH SPACE: Model Architecture & Training Quality --- #
# Categorical parameters: provide discrete choices for architectural decisions
HERMITES = [4, 5, 6, 7, 8]  # Expanded: include degree 8 for higher expressiveness
HERMITE_MAPS_A = [4, 5, 6, 7, 8, 9]  # Full range: controls polynomial feature expansion
HERMITE_MAPS_B = [1, 2, 3, 4, 5]  # Full range: controls interaction depth
HIDDEN_DIMS = [64, 96, 128, 160, 192]  # Finer granularity: 25% steps
LSTM_HIDDEN_CHOICES = [32, 48, 64, 80, 96]  # Expanded for better sequential modeling
HERMITES_VERSION_CHOICES = ["probabilist", "physicist"]  # Both conventions
CLASSIFICATION_LOSS_CHOICES = ["bce", "focal"]  # Both loss types

# FIXED architectural choices (proven effective in v4)
FIXED_PROB_SOURCE = "pdf"  # PDF-based probabilities worked best in v4
FIXED_USE_LSTM = True  # LSTM consistently improves temporal modeling

# Strong starting point: v4 best trial (Sharpe=1.20) adapted for Stage 1
INITIAL_PARAMS = {
    # Model architecture (OPTIMIZED in Stage 1)
    "hermite_version": "probabilist",
    "hermite_degree": 7,
    "hermite_maps_a": 5,  # v4 best value
    "hermite_maps_b": 1,  # v4 best value
    "hermite_hidden_dim": 192,  # v4 best value
    "dropout": 0.05946113101962681,  # v4 best value
    "lstm_hidden": 48,
    "classification_loss": "focal",
    "focal_gamma": 2.852771785107837,  # v4 best value

    # Learning dynamics (OPTIMIZED in Stage 1)
    "lr": 1.8200268769372706e-05,  # v4 best value
    "weight_decay": 0.0475382203767255,  # v4 best value
    "grad_clip": 3.2475220357203796,  # v4 best value

    # Loss balance (OPTIMIZED in Stage 1)
    "reg_weight": 2.953533795194038,  # v4 best value (outside v5 range!)
    "cls_weight": 1.4018596189337287,  # v4 best value
    "unc_weight": 0.2071804671376801,  # v4 best value
    "sign_hinge_weight": 0.6247833248810852,  # v4 best value (outside v5 range!)

    # Data/temporal (OPTIMIZED in Stage 1)
    "feature_window": 227,  # v4 best value (note: not 252!)
}

# ----------------------------- Helpers ------------------------------------------- #
def _safe_replace_dataclass(base, updates: dict):
    """
    Create a new dataclass instance from `base` with fields updated by `updates`.
    Ignores update keys that the dataclass does not have.
    """
    allowed = {k: v for k, v in updates.items() if hasattr(base, k)}
    if not allowed:
        return base
    try:
        return replace(base, **allowed)
    except TypeError:
        # Defensive fallback: build dict of only fields accepted by replace
        allowed2 = {k: v for k, v in allowed.items() if k in base.__dataclass_fields__}
        return replace(base, **allowed2)

def _apply_trial_to_config(trial: optuna.trial.Trial):
    """
    STAGE 1: Model Quality Optimization

    Constructs AppConfig with:
    - OPTIMIZED: Model architecture, learning dynamics, loss balance, temporal window
    - FIXED: Strategy parameters (threshold, confidence_margin, kelly_clip, batch_size)

    Search Space (13 parameters):
    - Model: hermite_version, degree, maps_a, maps_b, hidden_dim, dropout, lstm_hidden, classification_loss, focal_gamma
    - Training: lr, weight_decay, grad_clip
    - Loss: reg_weight, cls_weight, unc_weight, sign_hinge_weight
    - Data: feature_window
    """
    base = APP_CONFIG

    # ============ MODEL ARCHITECTURE (9 params) ============
    model_updates = {}
    model_updates["hermite_version"] = trial.suggest_categorical("hermite_version", HERMITES_VERSION_CHOICES)
    model_updates["hermite_degree"] = trial.suggest_categorical("hermite_degree", HERMITES)
    model_updates["hermite_maps_a"] = trial.suggest_categorical("hermite_maps_a", HERMITE_MAPS_A)
    model_updates["hermite_maps_b"] = trial.suggest_categorical("hermite_maps_b", HERMITE_MAPS_B)
    model_updates["hermite_hidden_dim"] = trial.suggest_categorical("hermite_hidden_dim", HIDDEN_DIMS)
    model_updates["dropout"] = trial.suggest_float("dropout", 0.01, 0.40, step=0.01)  # 1% precision, 40 steps
    model_updates["lstm_hidden"] = trial.suggest_categorical("lstm_hidden", LSTM_HIDDEN_CHOICES)
    model_updates["classification_loss"] = trial.suggest_categorical("classification_loss", CLASSIFICATION_LOSS_CHOICES)
    model_updates["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 4.0, step=0.1)  # 0.1 precision, 36 steps

    # FIXED architectural choices
    model_updates["use_lstm"] = FIXED_USE_LSTM
    model_updates["prob_source"] = FIXED_PROB_SOURCE

    # ============ LEARNING DYNAMICS (3 params) ============
    training_updates = {}
    training_updates["learning_rate"] = trial.suggest_float("lr", 5e-6, 5e-3, log=True)  # Wider log range
    training_updates["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)  # Include v4 range
    training_updates["gradient_clip"] = trial.suggest_float("grad_clip", 0.5, 5.0, step=0.25)  # 0.25 precision, 19 steps

    # FIXED batch size (proven optimal)
    training_updates["batch_size"] = FIXED_BATCH_SIZE

    # ============ LOSS BALANCE (4 params) - CRITICAL FOR MODEL QUALITY ============
    # Expanded ranges to include v4 optimal values
    training_updates["reg_weight"] = trial.suggest_float("reg_weight", 0.5, 4.0, step=0.1)  # Include 2.95, 36 steps
    training_updates["cls_weight"] = trial.suggest_float("cls_weight", 0.1, 3.0, step=0.1)  # Include 1.40, 30 steps
    training_updates["unc_weight"] = trial.suggest_float("unc_weight", 0.0, 1.0, step=0.05)  # 21 steps
    training_updates["sign_hinge_weight"] = trial.suggest_float("sign_hinge_weight", 0.0, 1.5, step=0.05)  # Include 0.62, 31 steps

    # ============ TEMPORAL WINDOW (1 param) ============
    # Step size 8 provides good coverage: 16, 24, 32, ..., 256 (31 values)
    data_updates = {"feature_window": trial.suggest_int("feature_window", low=16, high=256, step=8)}

    # ============ FIXED STRATEGY PARAMETERS (from v4 best) ============
    evaluation_updates = {
        "threshold": FIXED_THRESHOLD,
        "confidence_margin": FIXED_CONFIDENCE_MARGIN,
        "kelly_clip": FIXED_KELLY_CLIP,
    }

    # Build new config
    new_model = _safe_replace_dataclass(base.model, model_updates)
    new_training = _safe_replace_dataclass(base.training, training_updates)
    new_data = _safe_replace_dataclass(base.data, data_updates)
    new_evaluation = _safe_replace_dataclass(base.evaluation, evaluation_updates)

    new_cfg = _safe_replace_dataclass(base, {"model": new_model, "training": new_training, "data": new_data, "evaluation": new_evaluation})
    return new_cfg

# ----------------------------- Objective ---------------------------------------- #
def objective(trial: optuna.trial.Trial) -> float:
    """
    STAGE 1 OBJECTIVE: Optimize prediction quality (AUC + DirAcc) only.

    Scoring: 70% AUC + 30% DirAcc
    - AUC: Primary metric for probability calibration quality
    - DirAcc: Secondary metric for practical directional prediction

    Sharpe ratio is logged but NOT used in optimization.
    This separates model quality from strategy execution.

    Returns: float in range [0.0, 1.0] (higher is better)
    """
    try:
        cfg = _apply_trial_to_config(trial)
        set_seed(SEED + trial.number)

        trainer = HermiteTrainer(config=cfg)
        artifacts = trainer.run()

        if artifacts is None or not artifacts.fold_results:
            logging.warning(f"Trial {trial.number}: trainer returned None or empty fold_results.")
            return 0.0  # Minimum score for failures

        # Extract and validate metrics across folds
        import math
        auc_values = [fold.metrics.get("AUC", 0.5) for fold in artifacts.fold_results]
        dir_acc_values = [fold.metrics.get("DirAcc", 0.5) for fold in artifacts.fold_results]
        sharpe_values = [fold.metrics.get("Sharpe_strategy", float("nan")) for fold in artifacts.fold_results]

        # Filter out NaN/invalid values
        auc_clean = [v for v in auc_values if not math.isnan(v) and 0.0 <= v <= 1.0]
        dir_acc_clean = [v for v in dir_acc_values if not math.isnan(v) and 0.0 <= v <= 1.0]
        sharpe_clean = [v for v in sharpe_values if not math.isnan(v)]

        # Compute averages (default to baseline if all NaN)
        auc = float(sum(auc_clean) / len(auc_clean)) if auc_clean else 0.5
        dir_acc = float(sum(dir_acc_clean) / len(dir_acc_clean)) if dir_acc_clean else 0.5
        sharpe = float(sum(sharpe_clean) / len(sharpe_clean)) if sharpe_clean else float("nan")

        # STAGE 1 SCORE: Focus on prediction quality
        score = W_AUC * auc + W_DIRACC * dir_acc

        # Log all metrics for analysis (but only optimize on AUC+DirAcc)
        trial.set_user_attr("auc", auc)
        trial.set_user_attr("dir_acc", dir_acc)
        trial.set_user_attr("sharpe", sharpe)  # Logged but not optimized
        trial.set_user_attr("num_valid_auc", len(auc_clean))
        trial.set_user_attr("num_valid_dir_acc", len(dir_acc_clean))

        return float(score)

    except torch.cuda.OutOfMemoryError:
        logging.warning(f"Trial {trial.number}: CUDA OOM → pruned")
        raise optuna.exceptions.TrialPruned()

    except Exception as e:
        logging.error(f"Trial {trial.number} crashed: {type(e).__name__}: {e}")
        trial.set_user_attr("error", str(e))
        return 0.0  # Minimum score for failures

# ----------------------------- Execution ---------------------------------------- #
def has_complete_trial(study: optuna.study.Study) -> bool:
    return any(t.state == TrialState.COMPLETE for t in study.trials)

def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s: %(message)s")
    set_seed(SEED)

    # Professional TPE configuration for 2500 trials
    # n_startup_trials: ~10% of total for initial random exploration = 250 trials
    # This ensures good coverage before TPE modeling kicks in
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=250,         # 10% random exploration (covers parameter space broadly)
        n_ei_candidates=64,           # More candidates for better exploitation/exploration balance
        multivariate=True,            # Enable multivariate TPE for parameter interactions
        constant_liar=True,           # Helps parallel optimization avoid redundancy
        warn_independent_sampling=True  # Alert if fallback to independent sampling
    )

    # No pruner for Stage 1: We want full training metrics for each trial
    # Pruning makes sense for Stage 2 when optimizing strategy parameters

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",  # Maximize AUC + DirAcc score
        sampler=sampler,
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        # Enqueue v4 best trial as starting point
        study.enqueue_trial(INITIAL_PARAMS)
        logging.info("Enqueued v4 best configuration (Sharpe=1.20) as first trial for warm start.")

    attempted = 0
    with tqdm(total=N_TRIALS, desc="Optuna Trials", unit="trial") as pbar:
        while attempted < N_TRIALS:
            # run exactly one trial per loop so we can update the bar and inspect metrics
            study.optimize(objective, n_trials=1)
            attempted += 1
            pbar.update(1)

            if has_complete_trial(study):
                try:
                    best = study.best_trial
                    pbar.set_postfix({
                        "Score": f"{best.value:.4f}" if best.value is not None else "N/A",
                        "AUC": f"{best.user_attrs.get('auc', 0):.4f}",
                        "DirAcc": f"{best.user_attrs.get('dir_acc', 0):.4f}",
                        "Sharpe": f"{best.user_attrs.get('sharpe', float('nan')):.2f}",  # Logged only
                        "Total": len(study.trials),
                    })
                except ValueError:
                    pbar.set_postfix({"Status": "No complete trials yet"})

            # debug counts
            complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
            pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
            failed = sum(1 for t in study.trials if t.state == TrialState.FAIL)
            logging.info(f"After {attempted} attempts: COMPLETE={complete} PRUNED={pruned} FAIL={failed}")

    # Save best config only if we have at least one completed trial
    if has_complete_trial(study):
        best = study.best_trial
        best_yaml = f"""# STAGE 1 BEST CONFIG — Model Quality Optimization
# Score: {best.value:.4f} (70% AUC + 30% DirAcc)
# AUC: {best.user_attrs.get('auc', 0.0):.4f} | DirAcc: {best.user_attrs.get('dir_acc', 0.0):.4f}
# Sharpe (logged): {best.user_attrs.get('sharpe', float('nan')):.4f}
# Trial: {best.number} | Total trials: {len(study.trials)}

model:
  hermite_version: {best.params.get('hermite_version', INITIAL_PARAMS['hermite_version'])}
  hermite_degree: {best.params.get('hermite_degree', INITIAL_PARAMS['hermite_degree'])}
  hermite_maps_a: {best.params.get('hermite_maps_a', INITIAL_PARAMS['hermite_maps_a'])}
  hermite_maps_b: {best.params.get('hermite_maps_b', INITIAL_PARAMS['hermite_maps_b'])}
  hermite_hidden_dim: {best.params.get('hermite_hidden_dim', INITIAL_PARAMS['hermite_hidden_dim'])}
  dropout: {best.params.get('dropout', INITIAL_PARAMS['dropout']):.4f}
  use_lstm: {FIXED_USE_LSTM}
  lstm_hidden: {best.params.get('lstm_hidden', INITIAL_PARAMS['lstm_hidden'])}
  prob_source: {FIXED_PROB_SOURCE}

training:
  learning_rate: {best.params.get('lr', INITIAL_PARAMS['lr']):.2e}
  batch_size: {FIXED_BATCH_SIZE}
  weight_decay: {best.params.get('weight_decay', INITIAL_PARAMS['weight_decay']):.2e}
  gradient_clip: {best.params.get('grad_clip', INITIAL_PARAMS['grad_clip']):.2f}
  reg_weight: {best.params.get('reg_weight', INITIAL_PARAMS['reg_weight']):.4f}
  cls_weight: {best.params.get('cls_weight', INITIAL_PARAMS['cls_weight']):.4f}
  unc_weight: {best.params.get('unc_weight', INITIAL_PARAMS['unc_weight']):.4f}
  sign_hinge_weight: {best.params.get('sign_hinge_weight', INITIAL_PARAMS['sign_hinge_weight']):.4f}
  classification_loss: {best.params.get('classification_loss', INITIAL_PARAMS['classification_loss'])}
  focal_gamma: {best.params.get('focal_gamma', INITIAL_PARAMS['focal_gamma']):.2f}

data:
  feature_window: {best.params.get('feature_window', INITIAL_PARAMS['feature_window'])}

evaluation:
  # FIXED from v4 best (Sharpe=1.20) - optimize in Stage 2
  threshold: {FIXED_THRESHOLD:.6f}
  confidence_margin: {FIXED_CONFIDENCE_MARGIN:.6f}
  kelly_clip: {FIXED_KELLY_CLIP:.6f}
"""
        output_path = Path("best_config_optuna_v6_stage1.yaml")
        output_path.write_text(best_yaml)
        print(f"\n✓ Best Stage 1 config saved → {output_path}")
        print(f"  Score: {best.value:.4f} | AUC: {best.user_attrs.get('auc', 0):.4f} | DirAcc: {best.user_attrs.get('dir_acc', 0):.4f}")
    else:
        print("\n✗ No completed trials — check logs for errors.")

    print(f"\nOptuna dashboard: optuna-dashboard {STORAGE}")

if __name__ == "__main__":
    main()

