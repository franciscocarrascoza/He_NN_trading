# -*- coding: utf-8 -*-
# optuna_optimize_v6_stage2.py
"""
STAGE 2: STRATEGY PARAMETER OPTIMIZATION (Two-Stage Strategy)

⚠️ CRITICAL WARNINGS (See STAGE2_CRITICAL_WARNINGS.md):
- Sharpe values may be BIASED due to short validation windows
- High Sharpe (>3) is SUSPICIOUS and requires out-of-sample validation
- This is SIMULATION ONLY - not real trading
- DO NOT trust results without extensive validation

Approach:
- Fix model architecture at Stage 1 best configs
- Optimize strategy execution parameters (threshold, confidence_margin, kelly_clip)
- Test multiple models from Stage 1 (not just #1)
- Log consistency metrics (Sharpe std dev, trade frequency, position sizes)
- Objective: Maximize Sharpe ratio WITH skepticism

Usage:
    /home/francisco/.anaconda3/envs/binance_env/bin/python optuna_optimize_v6_stage2.py

Expected Runtime: ~2-3 hours for 500 trials (100 trials × 5 models)
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
import numpy as np

from src.config import APP_CONFIG
from src.pipeline import HermiteTrainer
from src.utils.utils import set_seed

# ----------------------------- USER CONFIGURATION -------------------------------- #
STUDY_NAME = "hermite_v6_stage2_strategy"
STORAGE = "sqlite:///optuna_hermite_v6_stage2.db"
N_TRIALS_PER_MODEL = 100  # 100 trials per model
SEED = 42

# Load Stage 1 study to get model configurations
STAGE1_STORAGE = "sqlite:///optuna_hermite_v6_stage1.db"
STAGE1_STUDY_NAME = "hermite_v6_stage1_model_quality"

# Top 5 models from Stage 1 (by Score, AUC, Sharpe diversity)
# These were selected from top 10 by each metric, deduplicated
TOP_MODELS = [1410, 356, 664, 632, 1001]

# ⚠️ CONSERVATIVE PARAMETER RANGES (avoid extremes)
THRESHOLD_MIN = 0.55  # Don't go too low (would trade too often)
THRESHOLD_MAX = 0.75  # Don't go too high (would rarely trade)
CONFIDENCE_MARGIN_MIN = 0.0  # Can disable margin
CONFIDENCE_MARGIN_MAX = 0.20  # Don't make too restrictive
KELLY_CLIP_MIN = 0.3  # Don't allow tiny positions (noise)
KELLY_CLIP_MAX = 1.0  # Full Kelly (risky but allowed)

# FIXED from Stage 1 (not optimized in Stage 2)
FIXED_BATCH_SIZE = 512

# ----------------------------- Load Stage 1 Model Configs ----------------------------- #
def load_stage1_model_config(trial_number: int) -> dict:
    """Load model parameters from Stage 1 trial."""
    stage1_study = optuna.load_study(
        study_name=STAGE1_STUDY_NAME,
        storage=STAGE1_STORAGE
    )

    stage1_trial = stage1_study.trials[trial_number]

    if stage1_trial.state != TrialState.COMPLETE:
        raise ValueError(f"Stage 1 trial {trial_number} is not completed")

    # Extract all model parameters from Stage 1
    model_params = {
        # Model architecture
        "hermite_version": stage1_trial.params.get("hermite_version"),
        "hermite_degree": stage1_trial.params.get("hermite_degree"),
        "hermite_maps_a": stage1_trial.params.get("hermite_maps_a"),
        "hermite_maps_b": stage1_trial.params.get("hermite_maps_b"),
        "hermite_hidden_dim": stage1_trial.params.get("hermite_hidden_dim"),
        "dropout": stage1_trial.params.get("dropout"),
        "lstm_hidden": stage1_trial.params.get("lstm_hidden"),
        "classification_loss": stage1_trial.params.get("classification_loss"),
        "focal_gamma": stage1_trial.params.get("focal_gamma"),

        # Learning dynamics
        "lr": stage1_trial.params.get("lr"),
        "weight_decay": stage1_trial.params.get("weight_decay"),
        "grad_clip": stage1_trial.params.get("grad_clip"),

        # Loss balance
        "reg_weight": stage1_trial.params.get("reg_weight"),
        "cls_weight": stage1_trial.params.get("cls_weight"),
        "unc_weight": stage1_trial.params.get("unc_weight"),
        "sign_hinge_weight": stage1_trial.params.get("sign_hinge_weight"),

        # Data/temporal
        "feature_window": stage1_trial.params.get("feature_window"),
    }

    # Also get Stage 1 metrics for logging
    stage1_metrics = {
        "stage1_score": stage1_trial.value,
        "stage1_auc": stage1_trial.user_attrs.get("auc", 0.0),
        "stage1_dir_acc": stage1_trial.user_attrs.get("dir_acc", 0.0),
        "stage1_sharpe": stage1_trial.user_attrs.get("sharpe", float("nan")),
    }

    return {"params": model_params, "metrics": stage1_metrics}

# ----------------------------- Helpers ------------------------------------------- #
def _safe_replace_dataclass(base, updates: dict):
    """Create a new dataclass instance from `base` with fields updated by `updates`."""
    allowed = {k: v for k, v in updates.items() if hasattr(base, k)}
    if not allowed:
        return base
    try:
        return replace(base, **allowed)
    except TypeError:
        allowed2 = {k: v for k, v in allowed.items() if k in base.__dataclass_fields__}
        return replace(base, **allowed2)

def _apply_stage2_config(model_trial_num: int, threshold: float, confidence_margin: float, kelly_clip: float):
    """
    STAGE 2: Build config with fixed model params (from Stage 1) and variable strategy params.
    """
    base = APP_CONFIG

    # Load model config from Stage 1
    stage1_config = load_stage1_model_config(model_trial_num)
    model_params = stage1_config["params"]

    # ============ MODEL PARAMETERS (FIXED from Stage 1) ============
    model_updates = {
        "hermite_version": model_params["hermite_version"],
        "hermite_degree": model_params["hermite_degree"],
        "hermite_maps_a": model_params["hermite_maps_a"],
        "hermite_maps_b": model_params["hermite_maps_b"],
        "hermite_hidden_dim": model_params["hermite_hidden_dim"],
        "dropout": model_params["dropout"],
        "lstm_hidden": model_params["lstm_hidden"],
        "use_lstm": True,  # Always true in Stage 1
        "prob_source": "pdf",  # Always pdf in Stage 1
    }

    # ============ TRAINING PARAMETERS (FIXED from Stage 1) ============
    training_updates = {
        "learning_rate": model_params["lr"],
        "weight_decay": model_params["weight_decay"],
        "gradient_clip": model_params["grad_clip"],
        "reg_weight": model_params["reg_weight"],
        "cls_weight": model_params["cls_weight"],
        "unc_weight": model_params["unc_weight"],
        "sign_hinge_weight": model_params["sign_hinge_weight"],
        "classification_loss": model_params["classification_loss"],
        "focal_gamma": model_params["focal_gamma"],
        "batch_size": FIXED_BATCH_SIZE,
    }

    # ============ DATA PARAMETERS (FIXED from Stage 1) ============
    data_updates = {
        "feature_window": model_params["feature_window"],
    }

    # ============ STRATEGY PARAMETERS (OPTIMIZED in Stage 2) ============
    strategy_updates = {
        "threshold": threshold,
        "confidence_margin": confidence_margin,
        "kelly_clip": kelly_clip,
    }

    # Build new config
    new_model = _safe_replace_dataclass(base.model, model_updates)
    new_training = _safe_replace_dataclass(base.training, training_updates)
    new_data = _safe_replace_dataclass(base.data, data_updates)
    new_strategy = _safe_replace_dataclass(base.strategy, strategy_updates)

    new_cfg = _safe_replace_dataclass(
        base,
        {"model": new_model, "training": new_training, "data": new_data, "strategy": new_strategy}
    )

    return new_cfg, stage1_config["metrics"]

# ----------------------------- Objective ---------------------------------------- #
def objective(trial: optuna.trial.Trial) -> float:
    """
    STAGE 2 OBJECTIVE: Optimize strategy parameters (threshold, confidence_margin, kelly_clip).

    - Cycle through TOP_MODELS (each gets equal number of trials)
    - For each model, optimize strategy params to maximize Sharpe
    - Log consistency metrics (Sharpe std, trade frequency, position sizes)
    - Return mean Sharpe across folds (with extreme skepticism about absolute values)
    """
    try:
        # Determine which model to use (cycle through TOP_MODELS)
        model_idx = trial.number % len(TOP_MODELS)
        model_trial_num = TOP_MODELS[model_idx]

        # Stage 2 strategy parameters (OPTIMIZED)
        threshold = trial.suggest_float("threshold", THRESHOLD_MIN, THRESHOLD_MAX, step=0.01)
        confidence_margin = trial.suggest_float("confidence_margin", CONFIDENCE_MARGIN_MIN, CONFIDENCE_MARGIN_MAX, step=0.01)
        kelly_clip = trial.suggest_float("kelly_clip", KELLY_CLIP_MIN, KELLY_CLIP_MAX, step=0.05)

        # Build config with fixed model + variable strategy
        cfg, stage1_metrics = _apply_stage2_config(model_trial_num, threshold, confidence_margin, kelly_clip)
        set_seed(SEED + trial.number)

        # Train model (same architecture as Stage 1)
        trainer = HermiteTrainer(config=cfg)
        artifacts = trainer.run()

        if artifacts is None or not artifacts.fold_results:
            logging.warning(f"Trial {trial.number}: trainer returned None or empty fold_results.")
            return float("-inf")  # Worst possible score

        # Extract Sharpe across folds
        import math
        sharpe_values = [fold.metrics.get("Sharpe_strategy", float("nan")) for fold in artifacts.fold_results]
        auc_values = [fold.metrics.get("AUC", 0.5) for fold in artifacts.fold_results]
        dir_acc_values = [fold.metrics.get("DirAcc", 0.5) for fold in artifacts.fold_results]

        # Filter valid Sharpe values
        sharpe_clean = [v for v in sharpe_values if not math.isnan(v) and math.isfinite(v)]

        if not sharpe_clean:
            logging.warning(f"Trial {trial.number}: All Sharpe values are NaN")
            # Log why it failed
            trial.set_user_attr("failure_reason", "all_sharpe_nan")
            trial.set_user_attr("num_nan_sharpe", len(sharpe_values))
            return float("-inf")

        # Compute mean and std of Sharpe
        sharpe_mean = float(np.mean(sharpe_clean))
        sharpe_std = float(np.std(sharpe_clean))
        sharpe_min = float(np.min(sharpe_clean))
        sharpe_max = float(np.max(sharpe_clean))

        # Compute other metrics
        auc_mean = float(np.mean([v for v in auc_values if not math.isnan(v)]))
        dir_acc_mean = float(np.mean([v for v in dir_acc_values if not math.isnan(v)]))

        # ⚠️ CRITICAL: Log consistency metrics (not just mean Sharpe)
        trial.set_user_attr("model_trial", model_trial_num)
        trial.set_user_attr("sharpe_mean", sharpe_mean)
        trial.set_user_attr("sharpe_std", sharpe_std)
        trial.set_user_attr("sharpe_min", sharpe_min)
        trial.set_user_attr("sharpe_max", sharpe_max)
        trial.set_user_attr("num_valid_sharpe", len(sharpe_clean))
        trial.set_user_attr("num_nan_sharpe", len(sharpe_values) - len(sharpe_clean))
        trial.set_user_attr("auc", auc_mean)
        trial.set_user_attr("dir_acc", dir_acc_mean)

        # Log Stage 1 metrics for reference
        trial.set_user_attr("stage1_score", stage1_metrics["stage1_score"])
        trial.set_user_attr("stage1_auc", stage1_metrics["stage1_auc"])
        trial.set_user_attr("stage1_sharpe", stage1_metrics["stage1_sharpe"])

        # ⚠️ SANITY CHECKS: Flag suspicious results
        if sharpe_mean > 10.0:
            trial.set_user_attr("warning", "sharpe_extremely_high")
        if sharpe_std > 3.0:
            trial.set_user_attr("warning", "sharpe_high_variance")
        if len(sharpe_clean) < 3:
            trial.set_user_attr("warning", "too_few_valid_folds")

        # Return mean Sharpe (optimizer maximizes this)
        return float(sharpe_mean)

    except torch.cuda.OutOfMemoryError:
        logging.warning(f"Trial {trial.number}: CUDA OOM → pruned")
        raise optuna.exceptions.TrialPruned()

    except Exception as e:
        logging.error(f"Trial {trial.number} crashed: {type(e).__name__}: {e}")
        trial.set_user_attr("error", str(e))
        return float("-inf")

# ----------------------------- Execution ---------------------------------------- #
def has_complete_trial(study: optuna.study.Study) -> bool:
    return any(t.state == TrialState.COMPLETE and t.value > float("-inf") for t in study.trials)

def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s: %(message)s")
    set_seed(SEED)

    # Load Stage 1 study to verify models exist
    try:
        stage1_study = optuna.load_study(
            study_name=STAGE1_STUDY_NAME,
            storage=STAGE1_STORAGE
        )
        logging.info(f"Loaded Stage 1 study with {len(stage1_study.trials)} trials")

        # Verify all TOP_MODELS exist and are complete
        for trial_num in TOP_MODELS:
            trial = stage1_study.trials[trial_num]
            if trial.state != TrialState.COMPLETE:
                raise ValueError(f"Stage 1 trial {trial_num} is not completed (state={trial.state})")
            logging.info(f"  Model #{trial_num}: Score={trial.value:.4f}, AUC={trial.user_attrs.get('auc', 0):.4f}")

    except Exception as e:
        logging.error(f"Failed to load Stage 1 study: {e}")
        logging.error("Make sure Stage 1 optimization is complete and database exists")
        return

    # TPE sampler for Stage 2
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=25,  # 25 random trials (5% of 500 total)
        n_ei_candidates=32,
        multivariate=True,
        constant_liar=True,
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",  # Maximize Sharpe
        sampler=sampler,
        load_if_exists=True,
    )

    total_trials = N_TRIALS_PER_MODEL * len(TOP_MODELS)
    logging.info(f"Stage 2: Testing {len(TOP_MODELS)} models × {N_TRIALS_PER_MODEL} trials = {total_trials} total")
    logging.info(f"Models: {TOP_MODELS}")

    attempted = 0
    with tqdm(total=total_trials, desc="Stage 2 Trials", unit="trial") as pbar:
        while attempted < total_trials:
            study.optimize(objective, n_trials=1)
            attempted += 1
            pbar.update(1)

            if has_complete_trial(study):
                try:
                    best = study.best_trial
                    pbar.set_postfix({
                        "Sharpe": f"{best.value:.2f}",
                        "Model": f"#{best.user_attrs.get('model_trial', '?')}",
                        "Thresh": f"{best.params.get('threshold', 0):.2f}",
                        "Margin": f"{best.params.get('confidence_margin', 0):.2f}",
                        "Total": len(study.trials),
                    })
                except ValueError:
                    pbar.set_postfix({"Status": "No valid trials yet"})

            complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE and t.value > float("-inf"))
            failed = sum(1 for t in study.trials if t.value == float("-inf"))
            logging.info(f"After {attempted} attempts: COMPLETE={complete} FAILED={failed}")

    # ⚠️ CRITICAL: Analyze results with skepticism
    if not has_complete_trial(study):
        print("\n✗ No valid trials completed - check logs for errors.")
        return

    best = study.best_trial

    print("\n" + "=" * 80)
    print("STAGE 2 BEST CONFIGURATION (⚠️ VALIDATION REQUIRED)")
    print("=" * 80)
    print(f"\nModel: Stage 1 Trial #{best.user_attrs.get('model_trial', '?')}")
    print(f"  Stage 1 Score: {best.user_attrs.get('stage1_score', 0):.4f}")
    print(f"  Stage 1 AUC: {best.user_attrs.get('stage1_auc', 0):.4f}")
    print(f"  Stage 1 Sharpe: {best.user_attrs.get('stage1_sharpe', float('nan')):.2f}")

    print(f"\nStrategy Parameters (OPTIMIZED):")
    print(f"  threshold: {best.params.get('threshold', 0):.4f}")
    print(f"  confidence_margin: {best.params.get('confidence_margin', 0):.4f}")
    print(f"  kelly_clip: {best.params.get('kelly_clip', 0):.4f}")

    print(f"\nStage 2 Performance:")
    print(f"  Sharpe (mean): {best.value:.2f}")
    print(f"  Sharpe (std): {best.user_attrs.get('sharpe_std', 0):.2f}")
    print(f"  Sharpe (min): {best.user_attrs.get('sharpe_min', 0):.2f}")
    print(f"  Sharpe (max): {best.user_attrs.get('sharpe_max', 0):.2f}")
    print(f"  Valid folds: {best.user_attrs.get('num_valid_sharpe', 0)}/5")

    # ⚠️ SANITY CHECKS
    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    warnings = []

    if best.value > 5.0:
        warnings.append(f"⚠️  Sharpe={best.value:.2f} is VERY HIGH - likely overfit")

    sharpe_std = best.user_attrs.get('sharpe_std', 0)
    if sharpe_std > 2.0:
        warnings.append(f"⚠️  Sharpe std={sharpe_std:.2f} is HIGH - results unstable")

    num_valid = best.user_attrs.get('num_valid_sharpe', 0)
    if num_valid < 4:
        warnings.append(f"⚠️  Only {num_valid}/5 folds valid - poor robustness")

    if best.params.get('threshold', 0.6) < 0.56:
        warnings.append(f"⚠️  Threshold={best.params.get('threshold'):.2f} very low - may overtrade")

    if best.params.get('threshold', 0.6) > 0.74:
        warnings.append(f"⚠️  Threshold={best.params.get('threshold'):.2f} very high - may undertrade")

    if warnings:
        for warning in warnings:
            print(warning)
        print("\n🚨 OUT-OF-SAMPLE VALIDATION IS MANDATORY")
    else:
        print("✓ Basic sanity checks passed")
        print("⚠️  Still requires out-of-sample validation before deployment")

    # Save best config
    best_yaml = f"""# STAGE 2 BEST CONFIG — Strategy Optimization
# ⚠️ WARNING: Sharpe may be biased/overfit - validate before use
# Model: Stage 1 Trial #{best.user_attrs.get('model_trial', '?')}
# Sharpe: {best.value:.2f} ± {best.user_attrs.get('sharpe_std', 0):.2f}
# Trial: {best.number} | Total trials: {len(study.trials)}

# Stage 1 Model (FIXED)
model:
  # Use parameters from Stage 1 trial #{best.user_attrs.get('model_trial', '?')}
  # See optuna_hermite_v6_stage1.db for full config

# Stage 2 Strategy (OPTIMIZED)
strategy:
  threshold: {best.params.get('threshold', 0.6):.6f}
  confidence_margin: {best.params.get('confidence_margin', 0.1):.6f}
  kelly_clip: {best.params.get('kelly_clip', 0.9):.6f}

# Performance Metrics
performance:
  sharpe_mean: {best.value:.4f}
  sharpe_std: {best.user_attrs.get('sharpe_std', 0):.4f}
  sharpe_min: {best.user_attrs.get('sharpe_min', 0):.4f}
  sharpe_max: {best.user_attrs.get('sharpe_max', 0):.4f}
  valid_folds: {best.user_attrs.get('num_valid_sharpe', 0)}
  auc: {best.user_attrs.get('auc', 0):.4f}
  dir_acc: {best.user_attrs.get('dir_acc', 0):.4f}

# Next Steps
# 1. Run out-of-sample validation (MANDATORY)
# 2. Check results on 2024-2025 data (if available)
# 3. Verify no data leakage
# 4. Paper trade for 30+ days before live deployment
"""

    output_path = Path("best_config_optuna_v6_stage2.yaml")
    output_path.write_text(best_yaml)
    print(f"\n✓ Best Stage 2 config saved → {output_path}")

    print(f"\nOptuna dashboard: optuna-dashboard {STORAGE}")

if __name__ == "__main__":
    main()
