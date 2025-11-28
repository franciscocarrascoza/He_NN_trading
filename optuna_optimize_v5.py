# -*- coding: utf-8 -*-
# optuna_optimize_v5.py
"""
Refactored Optuna optimization script — safe, self-contained, and deterministic:
- Uses static, centralized search-space constants (no dynamic categorical spaces).
- Works with frozen dataclasses by using dataclasses.replace to create new config instances.
- Keeps your INITIAL_PARAMS (strong starting point) and enqueues it when the study is empty.
- Runs true training per trial (one trial at a time) and updates a tqdm progress bar.
- Handles exceptions: CUDA OOM -> TrialPruned; other exceptions -> numeric low score (no None).
- Writes best-config YAML only if at least one COMPLETE trial exists.

Usage:
    python optuna_optimize_v5.py

Adjust paths / imports if your project layout differs.
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
STUDY_NAME = "hermite_v5_study"
STORAGE = "sqlite:///optuna_hermite_v5.db"  # you've removed old DBs as stated
N_TRIALS = 100
SEED = 42

# Composite scoring weights (tune as you like)
W_SHARPE = 0.65
W_AUC = 0.25
W_DIRACC = 0.10

# --- Static/search-space constants (do NOT change these after starting a study) --- #
HERMITES = [4, 5, 6, 7]
HIDDEN_DIMS = [64, 128, 192]
PROB_SOURCE_CHOICES = ["cdf", "pdf"]
BATCH_SIZES = [64, 128, 256, 512]
LSTM_HIDDEN_CHOICES = [32, 48, 64]
HERMITES_VERSION_CHOICES = ["probabilist", "physicist"]
CLASSIFICATION_LOSS_CHOICES = ["bce", "focal"]

# Strong starting point (must use only values from the lists above for categorical fields)
INITIAL_PARAMS = {
    "hermite_version": "probabilist",
    "hermite_degree": 7,
    "hermite_maps_a": 7,
    "hermite_maps_b": 2,
    "hermite_hidden_dim": 128,
    "dropout": 0.11746757583868696,
    "use_lstm": True,
    "lstm_hidden": 48,
    "prob_source": "pdf",
    "lr": 0.00010156437386955351,
    "batch_size": 512,
    "weight_decay": 0.010720290868957611,
    "grad_clip": 4.280794920747706,
    "reg_weight": 0.9474849601800794,
    "cls_weight": 0.3051715258700126,
    "unc_weight": 0.6952484527035384,
    "sign_hinge_weight": 0.201603953699083,
    "classification_loss": "focal",
    "focal_gamma": 2.708186253521049,
    "feature_window": 252,
    "threshold": 0.6391068718152106,
    "confidence_margin": 0.07957676745969516,
    "kelly_clip": 0.895941428621494,
    "use_kelly_position": True,
    "use_confidence_margin": True,
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
    Construct and return a NEW AppConfig-like instance for this trial.
    Does not mutate APP_CONFIG (works with frozen dataclasses).
    All distributions are taken from the static lists defined above.
    """
    base = APP_CONFIG

    # collect model updates (use static choice lists)
    model_updates = {}
    model_updates["hermite_version"] = trial.suggest_categorical("hermite_version", HERMITES_VERSION_CHOICES)
    model_updates["hermite_degree"] = trial.suggest_categorical("hermite_degree", HERMITES)
    model_updates["hermite_maps_a"] = trial.suggest_int("hermite_maps_a", 4, 9)
    model_updates["hermite_maps_b"] = trial.suggest_int("hermite_maps_b", 1, 5)
    model_updates["hermite_hidden_dim"] = trial.suggest_categorical("hermite_hidden_dim", HIDDEN_DIMS)
    model_updates["dropout"] = trial.suggest_float("dropout", 0.05, 0.35)
    model_updates["use_lstm"] = trial.suggest_categorical("use_lstm", [False, True])
    # Always suggest lstm_hidden from the same fixed set (even if use_lstm==False)
    model_updates["lstm_hidden"] = trial.suggest_categorical("lstm_hidden", LSTM_HIDDEN_CHOICES)
    model_updates["prob_source"] = trial.suggest_categorical("prob_source", PROB_SOURCE_CHOICES)

    # training updates
    training_updates = {}
    training_updates["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    training_updates["batch_size"] = int(trial.suggest_categorical("batch_size", BATCH_SIZES))
    training_updates["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    training_updates["gradient_clip"] = trial.suggest_float("grad_clip", 0.1, 5.0)
    training_updates["reg_weight"] = trial.suggest_float("reg_weight", 0.1, 2.0)
    training_updates["cls_weight"] = trial.suggest_float("cls_weight", 0.1, 2.0)
    training_updates["unc_weight"] = trial.suggest_float("unc_weight", 0.0, 1.0)
    training_updates["sign_hinge_weight"] = trial.suggest_float("sign_hinge_weight", 0.0, 0.5)
    training_updates["classification_loss"] = trial.suggest_categorical("classification_loss", CLASSIFICATION_LOSS_CHOICES)
    training_updates["focal_gamma"] = trial.suggest_float("focal_gamma", 0.0, 4.0)

    # data/eval updates
    data_updates = {"feature_window": trial.suggest_int("feature_window", low=16, high=256,step=16)}
    evaluation_updates = {
        "threshold": trial.suggest_float("threshold", 0.45, 0.75),
        "confidence_margin": trial.suggest_float("confidence_margin", 0.0, 0.2),
        "kelly_clip": trial.suggest_float("kelly_clip", 0.1, 1.0),
    }

    # build new nested dataclasses (works for frozen dataclasses)
    new_model = _safe_replace_dataclass(base.model, model_updates)
    new_training = _safe_replace_dataclass(base.training, training_updates)
    new_data = _safe_replace_dataclass(base.data, data_updates)
    new_evaluation = _safe_replace_dataclass(base.evaluation, evaluation_updates)


    # Compose new top-level config
    new_cfg = _safe_replace_dataclass(base, {"model": new_model, "training": new_training, "data": new_data, "evaluation": new_evaluation})
    return new_cfg

# ----------------------------- Objective ---------------------------------------- #
def objective(trial: optuna.trial.Trial) -> float:
    """
    Runs a real training job and returns a composite numeric score.
    Any exception returns a numeric low score (no None).
    CUDA OOM -> prune.
    """
    try:
        cfg = _apply_trial_to_config(trial)

        # per-trial reproducibility
        set_seed(SEED + trial.number)

        # instantiate trainer and run training (replace with your trainer)
        #trainer = HermiteTrainer(config=cfg, use_cv=True, cv_folds=3)
        trainer = HermiteTrainer(config=cfg)
        artifacts = trainer.run()  # expected to return TrainingArtifacts with fold_results

        if artifacts is None or not artifacts.fold_results:
            logging.warning(f"Trial {trial.number}: trainer returned None or empty fold_results; returning fallback low score.")
            return -100.0

        # Extract average metrics across folds, handling NaN values
        import math
        sharpe_values = [fold.metrics.get("Sharpe_strategy", -10.0) for fold in artifacts.fold_results]
        auc_values = [fold.metrics.get("AUC", 0.5) for fold in artifacts.fold_results]
        dir_acc_values = [fold.metrics.get("DirAcc", 0.5) for fold in artifacts.fold_results]
        turnover_values = [fold.metrics.get("Turnover", 0.0) for fold in artifacts.fold_results]

        # Filter out NaN values and compute means
        sharpe_clean = [v for v in sharpe_values if not math.isnan(v)]
        auc_clean = [v for v in auc_values if not math.isnan(v)]
        dir_acc_clean = [v for v in dir_acc_values if not math.isnan(v)]

        sharpe = float(sum(sharpe_clean) / len(sharpe_clean)) if sharpe_clean else -10.0
        auc = float(sum(auc_clean) / len(auc_clean)) if auc_clean else 0.5
        dir_acc = float(sum(dir_acc_clean) / len(dir_acc_clean)) if dir_acc_clean else 0.5
        turnover = float(sum(turnover_values) / len(turnover_values))

        # Penalize trials with NaN sharpe heavily
        if not sharpe_clean or sharpe < -5.0:
            score = -100.0 - trial.number * 0.001
        else:
            score = W_SHARPE * sharpe + W_AUC * auc + W_DIRACC * dir_acc

        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("auc", auc)
        trial.set_user_attr("dir_acc", dir_acc)
        trial.set_user_attr("turnover", turnover)

        return float(score)

    except torch.cuda.OutOfMemoryError:
            print(f"Trial {trial.number} → OOM → pruned")
            raise optuna.exceptions.TrialPruned()

    except Exception as e:
            print(f"T {trial.number} → crashed: {type(e).__name__}: {e}")
            # Return a UNIQUE bad score so TPE can still learn
            # The -trial.number makes every failure different
            bad_score = -1000.0 - trial.number * 0.001
            trial.set_user_attr("error", str(e))
            return bad_score

    except Exception:
        logging.exception(f"Trial {trial.number} failed with exception; returning fallback low score.")
        return -100.0

# ----------------------------- Execution ---------------------------------------- #
def has_complete_trial(study: optuna.study.Study) -> bool:
    return any(t.state == TrialState.COMPLETE for t in study.trials)

def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s: %(message)s")
    set_seed(SEED)

    # Existing code (from your uploaded script) - modify the sampler initialization
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=40,          # Increase for more initial random exploration
        n_ei_candidates=48,          # Samples more candidates per suggestion for broader exploration
        constant_liar=True            # Helps avoid clustering in local minima
    )
    # medium pruner: much less aggressive than defaults — tune as needed
    #pruner = optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=15, interval_steps=2)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=sampler,
        #pruner=pruner,
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        # Enqueue the known-good configuration. Ensure its categorical values are valid.
        study.enqueue_trial(INITIAL_PARAMS)
        logging.info("Enqueued strong initial configuration as the first trial.")

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
                        "Best": f"{best.value:.3f}" if best.value is not None else "N/A",
                        "Sharpe": f"{best.user_attrs.get('sharpe', -10):.2f}",
                        "AUC": f"{best.user_attrs.get('auc', 0):.3f}",
                        "Trials": len(study.trials),
                    })
                except ValueError:
                    pbar.set_postfix({"Best": "N/A"})
            else:
                pbar.set_postfix({"Best": "N/A", "Trials": len(study.trials)})

            # debug counts
            complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
            pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
            failed = sum(1 for t in study.trials if t.state == TrialState.FAIL)
            logging.info(f"After {attempted} attempts: COMPLETE={complete} PRUNED={pruned} FAIL={failed}")

    # Save best config only if we have at least one completed trial
    if has_complete_trial(study):
        best = study.best_trial
        best_yaml = f"""# BEST CONFIG — Score: {best.value:.3f}
model:
  hermite_version: {best.params.get('hermite_version', INITIAL_PARAMS['hermite_version'])}
  hermite_degree: {best.params.get('hermite_degree', INITIAL_PARAMS['hermite_degree'])}
  hermite_maps_a: {best.params.get('hermite_maps_a', INITIAL_PARAMS['hermite_maps_a'])}
  hermite_maps_b: {best.params.get('hermite_maps_b', INITIAL_PARAMS['hermite_maps_b'])}
  hermite_hidden_dim: {best.params.get('hermite_hidden_dim', INITIAL_PARAMS['hermite_hidden_dim'])}
  dropout: {best.params.get('dropout', INITIAL_PARAMS['dropout']):.3f}
  use_lstm: {best.params.get('use_lstm', INITIAL_PARAMS['use_lstm'])}
  lstm_hidden: {best.params.get('lstm_hidden', INITIAL_PARAMS['lstm_hidden'])}
  prob_source: {best.params.get('prob_source', INITIAL_PARAMS['prob_source'])}

training:
  lr: {best.params.get('lr', INITIAL_PARAMS['lr']):.2e}
  batch_size: {best.params.get('batch_size', INITIAL_PARAMS['batch_size'])}
  reg_weight: {best.params.get('reg_weight', INITIAL_PARAMS['reg_weight']):.3f}
  cls_weight: {best.params.get('cls_weight', INITIAL_PARAMS['cls_weight']):.3f}
  unc_weight: {best.params.get('unc_weight', INITIAL_PARAMS['unc_weight']):.3f}
  sign_hinge_weight: {best.params.get('sign_hinge_weight', INITIAL_PARAMS['sign_hinge_weight']):.3f}
  weight_decay: {best.params.get('weight_decay', INITIAL_PARAMS['weight_decay']):.2e}
  grad_clip: {best.params.get('grad_clip', INITIAL_PARAMS['grad_clip']):.2f}
  classification_loss: {best.params.get('classification_loss', INITIAL_PARAMS['classification_loss'])}
  focal_gamma: {best.params.get('focal_gamma', INITIAL_PARAMS['focal_gamma']):.3f}

data:
  feature_window: {best.params.get('feature_window', INITIAL_PARAMS['feature_window'])}

evaluation:
  threshold: {best.params.get('threshold', INITIAL_PARAMS['threshold']):.3f}
  confidence_margin: {best.params.get('confidence_margin', INITIAL_PARAMS['confidence_margin']):.3f}
  kelly_clip: {best.params.get('kelly_clip', INITIAL_PARAMS['kelly_clip']):.3f}
"""
        Path("best_config_optuna_v5.yaml").write_text(best_yaml)
        print("\nBest config saved → best_config_optuna_v5.yaml")
    else:
        print("\nNo completed trials — check trainer logs or adjust pruner settings.")

    print("Optuna dashboard (if desired): optuna-dashboard sqlite:///" + Path(STORAGE.replace("sqlite:///", "")).name)

if __name__ == "__main__":
    main()

