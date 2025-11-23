# Optuna Integration Readiness Guide

This document describes how to activate and integrate Optuna for hyperparameter optimization (HPO) with the He_NN trading desktop application.

---

## Current Status

**Status**: **Skeleton Ready, Not Enabled**

The HPO adapter skeleton is implemented but Optuna integration is **disabled by default**. The current implementation provides:

✅ `HPOAdapter` class skeleton in `backend/hpo/hpo_adapter.py`
✅ SQLite storage configuration in `config/app.yaml`
✅ `optuna.enabled` flag (default: `false`)
✅ Placeholder methods for trial runs and hyperparameter suggestions

**Not Implemented Yet**:
- ❌ Optuna study creation and trial loop
- ❌ REST API endpoint `/hpo/suggest` handler in `backend/app.py`
- ❌ Desktop GUI HPO control panel
- ❌ Hyperparameter search space definition
- ❌ Trial pruning logic
- ❌ Multi-objective optimization (Pareto front)

---

## Why Optuna?

Optuna is a powerful hyperparameter optimization framework with:
- **Tree-structured Parzen Estimator (TPE)** for efficient search
- **Pruning** for early stopping of unpromising trials
- **Multi-objective** optimization support
- **Parallel trials** via SQLite or PostgreSQL storage
- **Visualization** tools for trial history and parameter importance

---

## Activation Steps

### Step 1: Install Optuna

```bash
# FIX: Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# FIX: Install Optuna
pip install optuna optuna-dashboard plotly
```

### Step 2: Enable Optuna in Config

Edit `config/app.yaml`:

```yaml
optuna:
  enabled: true  # FIX: enable Optuna integration
  storage_url: "sqlite:///storage/optuna_studies.db"  # FIX: SQLite storage URL
  n_trials: 100  # FIX: number of trials per study (optional)
  timeout: 3600  # FIX: timeout in seconds (optional)
```

**Storage Options**:
- **SQLite** (default): `sqlite:///storage/optuna_studies.db`
- **PostgreSQL**: `postgresql://user:password@localhost/optuna_db`
- **MySQL**: `mysql://user:password@localhost/optuna_db`

For parallel trials, use PostgreSQL or MySQL instead of SQLite.

### Step 3: Implement HPO Adapter

Update `backend/hpo/hpo_adapter.py` with actual Optuna integration:

```python
"""HPO adapter for Optuna hyperparameter optimization."""  # FIX: updated docstring

import logging  # FIX: logging
from pathlib import Path  # FIX: path handling
from typing import Any, Dict, Optional  # FIX: type annotations

import optuna  # FIX: import Optuna
from optuna.pruners import MedianPruner  # FIX: import pruner
from optuna.samplers import TPESampler  # FIX: import sampler

from src.config import AppConfig, load_config  # FIX: import config
from src.pipeline.training import HermiteTrainer  # FIX: import trainer

LOGGER = logging.getLogger(__name__)  # FIX: module logger


class HPOAdapter:
    """HPO adapter for Optuna hyperparameter optimization."""  # FIX: class docstring

    def __init__(
        self,
        storage_url: str = "sqlite:///storage/optuna_studies.db",  # FIX: storage URL
        study_name: str = "hermite_hpo",  # FIX: study name
    ) -> None:
        """Initialize HPO adapter."""  # FIX: constructor docstring
        self.storage_url = storage_url  # FIX: store storage URL
        self.study_name = study_name  # FIX: store study name

        # FIX: Create or load Optuna study
        self.study = optuna.create_study(
            study_name=study_name,  # FIX: study name
            storage=storage_url,  # FIX: storage URL
            load_if_exists=True,  # FIX: load existing study if available
            direction="maximize",  # FIX: maximize AUC (primary objective)
            sampler=TPESampler(seed=42),  # FIX: TPE sampler with seed
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),  # FIX: median pruner
        )  # FIX: create study

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna trial."""  # FIX: objective function docstring

        # FIX: Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)  # FIX: suggest LR
        batch_size = trial.suggest_int("batch_size", 64, 512, step=64)  # FIX: suggest batch size
        hermite_degree = trial.suggest_int("hermite_degree", 3, 10)  # FIX: suggest Hermite degree
        hermite_maps_a = trial.suggest_int("hermite_maps_a", 1, 4)  # FIX: suggest maps A
        hermite_maps_b = trial.suggest_int("hermite_maps_b", 1, 4)  # FIX: suggest maps B
        dropout = trial.suggest_float("dropout", 0.0, 0.5)  # FIX: suggest dropout

        # FIX: Create config with suggested hyperparameters
        # FIX: (Implementation depends on how to override config dataclasses)
        # FIX: For now, using default config (skeleton)
        config = load_config()  # FIX: load default config

        # FIX: Run training with suggested hyperparameters
        trainer = HermiteTrainer(config=config)  # FIX: create trainer
        dataset = trainer.prepare_dataset()  # FIX: prepare dataset
        artifacts = trainer.run(dataset=dataset, use_cv=True, results_dir=Path("reports/hpo"))  # FIX: run training

        # FIX: Extract validation AUC as objective metric
        fold_results = artifacts.fold_results  # FIX: get fold results
        auc_values = [fold.metrics.get("AUC", 0.0) for fold in fold_results]  # FIX: extract AUC values
        mean_auc = sum(auc_values) / len(auc_values) if auc_values else 0.0  # FIX: compute mean AUC

        # FIX: Optionally prune trial based on intermediate results
        # FIX: (Advanced: report intermediate values and use pruner)

        return mean_auc  # FIX: return objective value (maximize AUC)

    def optimize(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization."""  # FIX: optimize method docstring

        LOGGER.info(f"Starting Optuna optimization: {n_trials} trials, timeout={timeout}s")  # FIX: log start

        # FIX: Run optimization
        self.study.optimize(
            self.objective,  # FIX: objective function
            n_trials=n_trials,  # FIX: number of trials
            timeout=timeout,  # FIX: timeout in seconds
            show_progress_bar=True,  # FIX: show progress bar
        )  # FIX: optimize study

        # FIX: Get best trial
        best_trial = self.study.best_trial  # FIX: best trial

        LOGGER.info(
            f"Best trial: {best_trial.number}, AUC={best_trial.value:.4f}, params={best_trial.params}"
        )  # FIX: log best trial

        return {
            "best_trial_number": best_trial.number,  # FIX: best trial number
            "best_auc": best_trial.value,  # FIX: best AUC
            "best_params": best_trial.params,  # FIX: best hyperparameters
        }  # FIX: return best trial info
```

### Step 4: Add REST API Endpoint

Add `/hpo/suggest` endpoint to `backend/app.py`:

```python
# FIX: Import HPOAdapter at top of backend/app.py
from backend.hpo.hpo_adapter import HPOAdapter

# FIX: Add endpoint after existing endpoints

@app.post("/hpo/optimize")  # FIX: HPO optimize endpoint
async def hpo_optimize(n_trials: int = 100, timeout: Optional[int] = None) -> JSONResponse:
    """Start hyperparameter optimization."""  # FIX: endpoint docstring

    if not APP_CONFIG.optuna.enabled:  # FIX: check if Optuna enabled
        raise HTTPException(status_code=400, detail="Optuna is not enabled in config")  # FIX: error response

    # FIX: Create HPO adapter
    hpo_adapter = HPOAdapter(storage_url=APP_CONFIG.optuna.storage_url)  # FIX: create adapter

    # FIX: Run optimization in background (simplified - should use background task)
    result = hpo_adapter.optimize(n_trials=n_trials, timeout=timeout)  # FIX: run optimization

    return JSONResponse(content=result, status_code=200)  # FIX: return results


@app.get("/hpo/best_trial")  # FIX: get best trial endpoint
async def hpo_get_best_trial() -> JSONResponse:
    """Get best trial from current study."""  # FIX: endpoint docstring

    if not APP_CONFIG.optuna.enabled:  # FIX: check if Optuna enabled
        raise HTTPException(status_code=400, detail="Optuna is not enabled in config")  # FIX: error response

    hpo_adapter = HPOAdapter(storage_url=APP_CONFIG.optuna.storage_url)  # FIX: create adapter

    best_trial = hpo_adapter.study.best_trial  # FIX: get best trial

    return JSONResponse(
        content={
            "trial_number": best_trial.number,  # FIX: trial number
            "value": best_trial.value,  # FIX: objective value
            "params": best_trial.params,  # FIX: hyperparameters
        },
        status_code=200,  # FIX: HTTP 200
    )
```

### Step 5: Test Optuna Integration

```bash
# FIX: Start backend with Optuna enabled
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

# FIX: Trigger HPO optimization via REST API
curl -X POST "http://localhost:8000/hpo/optimize?n_trials=10&timeout=600"

# FIX: Get best trial results
curl http://localhost:8000/hpo/best_trial
```

---

## Hyperparameter Search Space

Define the hyperparameter search space in `HPOAdapter.objective()`:

### Model Architecture
- `hermite_degree`: [3, 10] (discrete)
- `hermite_maps_a`: [1, 4] (discrete)
- `hermite_maps_b`: [1, 4] (discrete)
- `hermite_hidden_dim`: [32, 128] (discrete, step=16)
- `dropout`: [0.0, 0.5] (continuous)
- `use_lstm`: [True, False] (categorical)
- `lstm_hidden`: [32, 128] (discrete, step=16)

### Training
- `learning_rate`: [1e-4, 1e-2] (log-uniform)
- `batch_size`: [64, 512] (discrete, step=64)
- `weight_decay`: [1e-6, 1e-3] (log-uniform)
- `gradient_clip`: [0.5, 5.0] (continuous)

### Loss Weights
- `reg_weight`: [0.5, 2.0] (continuous)
- `cls_weight`: [0.5, 2.0] (continuous)
- `unc_weight`: [0.0, 1.0] (continuous)
- `sign_hinge_weight`: [0.0, 0.1] (continuous)

### Evaluation
- `calibration_method`: ["abs", "std_gauss"] (categorical)
- `p_up_source`: ["cdf", "logit"] (categorical)

---

## Multi-Objective Optimization

To optimize multiple objectives (e.g., maximize AUC and minimize ECE):

```python
# FIX: Create multi-objective study
self.study = optuna.create_study(
    study_name=study_name,
    storage=storage_url,
    load_if_exists=True,
    directions=["maximize", "minimize"],  # FIX: maximize AUC, minimize ECE
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
)

# FIX: Return tuple of objectives from objective function
def objective(self, trial: optuna.Trial) -> Tuple[float, float]:
    # ... training code ...
    mean_auc = ...  # FIX: compute mean AUC
    mean_ece = ...  # FIX: compute mean ECE
    return (mean_auc, mean_ece)  # FIX: return both objectives
```

To retrieve Pareto-optimal trials:

```python
# FIX: Get Pareto-optimal trials
pareto_trials = self.study.best_trials  # FIX: Pareto front
for trial in pareto_trials:
    print(f"Trial {trial.number}: AUC={trial.values[0]:.4f}, ECE={trial.values[1]:.4f}")
```

---

## Pruning and Early Stopping

Optuna supports pruning unpromising trials early:

```python
# FIX: Report intermediate values during training
for epoch in range(num_epochs):
    # ... training code ...
    epoch_auc = ...  # FIX: compute epoch AUC

    # FIX: Report intermediate value to Optuna
    trial.report(epoch_auc, epoch)

    # FIX: Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()  # FIX: prune trial
```

---

## Visualization with Optuna Dashboard

Launch interactive Optuna Dashboard:

```bash
# FIX: Install Optuna Dashboard
pip install optuna-dashboard

# FIX: Launch dashboard
optuna-dashboard sqlite:///storage/optuna_studies.db

# FIX: Open browser at http://localhost:8080
```

Dashboard features:
- Trial history plot
- Parameter importance plot
- Optimization history
- Parallel coordinate plot
- Slice plot for parameter effects

---

## Parallel Trials

To run trials in parallel, use PostgreSQL storage and launch multiple workers:

```bash
# FIX: Terminal 1 - Worker 1
python -m backend.hpo.hpo_adapter --study-name hermite_hpo --n-trials 50

# FIX: Terminal 2 - Worker 2
python -m backend.hpo.hpo_adapter --study-name hermite_hpo --n-trials 50

# FIX: Terminal 3 - Worker 3
python -m backend.hpo.hpo_adapter --study-name hermite_hpo --n-trials 50
```

All workers will share the same study via PostgreSQL storage and coordinate trial selection.

---

## Best Practices

1. **Start with Coarse Search**: Use wider ranges and fewer trials (e.g., 20 trials) to identify promising regions
2. **Refine Search Space**: Narrow ranges based on initial results and run more trials (e.g., 100+)
3. **Use Pruning**: Enable `MedianPruner` to stop unpromising trials early and save compute time
4. **Log Everything**: Store trial metadata, logs, and artifacts for reproducibility
5. **Validate on Hold-Out Set**: After HPO, validate best trial on a separate hold-out test set
6. **Monitor Overfitting**: Check if best trial performance drops significantly on hold-out set
7. **Multi-Objective**: Consider optimizing for multiple objectives (AUC, ECE, conformal coverage) to balance calibration and discrimination

---

## Integration with Desktop GUI

To add HPO controls to the desktop GUI:

1. **Add HPO Panel** to `ui/desktop/src/mainwindow.cpp`:
   - Add "HPO" tab or dockable panel
   - Add "Start HPO" button
   - Add trial history table
   - Add best trial display

2. **Implement HPO Trigger**:
   - Send HTTP POST to `/hpo/optimize` on button click
   - Poll `/hpo/best_trial` periodically to update display

3. **Display Trial History**:
   - Query Optuna study via API
   - Display trial number, AUC, hyperparameters in table
   - Highlight Pareto-optimal trials (multi-objective)

4. **Visualize Results**:
   - Embed Optuna Dashboard in WebView widget
   - Or implement custom Qt Charts visualization

---

## Storage Migration

If you need to migrate from SQLite to PostgreSQL:

```python
import optuna

# FIX: Load study from SQLite
source_storage = "sqlite:///storage/optuna_studies.db"
source_study = optuna.load_study(study_name="hermite_hpo", storage=source_storage)

# FIX: Copy to PostgreSQL
target_storage = "postgresql://user:password@localhost/optuna_db"
optuna.copy_study(from_study_name="hermite_hpo", from_storage=source_storage, to_storage=target_storage)
```

---

## References

- **Optuna Documentation**: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
- **Optuna Dashboard**: [https://github.com/optuna/optuna-dashboard](https://github.com/optuna/optuna-dashboard)
- **TPE Paper**: [Algorithms for Hyper-Parameter Optimization (Bergstra et al., 2011)](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
- **Optuna Examples**: [https://github.com/optuna/optuna-examples](https://github.com/optuna/optuna-examples)

---

**Last Updated**: 2025-01-23
