# Optuna Hyperparameter Optimization Guide

## Overview

This repository includes two comprehensive Optuna optimization scripts:

1. **optuna_optimize.py** - 25 parameters (basic)
2. **optuna_optimize_v2.py** - 31 parameters (comprehensive with strategy)

## optuna_optimize_v2.py - RECOMMENDED

### 31 Hyperparameters Optimized

#### Model Architecture (9 params)
- `hermite_version`: ['probabilist', 'physicist']
- `hermite_degree`: [3-7]
- `hermite_maps_a`: [1-8]
- `hermite_maps_b`: [1-8]
- `hermite_hidden_dim`: [32, 64, 128, 256]
- `dropout`: [0.0-0.5]
- `use_lstm`: [True, False]
- `lstm_hidden`: [32, 48, 64, 128]
- `prob_source`: ['cdf', 'logit']

#### Training Dynamics (13 params)
- `learning_rate`: [1e-5 to 1e-2, log scale]
- `batch_size`: [32, 64, 128, 256, 512]
- `reg_weight`: [0.5-2.0]
- `cls_weight`: [0.1-5.0] ⚡ Extended range!
- `unc_weight`: [0.0-1.0]
- `sign_hinge_weight`: [0.0-1.0] ⚡ Extended range!
- `weight_decay`: [1e-6 to 1e-2, log scale]
- `gradient_clip`: [0.5-5.0]
- `classification_loss`: ['bce', 'focal']
- `focal_gamma`: [1.0-3.0]
- `optimizer`: ['adam', 'adamw']
- `scheduler`: ['onecycle', 'cosine', 'none']
- `scheduler_warmup_pct`: [0.05-0.25]

#### Data Configuration (2 params)
- `feature_window`: [32, 64, 96, 128]
- `forecast_horizon`: [1-3]

#### Strategy Execution (7 params) ⚡ NEW!
- `threshold`: [0.50-0.65]
- `confidence_margin`: [0.02-0.10]
- `kelly_clip`: [0.5-1.0]
- `conformal_p_min`: [0.05-0.15]
- `use_kelly_position`: [True, False]
- `use_confidence_margin`: [True, False]
- `use_conformal_filter`: [True, False]

## Installation

```bash
# Activate conda environment
conda activate binance_env

# Install Optuna dependencies
pip install -r requirements_optuna.txt

# Or install manually
pip install optuna>=3.5.0 optuna-dashboard>=0.15.0 plotly>=5.18.0
```

## Running Optimization

### Basic Run (150 trials, 6 hours)

```bash
# Activate conda environment
conda activate binance_env

# Run optimization
python optuna_optimize_v2.py
```

### Monitor Progress (Separate Terminal)

```bash
# Launch interactive dashboard
optuna-dashboard sqlite:///optuna_hermite_v2.db

# Open browser to: http://localhost:8080
```

### Resume Interrupted Study

The script automatically resumes from the database if it exists:

```bash
# Just run again - it will continue where it left off
python optuna_optimize_v2.py
```

### Customize Run Parameters

Edit `optuna_optimize_v2.py` line 228-231:

```python
study.optimize(
    objective,
    n_trials=150,  # Change: More trials = better results
    timeout=21600,  # Change: 6 hours = 21600 seconds
    show_progress_bar=True,
    n_jobs=1  # Change: Use >1 if multiple GPUs available
)
```

## Understanding Results

### Composite Score

The optimization maximizes a weighted composite:

```
Score = 0.50 × Sharpe + 0.25 × AUC + 0.15 × DirAcc + 0.10 × (1 - Brier)
```

**Why this weighting?**
- **50% Sharpe**: Primary goal is trading performance
- **25% AUC**: Calibration quality is critical
- **15% DirAcc**: Directional prediction matters
- **10% Brier**: Probabilistic sharpness

### Output Files

After optimization completes:

1. **optuna_hermite_v2.db** - SQLite database with all trial history
2. **best_config_optuna_v2.yaml** - Best hyperparameters in YAML format
3. **optuna_runs/** - Individual trial artifacts

### Best Config Format

```yaml
# Best hyperparameters from Optuna v2 (31 params)
# Composite Score: 1.2345
# Sharpe: 0.8500, AUC: 0.7800, DirAcc: 0.5600, Brier: 0.1200

model:
  hermite_version: physicist
  hermite_degree: 5
  hermite_maps_a: 4
  hermite_maps_b: 3
  # ... etc

training:
  learning_rate: 0.0012
  batch_size: 128
  # ... etc

data:
  feature_window: 96
  forecast_horizon: 2

strategy:
  threshold: 0.58
  confidence_margin: 0.05
  # ... etc
```

## Using Best Config

### Final Training Run

```bash
# Use best hyperparameters with full 5-fold CV
python main.py --config best_config_optuna_v2.yaml --cv

# Or single fold for quick test
python main.py --config best_config_optuna_v2.yaml
```

### Override Specific Params

```bash
# Use best config but change seed
python main.py --config best_config_optuna_v2.yaml --seed 123

# Use best config but force different threshold
python main.py --config best_config_optuna_v2.yaml --threshold 0.60
```

## Analyzing Studies

### Load Study in Python

```python
import optuna

# Load completed study
study = optuna.load_study(
    study_name='hermite_v2_comprehensive',
    storage='sqlite:///optuna_hermite_v2.db'
)

# Best trial info
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
print(f"Best trial number: {study.best_trial.number}")

# Access user attributes (individual metrics)
print(f"Sharpe: {study.best_trial.user_attrs['sharpe']}")
print(f"AUC: {study.best_trial.user_attrs['auc']}")
```

### Visualizations

```python
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)

study = optuna.load_study('hermite_v2_comprehensive', 'sqlite:///optuna_hermite_v2.db')

# Optimization history over time
plot_optimization_history(study).show()

# Parameter importance ranking
plot_param_importances(study).show()

# Parallel coordinate plot
plot_parallel_coordinate(study).show()

# Slice plot for specific parameter
plot_slice(study, params=['learning_rate', 'batch_size']).show()
```

### Export All Trials to CSV

```python
import optuna
import pandas as pd

study = optuna.load_study('hermite_v2_comprehensive', 'sqlite:///optuna_hermite_v2.db')

# Convert to dataframe
df = study.trials_dataframe()
df.to_csv('optuna_trials.csv', index=False)

# Show top 10 trials
print(df.nlargest(10, 'value')[['number', 'value', 'state']])
```

## Advanced Features

### Pruning

The script includes a MedianPruner that stops unpromising trials early:

```python
pruner=optuna.pruners.MedianPruner(
    n_startup_trials=10,  # No pruning for first 10 trials
    n_warmup_steps=5      # Wait 5 steps before pruning
)
```

### Multi-objective Optimization

To optimize multiple metrics separately (instead of composite):

```python
# In main() function, change:
study = optuna.create_study(
    directions=['maximize', 'maximize'],  # Two objectives
    study_name='hermite_multiobjective',
    # ... rest same
)

# In objective() function, return tuple:
return (sharpe, auc)  # Instead of composite score
```

### Parallel Execution

If you have multiple GPUs:

```python
# In study.optimize(), change:
study.optimize(
    objective,
    n_trials=150,
    timeout=21600,
    n_jobs=4  # Run 4 trials in parallel
)
```

**Note**: Each trial will use 1 GPU. Ensure `CUDA_VISIBLE_DEVICES` is set appropriately.

## Troubleshooting

### "ModuleNotFoundError: No module named 'optuna'"

```bash
# Install in correct environment
conda activate binance_env
pip install optuna optuna-dashboard
```

### "Trial failed" repeatedly

- Check that your data files exist at the configured paths
- Verify that the config in `src/config/` has all required attributes
- Reduce `num_epochs` to 20 for faster debugging
- Check logs in `optuna_runs/` for specific errors

### Database locked errors

```bash
# Stop any running optuna-dashboard processes
pkill -f optuna-dashboard

# Or use a different database name
# Edit line 192 in optuna_optimize_v2.py:
storage='sqlite:///optuna_hermite_v2_new.db'
```

### Out of memory

- Reduce `batch_size` range to [32, 64, 128]
- Reduce `num_epochs` to 30
- Reduce `hermite_hidden_dim` max to 128
- Disable LSTM search: `use_lstm = False` (remove from suggestions)

## Performance Tips

### Speed Up Trials

1. **Reduce epochs**: Line 87, change `num_epochs=50` → `num_epochs=30`
2. **Smaller batch sizes**: Remove 512 from batch_size options
3. **Fewer startup trials**: Line 199, change `n_startup_trials=20` → `10`
4. **Disable CV**: Already disabled (single fold only)

### Improve Quality

1. **More trials**: Line 228, change `n_trials=150` → `300`
2. **Longer timeout**: Line 229, change `timeout=21600` → `43200` (12 hours)
3. **Final validation**: After finding best, train with `num_epochs=200` and full CV

### Balance Speed/Quality

**Quick exploration (2 hours, ~40 trials)**:
```python
num_epochs=30
n_trials=50
timeout=7200
```

**Standard run (6 hours, ~150 trials)**:
```python
num_epochs=50
n_trials=150
timeout=21600
```

**Deep search (24 hours, ~500 trials)**:
```python
num_epochs=80
n_trials=500
timeout=86400
```

## Expected Results

Based on 31-parameter optimization with 150 trials:

- **Typical composite scores**: 0.8 - 1.5
- **Top Sharpe ratios**: 0.6 - 1.2
- **Top AUC values**: 0.70 - 0.85
- **Directional accuracy**: 52% - 58%
- **Brier scores**: 0.10 - 0.20

**Improvement over defaults**: Expect 10-30% boost in Sharpe ratio.

## Next Steps After Optimization

1. **Final training with CV**:
   ```bash
   python main.py --config best_config_optuna_v2.yaml --cv
   ```

2. **Test on different market conditions**:
   - Try different symbols (ETH, BNB, etc.)
   - Try different timeframes (15m, 4h, etc.)

3. **Ensemble strategies**:
   - Train multiple models with top-5 configs
   - Average predictions or use voting

4. **Production deployment**:
   - Use best config in `config/app.yaml`
   - Run backend with optimized settings
   - Monitor live performance

## References

- Optuna Documentation: https://optuna.readthedocs.io/
- Dashboard Guide: https://optuna-dashboard.readthedocs.io/
- TPE Sampler Paper: https://proceedings.mlr.press/v28/bergstra13.pdf

---

**Ready to optimize? Run:**

```bash
python optuna_optimize_v2.py
```

**Monitor at:** http://localhost:8080 (in separate terminal: `optuna-dashboard sqlite:///optuna_hermite_v2.db`)
