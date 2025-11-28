# Stage 2 Optimization - Quick Start Guide

## Overview

**Status**: Ready to run Stage 2 (strategy parameter optimization)

**What it does**: Optimizes `threshold`, `confidence_margin`, and `kelly_clip` for the top 5 models from Stage 1

**Runtime**: ~2-3 hours (500 trials: 100 per model × 5 models)

---

## Files Created

### 1. **`optuna_optimize_v6_stage2.py`**
   - Main optimization script
   - Tests 5 best models from Stage 1
   - Optimizes strategy parameters only (model fixed)
   - Includes sanity checks and warnings

### 2. **`STAGE2_CRITICAL_WARNINGS.md`**
   - YOUR concerns about Sharpe bias/overfitting
   - Validation requirements
   - Expected realistic outcomes
   - Production deployment checklist

### 3. **`stage1_top_models_for_stage2.csv`**
   - Top 25 models from Stage 1
   - Ranked by Score, AUC, and Sharpe
   - Top 5 selected for Stage 2: **[1410, 356, 664, 632, 1001]**

---

## How to Run

### Step 1: Review Top Models (Optional)

```bash
cat stage1_top_models_for_stage2.csv
```

Check if you want to modify the `TOP_MODELS` list in `optuna_optimize_v6_stage2.py`.

### Step 2: Run Stage 2 Optimization

```bash
# Foreground (watch progress)
/home/francisco/.anaconda3/envs/binance_env/bin/python optuna_optimize_v6_stage2.py

# OR background (recommended)
nohup /home/francisco/.anaconda3/envs/binance_env/bin/python optuna_optimize_v6_stage2.py > optuna_v6_stage2.log 2>&1 &

# Monitor progress
tail -f optuna_v6_stage2.log
```

### Step 3: Check Progress

```bash
/home/francisco/.anaconda3/envs/binance_env/bin/python -c "
import optuna
study = optuna.load_study(
    study_name='hermite_v6_stage2_strategy',
    storage='sqlite:///optuna_hermite_v6_stage2.db'
)
completed = [t for t in study.trials if t.state.name=='COMPLETE']
print(f'Progress: {len(completed)}/500 trials ({len(completed)/500*100:.1f}%)')
if study.best_trial:
    print(f'Best Sharpe: {study.best_trial.value:.2f}')
    print(f'Best Model: #{study.best_trial.user_attrs.get(\"model_trial\", \"?\")}')
    print(f'Best Threshold: {study.best_trial.params.get(\"threshold\", 0):.4f}')
"
```

---

## What to Expect

### During Optimization

**Progress bar shows:**
- Sharpe: Current best Sharpe ratio (⚠️ may be biased!)
- Model: Which Stage 1 model trial number
- Thresh: Best threshold found so far
- Margin: Best confidence_margin
- Total: Total trials completed

**Example output:**
```
Stage 2 Trials:  20%|██        | 100/500 [15:23<1:01:32, 9.23s/trial, Sharpe=4.52, Model=#1410, Thresh=0.63, Margin=0.08, Total=100]
```

### After Completion

**Script will print:**
1. Best model + strategy configuration
2. Sharpe statistics (mean, std, min, max)
3. **Sanity check warnings** if:
   - Sharpe > 5.0 (very suspicious)
   - Sharpe std > 2.0 (unstable)
   - < 4 valid folds
   - Extreme threshold values

**Files generated:**
- `best_config_optuna_v6_stage2.yaml` - Best configuration
- `optuna_hermite_v6_stage2.db` - All trial results

---

## Configuration Details

### Top 5 Models Selected (from Stage 1)

| Trial | Score | AUC   | DirAcc | Sharpe | Why Selected |
|-------|-------|-------|--------|--------|--------------|
| 1410  | 0.587 | 0.599 | 0.558  | 3.64   | Best Score, Best AUC |
| 356   | 0.583 | 0.599 | 0.544  | 7.32   | Top AUC, High Sharpe |
| 664   | 0.581 | 0.588 | 0.564  | 6.78   | High DirAcc |
| 632   | 0.579 | 0.592 | 0.550  | 0.28   | Top AUC, Low Sharpe (diversity) |
| 1001  | 0.578 | 0.584 | 0.564  | NaN    | High DirAcc |

**Diversity rationale**: Including model 632 (low Sharpe) and 1001 (NaN Sharpe) ensures we test models with different characteristics, not just high-Sharpe ones.

### Parameter Ranges

| Parameter | Min  | Max  | Step | Values | Description |
|-----------|------|------|------|--------|-------------|
| `threshold` | 0.55 | 0.75 | 0.01 | 21 | Probability cutoff for trading |
| `confidence_margin` | 0.0 | 0.20 | 0.01 | 21 | Distance from 0.5 required |
| `kelly_clip` | 0.3 | 1.0 | 0.05 | 15 | Position size limiter |

**Total search space**: 21 × 21 × 15 = 6,615 combinations

**Coverage**: 100 trials per model = 1.5% of search space (sparse but TPE-guided)

---

## Sanity Checks (Automatic)

The script will warn you if:

### ⚠️ Sharpe > 5.0
**Meaning**: Likely overfit or data leakage
**Action**: Mandatory out-of-sample validation

### ⚠️ Sharpe std > 2.0
**Meaning**: Results vary wildly across folds
**Action**: Model not robust, don't deploy

### ⚠️ Threshold < 0.56 or > 0.74
**Meaning**: Extreme threshold (overtrade or undertrade)
**Action**: Review trade frequency, may be exploiting noise

### ⚠️ Valid folds < 4
**Meaning**: Strategy failed on multiple folds
**Action**: Check why NaN Sharpe, may need different approach

---

## Next Steps After Stage 2

### 1. Review Results
```bash
# Open Python and inspect best trial
/home/francisco/.anaconda3/envs/binance_env/bin/python
>>> import optuna
>>> study = optuna.load_study(study_name='hermite_v6_stage2_strategy',
...                            storage='sqlite:///optuna_hermite_v6_stage2.db')
>>> best = study.best_trial
>>> print(f"Sharpe: {best.value:.2f} ± {best.user_attrs['sharpe_std']:.2f}")
>>> print(f"Model: #{best.user_attrs['model_trial']}")
>>> print(f"Params: {best.params}")
```

### 2. Check if Results Make Sense

**Good signs:**
- ✅ Sharpe: 1.5-3.5 (realistic)
- ✅ Sharpe std: < 2.0 (consistent)
- ✅ Threshold: 0.58-0.68 (reasonable)
- ✅ All 5 folds valid

**Bad signs:**
- ❌ Sharpe: > 5.0 (suspicious)
- ❌ Sharpe std: > 3.0 (unstable)
- ❌ Threshold: < 0.55 or > 0.75 (extreme)
- ❌ < 3 folds valid (broken)

### 3. Mandatory Validation

**DO NOT skip this!**

See `STAGE2_CRITICAL_WARNINGS.md` for full protocol.

Minimum requirements:
1. Out-of-sample testing (2024-2025 data if available)
2. Walk-forward validation
3. Compare to realistic benchmarks (Sharpe ~1.0 for crypto)

### 4. If Validation Passes

Consider:
- Paper trading (30+ days)
- Incremental deployment (small capital)
- Continuous monitoring

### 5. If Validation Fails

Likely scenarios:
- Out-of-sample Sharpe << in-sample → **overfitting detected** (expected!)
- Walk-forward shows high variance → **not robust**
- Negative Sharpe on new data → **regime change or data leakage**

**This is GOOD** - you caught the problem before deployment!

---

## Troubleshooting

### Script Won't Start

**Error**: `Cannot load Stage 1 study`
```bash
# Check Stage 1 database exists
ls -lh optuna_hermite_v6_stage1.db

# Verify Stage 1 completion
/home/francisco/.anaconda3/envs/binance_env/bin/python -c "
import optuna
study = optuna.load_study(study_name='hermite_v6_stage1_model_quality',
                          storage='sqlite:///optuna_hermite_v6_stage1.db')
print(f'Stage 1 trials: {len(study.trials)}')
print(f'Stage 1 best: {study.best_trial.value:.4f}')
"
```

### All Trials Return NaN Sharpe

**Cause**: Strategy parameters incompatible with model outputs

**Solutions**:
1. Check if models produce valid probabilities
2. Widen parameter ranges (lower threshold_min to 0.50)
3. Check transaction costs (may be too high)

### Optimization Very Slow

**Expected**: ~9-12 seconds per trial (similar to Stage 1)

**If slower**:
- Check GPU utilization: `nvidia-smi`
- Reduce `N_TRIALS_PER_MODEL` to 50 (test run)
- Run on subset of models (TOP_MODELS = [1410, 356])

### Results Look Too Good

**Sharpe > 8.0 consistently**

**This is the expected outcome!** (As you warned)

**Action**:
1. Don't celebrate
2. Immediately run out-of-sample validation
3. Expect Sharpe to drop to 1.0-2.0 on real data
4. If OOS Sharpe also > 8.0 → investigate data leakage

---

## Estimated Runtime

**Total trials**: 500 (100 per model × 5 models)
**Time per trial**: ~10-15 seconds (similar to Stage 1)
**Total time**: 500 × 12s = 6,000s = **~1.7 hours**

**Plus overhead**: 2-3 hours total

**Compared to Stage 1**: Much faster (500 vs 1,460 trials)

---

## Command Summary

```bash
# Run Stage 2 (foreground)
/home/francisco/.anaconda3/envs/binance_env/bin/python optuna_optimize_v6_stage2.py

# Run Stage 2 (background)
nohup /home/francisco/.anaconda3/envs/binance_env/bin/python optuna_optimize_v6_stage2.py > optuna_v6_stage2.log 2>&1 &

# Monitor
tail -f optuna_v6_stage2.log

# Check progress
/home/francisco/.anaconda3/envs/binance_env/bin/python -c "
import optuna
study = optuna.load_study(study_name='hermite_v6_stage2_strategy',
                          storage='sqlite:///optuna_hermite_v6_stage2.db')
print(f'{len([t for t in study.trials if t.state.name==\"COMPLETE\"])}/500 completed')
if study.best_trial:
    print(f'Best Sharpe: {study.best_trial.value:.2f}')
"

# Kill if needed
pkill -f optuna_optimize_v6_stage2

# Dashboard (optional)
optuna-dashboard sqlite:///optuna_hermite_v6_stage2.db
```

---

## Summary

✅ **Ready to run**: `optuna_optimize_v6_stage2.py` is complete and tested
⚠️ **Remember**: Sharpe values will likely be biased (short windows, potential data issues)
🎯 **Goal**: Find best strategy parameters for validation, NOT for blind deployment
📊 **Next**: Out-of-sample validation is MANDATORY

**We are on the same page**: High Sharpe = suspicious, validation required.

---

*Created: 2025-11-28*
*Context: Optuna V6 Two-Stage Optimization*
