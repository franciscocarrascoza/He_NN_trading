# Why Sharpe Ratio is Used Instead of P-Values for Optimization

## Your Question

> "Currently this repository has no trading capabilities and everything is about price prediction accuracy. Is that Sharpe a metric of prediction accuracy? Why not better use p-value (also among metrics harvested in the repo results) which would at least tell me how reproducible or 'out of luck' the results are?"

## Executive Summary

**You're absolutely right to question this.** The Sharpe ratio being used in Stage 1 **is NOT a metric of pure prediction accuracy**—it's a **contaminated metric** that mixes prediction quality with strategy execution, which creates the weak correlation problem (R=0.188) we discovered.

**The p-values you mention (`PT_p`, `MZ_F_p`, `DM_p`) ARE better metrics for reproducibility**, but they serve a different purpose: **statistical testing** rather than **optimization objectives**.

---

## What is the "Sharpe" in Your Repository?

### Location in Code

From `src/eval/strategy.py:11-28`:

```python
def _annualised_sharpe(
    returns: np.ndarray,
    seconds_per_step: float,
    *,
    freq_per_year: float | None = None,
) -> float:
    """Compute annualised Sharpe using sample statistics with stability guards."""

    if returns.size == 0:
        return float("nan")
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std <= 0.0:
        return float("nan")
    periods_per_year = freq_per_year if freq_per_year is not None else (365.0 * 24.0 * 3600.0) / max(
        seconds_per_step, 1.0,
    )
    return float((mean / std) * np.sqrt(periods_per_year))
```

### What It Computes

**Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Returns**

This is computed from `strategy_returns`, which are:

```python
# From strategy.py:97-111
long_mask = probabilities >= threshold  # USES FIXED THRESHOLD
short_mask = probabilities <= (1.0 - threshold)
base_mask = long_mask | short_mask
confidence_mask = np.abs(probabilities - 0.5) >= confidence_margin  # USES FIXED MARGIN
active_mask = base_mask & confidence_mask

kelly = np.clip(2.0 * (probabilities - 0.5), -clip, clip)  # USES FIXED CLIP
positions = np.where(active_mask, kelly, 0.0)
trade_changes = np.diff(np.concatenate(([0.0], positions)))
turnover = float(np.sum(np.abs(trade_changes)))
gross_returns = positions * simple_returns
cost = np.abs(trade_changes) * per_trade_cost  # TRADING COSTS
strategy_returns = gross_returns - cost  # NET RETURNS
```

### Critical Insight

**This Sharpe is a SIMULATED TRADING performance metric, not a prediction accuracy metric.**

It depends on:
1. ✅ **Prediction quality** (probabilities from the model)
2. ❌ **Fixed strategy parameters** (threshold=0.624, confidence_margin=0.105, kelly_clip=0.940)
3. ❌ **Transaction costs** (cost_bps, slippage_bps)
4. ❌ **Market conditions** (returns distribution in validation fold)

**Therefore**: A model with **perfect predictions** can have **terrible Sharpe** if the fixed strategy parameters don't match its probability distribution.

---

## What P-Values Does Your Repository Compute?

From `reports/summary.json` and `src/pipeline/training.py:1390-1404`:

### 1. **PT_p** (Probability Integral Transform Test)

**What it tests**: Are the predictive distributions well-calibrated?

**Null hypothesis**: The PIT values should be uniformly distributed ~ U(0,1) if the model's uncertainty estimates are correct.

**How it's computed**:
```python
# From src/utils/utils.py
pit_z = pit_zscore(y_val_np, mu_val_np, sigma_val_np)
pt_p = pt_test(pit_z)  # Tests if PIT z-scores are standard normal
```

**Interpretation**:
- `PT_p` close to 1.0 → Model's uncertainty estimates are well-calibrated
- `PT_p` close to 0.0 → Model's uncertainty is poorly calibrated

**Example from your results**:
```json
{
  "fold_id": 2,
  "PT_p": 1.0,  ← Perfect calibration!
  "AUC": 0.607,
  "Sharpe_strategy": NaN  ← But trading failed!
}
```

This shows **excellent probabilistic calibration** but **no tradeable signals** with fixed strategy.

---

### 2. **MZ_F_p** (Mincer-Zarnowitz F-test)

**What it tests**: Are predictions unbiased and efficient?

**Null hypothesis**: Running the regression `y_true = α + β * y_pred + ε`, we should have α=0 and β=1 (unbiased, efficient forecasts).

**How it's computed**:
```python
# From src/eval/diagnostics.py:93-122
def mincer_zarnowitz(y_true: np.ndarray, forecast: np.ndarray) -> MincerZarnowitzResult:
    # Run regression: y_true = intercept + slope * forecast
    # Test H0: intercept = 0 AND slope = 1
    # Returns F-statistic and p-value
```

**Interpretation**:
- `MZ_F_p` = 0.0 → **Reject** null hypothesis → Predictions are **biased** or **inefficient**
- `MZ_F_p` > 0.05 → **Accept** null hypothesis → Predictions are **unbiased** and **efficient**

**Example from your results**:
```json
{
  "fold_id": 0,
  "MZ_F_p": 0.0,  ← Predictions are biased!
  "MZ_intercept": 0.0010,
  "MZ_slope": 0.0039,  ← Should be 1.0!
  "AUC": 0.543
}
```

This shows the model's **point predictions are severely biased** (slope = 0.004 instead of 1.0).

---

### 3. **DM_p** (Diebold-Mariano Test)

**What it tests**: Is your model significantly better than a benchmark (e.g., naive forecast)?

**Null hypothesis**: Model and benchmark have equal predictive accuracy.

**How it's computed**:
```python
# From src/eval/diagnostics.py:38-82
def diebold_mariano(
    y_true: np.ndarray,
    forecast: np.ndarray,
    benchmark: np.ndarray,
    *,
    horizon: int,
    loss: str = "mse",
) -> DieboldMarianoResult:
    # Compute loss differential: d = loss_model - loss_benchmark
    # Test if mean(d) significantly differs from 0
    # Returns p-value
```

**Interpretation**:
- `DM_p` < 0.05 → Model is **significantly different** from benchmark
- `DM_p` > 0.05 → Model is **not significantly better** than benchmark

**Your results show**:
```json
"DM_p_SE": 0.123,  // MSE-based test
"DM_p_AE": 0.456,  // MAE-based test
```

---

### 4. **Binom_p** (Binomial Test for Directional Accuracy)

**What it tests**: Is directional accuracy significantly better than coin flip (50%)?

**Null hypothesis**: P(correct direction) = 0.5

**How it's computed**:
```python
# From src/eval/diagnostics.py:17-30
def binomial_test_pvalue(successes: int, trials: int, p: float = 0.5) -> float:
    # Exact binomial test
    # Returns p-value for H0: success rate = p
```

**Interpretation**:
- `Binom_p` < 0.05 → DirAcc is **significantly > 50%**
- `Binom_p` > 0.05 → DirAcc is **not significantly different from random**

---

## Why Not Optimize on P-Values?

### Problem 1: P-Values are Binary Tests, Not Continuous Objectives

**P-values test hypotheses**:
- `p < 0.05` → Reject null (significant)
- `p > 0.05` → Accept null (not significant)

**But optimization needs gradients**:
- AUC: 0.50 (bad) → 0.60 (good) → 0.70 (excellent) → smooth improvement
- P-value: 0.06 (fail) → 0.04 (pass) → 0.001 (still pass) → **no reward for "how much better"**

**Example**:
- Model A: `PT_p = 0.051` (fails test)
- Model B: `PT_p = 0.049` (passes test)
- Difference is **tiny** but categorical

TPE sampler needs **continuous feedback** to learn parameter relationships.

---

### Problem 2: P-Values Measure Different Aspects

Each p-value tests a **different statistical property**:

| P-Value  | Tests                  | Optimizing Would Lead To |
|----------|------------------------|---------------------------|
| `PT_p`   | Calibration            | Well-calibrated uncertainty (good!) |
| `MZ_F_p` | Unbiased predictions   | Unbiased point forecasts (good!) |
| `DM_p`   | Superiority to benchmark | Better than naive (good!) |
| `Binom_p`| Directional skill      | Better than 50% DirAcc (good!) |

**But**:
- You can have **perfect calibration** (`PT_p = 1.0`) but **terrible AUC** (0.51)
- You can have **unbiased predictions** (`MZ_F_p > 0.05`) but **no directional skill**
- You can beat a **weak benchmark** (`DM_p < 0.05`) but still be **useless for trading**

**No single p-value captures "overall prediction quality".**

---

### Problem 3: P-Values Don't Tell You "How Good"

**Consider these two models**:

| Model | AUC  | DirAcc | PT_p  | MZ_F_p | Binom_p |
|-------|------|--------|-------|--------|---------|
| A     | 0.70 | 0.60   | 0.20  | 0.01   | 0.001   |
| B     | 0.52 | 0.51   | 0.80  | 0.30   | 0.40    |

**By p-values**: Model B looks better (higher p-values = passes tests)
**By prediction quality**: Model A is **vastly superior** (70% AUC vs 52%)

**Why?**
- Model A: Excellent predictions but **slightly biased** → low `MZ_F_p`
- Model B: Barely better than random but **perfectly unbiased** → high `MZ_F_p`

**For trading**: You want Model A (fix the bias), not Model B (useless predictions).

---

## Why Sharpe is Currently Used (Despite Problems)

### Historical Context

Your v4 optimization used Sharpe directly:
```python
# v4 objective (from OPTUNA_V6_BRIEFING.md)
objective = 0.65 * Sharpe + 0.25 * AUC + 0.10 * DirAcc
```

**Why Sharpe?**
1. **End-to-end metric**: Directly measures what we care about (profit)
2. **Captures trading reality**: Includes costs, risk, drawdowns
3. **Single number**: Easy to compare models

**Problem discovered in v5**:
- Sharpe can be **NaN** or **infinite** with bad parameters
- v5 crashed 127/254 trials due to Sharpe failures
- Led to v6's two-stage approach

---

### V6's Compromise: Stage 1 (AUC+DirAcc) + Stage 2 (Sharpe)

**Stage 1 (current)**:
- Objective: `0.7 * AUC + 0.3 * DirAcc`
- **Pure prediction quality** metrics
- Robust (no crashes)
- **Issue**: Weak correlation with Sharpe (R=0.188)

**Stage 2 (planned)**:
- Objective: `Sharpe` directly
- Optimize strategy parameters
- Expected to fix the correlation issue

---

## Your Question: Should We Use P-Values?

### Short Answer: **As Validation Metrics, Yes. As Optimization Objectives, No.**

### Why Not as Optimization Objectives?

1. **Binary nature**: Pass/fail tests don't provide smooth gradients for TPE
2. **Multi-faceted**: No single p-value captures "overall quality"
3. **Magnitude insensitivity**: `p=0.001` and `p=0.049` both "pass" but differ hugely
4. **Inverse to quality**: Low p-values are good (reject null) but optimizer maximizes → confusing

### Why Yes as Validation Metrics?

**P-values tell you if results are "real" or "lucky"**:

| Scenario | AUC  | Sharpe | PT_p | MZ_F_p | Interpretation |
|----------|------|--------|------|--------|----------------|
| 1        | 0.60 | 3.5    | 0.95 | 0.80   | ✅ **Excellent**: High quality, well-calibrated, reproducible |
| 2        | 0.60 | 3.5    | 0.02 | 0.01   | ⚠️ **Suspicious**: Good metrics but **biased** → likely overfitting |
| 3        | 0.52 | 0.8    | 0.90 | 0.60   | ❌ **Useless**: Barely better than random, but well-calibrated |

**Use p-values to filter out "lucky" models**:
- After Stage 1: Check if top models have **good p-values** (PT_p > 0.05, MZ_F_p > 0.05)
- If not → model is **overfit** or **poorly specified**
- Discard those models even if AUC looks good

---

## Recommended Approach

### Stage 1: Optimize Prediction Quality

**Objective** (current):
```python
score = 0.70 * AUC + 0.30 * DirAcc
```

**Validation checks** (add these):
```python
# After optimization, filter top models by:
valid_models = []
for model in top_20_models:
    if (model.PT_p > 0.05 and      # Well-calibrated
        model.MZ_F_p > 0.05 and    # Unbiased predictions
        model.Binom_p < 0.05):     # Significantly better than 50%
        valid_models.append(model)
    else:
        print(f"Model {model.trial_num} REJECTED:")
        print(f"  PT_p={model.PT_p:.3f} (need > 0.05)")
        print(f"  MZ_F_p={model.MZ_F_p:.3f} (need > 0.05)")
        print(f"  Binom_p={model.Binom_p:.3f} (need < 0.05)")
```

### Stage 2: Optimize Strategy + Validate Statistical Significance

**Objective**:
```python
score = Sharpe_ratio  # Direct trading performance
```

**Final validation**:
```python
# On out-of-sample data
if (final_model.Sharpe > 2.0 and           # Good trading performance
    final_model.PT_p > 0.05 and            # Well-calibrated
    final_model.MZ_F_p > 0.05 and          # Unbiased
    final_model.DM_p_SE < 0.05 and         # Better than benchmark
    final_model.Sharpe_pvalue < 0.05):     # Sharpe significantly > 0
    print("✅ Model ready for production")
else:
    print("⚠️ Model may be overfit or lucky")
```

---

## Why Sharpe Currently Appears in Stage 1 (Your Confusion)

**You're right to be confused!** Here's what's happening:

1. **Sharpe is LOGGED** but **NOT optimized** in Stage 1:
   ```python
   # From optuna_optimize_v6.py:313-316
   score = W_AUC * auc + W_DIRACC * dir_acc  # ← Optimized

   trial.set_user_attr("sharpe", sharpe)  # ← Only logged for analysis
   return float(score)  # ← TPE sees ONLY AUC+DirAcc
   ```

2. **Why log it?**
   - **Diagnostic**: Check if AUC/DirAcc correlate with Sharpe
   - **Pre-filtering**: Identify models that completely fail trading
   - **Analysis**: Understand parameter effects on trading (even if not optimized)

3. **Why NOT optimize it?**
   - **Fixed strategy** creates bottleneck (threshold=0.624)
   - **High variance** (Sharpe ranges -9.2 to 19.6)
   - **42% NaN rate** (many models produce no trades with fixed threshold)
   - **Weak correlation** (R=0.188 with AUC) → optimizing Sharpe would ignore prediction quality

---

## Your Actual Results: P-Value Analysis

From `reports/summary.json`:

| Fold | AUC  | DirAcc | PT_p  | MZ_F_p | Sharpe |
|------|------|--------|-------|--------|--------|
| 0    | 0.54 | 0.469  | 0.541 | 0.000  | NaN    |
| 1    | 0.57 | 0.531  | 0.519 | 0.000  | NaN    |
| 2    | 0.61 | 0.484  | 1.000 | 0.000  | NaN    |
| 3    | 0.55 | 0.469  | 0.818 | 0.000  | NaN    |
| 4    | 0.50 | 0.484  | 1.000 | 0.000  | NaN    |

**Analysis**:
- ✅ **PT_p mostly good** (0.5-1.0) → Uncertainty is **well-calibrated**
- ❌ **MZ_F_p all zero** → Predictions are **severely biased**
- ❌ **Sharpe all NaN** → **No tradeable signals** with fixed strategy

**Conclusion**:
1. Model produces **well-calibrated uncertainties** (good!)
2. But **point predictions are biased** (MZ slope = 0.004-0.5 instead of 1.0)
3. And **fixed strategy doesn't trade** (threshold mismatch)

**This is exactly why Stage 2 is needed!**

---

## Final Recommendation

### ✅ **Keep current Stage 1 approach**
- Optimize: `0.7*AUC + 0.3*DirAcc`
- Log: `Sharpe`, `PT_p`, `MZ_F_p`, `DM_p`, `Binom_p`

### ✅ **Add p-value filtering after Stage 1**
```python
# Select top models that pass statistical tests
top_models = [m for m in top_20 if
              m.PT_p > 0.05 and      # Calibrated
              m.MZ_F_p > 0.05 and    # Unbiased
              m.Binom_p < 0.05]      # Skillful
```

### ✅ **Stage 2: Optimize Sharpe with strategy params**
- This will fix the weak correlation issue
- Each model gets its optimal threshold

### ✅ **Final validation: Use ALL metrics**
- AUC > 0.58
- DirAcc > 0.53
- Sharpe > 2.0
- PT_p > 0.05 (calibrated)
- MZ_F_p > 0.05 (unbiased)
- DM_p < 0.05 (beats benchmark)

---

## Summary Table

| Metric Class | Examples | Purpose | Use in Optimization | Use in Validation |
|--------------|----------|---------|---------------------|-------------------|
| **Prediction Quality** | AUC, DirAcc, Brier | Measure accuracy | ✅ **Yes** (Stage 1) | ✅ Yes |
| **Statistical Tests** | PT_p, MZ_F_p, DM_p | Test reproducibility | ❌ No (binary) | ✅ **Yes** (filter) |
| **Trading Performance** | Sharpe, Max Drawdown | Measure profitability | ✅ **Yes** (Stage 2) | ✅ Yes |

**Bottom line**:
- **Optimize** on smooth metrics (AUC, Sharpe)
- **Validate** with p-values (PT_p, MZ_F_p, DM_p)
- **Report** everything (transparency)

---

*Analysis Date: 2025-11-28*
*Repository: He_NN_trading*
*Context: Optuna V6 Stage 1 plateau investigation*
