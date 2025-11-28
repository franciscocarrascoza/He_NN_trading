# CRITICAL WARNINGS FOR STAGE 2 OPTIMIZATION

## Date: 2025-11-28
## Context: Before Starting Optuna V6 Stage 2

---

## ⚠️ USER'S CRITICAL CONCERNS ABOUT SHARPE RATIO

### Francisco's Warning (2025-11-28):

> "I need you to understand (and record it somewhere for future context recovery) that this Sharpe metric is **likely very biased** due to the short window terms, and also because I hesitate that it is true real-time fed timeseries, thus is **prone to overfitting**, that's why we have so high Sharpe's metrics."

---

## Why Sharpe May Be Unreliable

### 1. **Short Window Terms**
- Validation folds are limited in size
- Each fold evaluation uses only a subset of data
- **Small sample size** → High variance in Sharpe estimates
- **Result**: Sharpe=10+ may be statistical noise, not real signal

### 2. **Time Series Data Leakage Concerns**
- User is NOT confident the data is "true real-time fed timeseries"
- Potential issues:
  - Look-ahead bias (features computed with future information)
  - Data ordering problems
  - Temporal leakage in validation splits

### 3. **Overfitting Risk**
- High Sharpe ratios (3.6, 9.5, 19.6) are **suspiciously high**
- Real-world crypto trading Sharpe ratios are typically:
  - Excellent: 1.5-2.0
  - Good: 1.0-1.5
  - Acceptable: 0.5-1.0
- **Sharpe > 3.0 should trigger immediate skepticism**

---

## Implications for Stage 2

### DO NOT Trust Absolute Sharpe Values

**Stage 2 should focus on RELATIVE improvements, not absolute values:**

❌ **Wrong approach**: "Achieve Sharpe > 5.0"
✅ **Correct approach**: "Find strategy parameters that improve Sharpe compared to baseline"

### Validation Requirements

After Stage 2 optimization, **MANDATORY validation**:

1. **Out-of-sample testing** on completely held-out data
2. **Walk-forward validation** with strict temporal ordering
3. **Reality check**: Compare to known crypto trading benchmarks
4. **Paper trading**: Simulate in real-time before any live deployment

### Expected Outcomes

**If Sharpe IS biased/overfit** (likely):
- Stage 2 best Sharpe: 5-10 (in-sample)
- Out-of-sample Sharpe: 0.5-1.5 (realistic)
- Paper trading Sharpe: 0.3-0.8 (after costs)

**If this happens**: SUCCESS - we found realistic parameters
**If out-of-sample matches in-sample**: SUSPICIOUS - likely data leakage

---

## Stage 2 Optimization Strategy (REVISED)

### Primary Objective
**Find strategy parameters that maximize Sharpe ON THE VALIDATION SET**

But with **extreme skepticism** about absolute values.

### Secondary Objectives (Critical!)
1. **Consistency across folds**: Low std dev of Sharpe across 5 folds
2. **Robustness**: Parameters that work across different models
3. **Sanity checks**: Reasonable trade frequency, position sizes

### Parameters to Optimize
- `threshold`: Probability cutoff for trading (0.50-0.80)
- `confidence_margin`: Minimum distance from 0.5 (0.0-0.25)
- `kelly_clip`: Position size limiter (0.1-1.0)

**Keep fixed**:
- Model parameters (from Stage 1 best)
- Transaction costs (realistic: 10-20 bps)
- Time series ordering (walk-forward validation)

---

## Red Flags to Watch For

### During Stage 2 Optimization

| Observation | Interpretation | Action |
|-------------|----------------|---------|
| Sharpe > 10 consistently | **Overfitting likely** | Increase transaction costs, validate rigorously |
| Best threshold → 0.5 | Model has **no skill** | Return to Stage 1, select different model |
| Best threshold → extremes (0.7+, 0.3-) | **Cherry-picking rare events** | Reduce to more conservative range |
| NaN Sharpe > 50% of trials | **Strategy parameters incompatible** | Adjust parameter ranges |
| Sharpe std dev > 3.0 across folds | **High variance/unstable** | Not production-ready |

### After Stage 2 Completion

**BEFORE celebrating high Sharpe:**

1. ✅ Check trade frequency (should be reasonable, not 1 trade/year)
2. ✅ Check position sizes (should be within kelly_clip)
3. ✅ Check consistency (Sharpe should be similar across folds)
4. ✅ Verify temporal ordering (no future data in features)
5. ✅ Test on out-of-sample period (2024-2025 if not in training)

---

## Recommended Stage 2 Configuration

### Number of Trials
**500-1000 trials total** (not 2500 like Stage 1)

Reasoning:
- Only 3 parameters to optimize (threshold, confidence_margin, kelly_clip)
- 500 trials = ~167 trials per parameter (sufficient)
- Faster iteration for validation

### Multi-Model Testing
**Test top 5-10 models from Stage 1**, not just #1:

```python
TOP_MODELS = [1410, 356, 664, 632, 1001]  # Top 5 by Score from Stage 1

# For each model, run 100 trials to find best strategy params
# Total: 5 models × 100 trials = 500 trials
```

### Objective Function

```python
def objective_stage2(trial):
    # ... get model params from Stage 1 ...

    threshold = trial.suggest_float("threshold", 0.55, 0.75, step=0.01)
    confidence_margin = trial.suggest_float("confidence_margin", 0.0, 0.20, step=0.01)
    kelly_clip = trial.suggest_float("kelly_clip", 0.3, 1.0, step=0.05)

    # ... train model, compute Sharpe ...

    # CRITICAL: Also log consistency metrics
    trial.set_user_attr("sharpe_std", np.std(sharpe_per_fold))
    trial.set_user_attr("num_trades", total_trades)
    trial.set_user_attr("avg_position", np.mean(positions))

    return sharpe_mean  # Optimize mean Sharpe across folds
```

---

## Post-Optimization Validation Protocol

### Step 1: Sanity Checks (Immediate)

```python
best_trial = study.best_trial

# Check 1: Reasonable Sharpe
assert 0.5 < best_trial.value < 5.0, "Sharpe outside reasonable range"

# Check 2: Consistent across folds
sharpe_std = best_trial.user_attrs['sharpe_std']
assert sharpe_std < 2.0, "Sharpe too volatile across folds"

# Check 3: Sufficient trading activity
num_trades = best_trial.user_attrs['num_trades']
assert num_trades > 10, "Too few trades (overfitting to rare events)"

# Check 4: Reasonable position sizing
avg_position = best_trial.user_attrs['avg_position']
assert 0.1 < avg_position < 0.8, "Position sizes unrealistic"
```

### Step 2: Out-of-Sample Validation (Critical!)

**Use data from 2024-2025 that was NOT in any training/validation fold:**

```python
# Load completely fresh data
oos_data = load_data(start="2024-01-01", end="2025-11-28")

# Apply Stage 2 best config
best_config = apply_stage2_config(stage1_best_model, stage2_best_params)

# Train on full historical data
model = train_final_model(best_config, data=historical_data)

# Test on out-of-sample
oos_metrics = evaluate(model, oos_data)

print(f"In-sample Sharpe: {best_trial.value:.2f}")
print(f"Out-of-sample Sharpe: {oos_metrics.sharpe:.2f}")

# EXPECTED: Out-of-sample should be 30-70% of in-sample
# If OOS > 80% of in-sample → suspiciously good (possible leakage)
# If OOS < 30% of in-sample → severe overfitting
```

### Step 3: Walk-Forward Validation

```python
# Simulate real-time deployment
# Train on [0, T], validate on [T, T+1 month], retrain, validate on [T+1, T+2], etc.

wf_results = walk_forward_validation(
    model_config=best_config,
    data=full_data,
    train_window_months=12,
    test_window_months=1,
    step_months=1
)

print(f"Walk-forward avg Sharpe: {np.mean(wf_results.sharpe_per_window):.2f}")
print(f"Walk-forward Sharpe std: {np.std(wf_results.sharpe_per_window):.2f}")
```

---

## Production Deployment Criteria

**DO NOT deploy to live trading unless ALL criteria met:**

### Minimum Requirements

| Criterion | Threshold | Current Status |
|-----------|-----------|----------------|
| Out-of-sample Sharpe | > 0.5 | ❓ TBD |
| Walk-forward Sharpe | > 0.5 | ❓ TBD |
| Sharpe consistency (std) | < 1.5 | ❓ TBD |
| Max drawdown | < 30% | ❓ TBD |
| Win rate | > 48% | ❓ TBD |
| Avg trade frequency | 10-100/year | ❓ TBD |
| Position sizing | < 50% per trade | ❓ TBD |

### Paper Trading Phase

**Even if criteria met, MANDATORY 30-90 days paper trading:**

1. Deploy bot in simulation mode
2. Connect to real Binance API (live prices)
3. Simulate trades with realistic delays/slippage
4. Track metrics daily
5. **Only proceed to live if paper trading Sharpe > 0.3 after 30 days**

---

## Known Risks & Mitigation

### Risk 1: Data Leakage
**Symptom**: Extremely high Sharpe in-sample, terrible out-of-sample
**Mitigation**:
- Audit feature engineering (no look-ahead)
- Verify walk-forward split correctness
- Test on completely different time period

### Risk 2: Market Regime Change
**Symptom**: Good backtest, terrible in 2024-2025 (bear market, high volatility)
**Mitigation**:
- Test on multiple regimes (bull, bear, sideways)
- Add regime detection
- Reduce position sizing in high volatility

### Risk 3: Transaction Costs Underestimated
**Symptom**: Backtest profitable, live trading losses
**Mitigation**:
- Use realistic fees (10-20 bps, not 1 bps)
- Add slippage (5-10 bps for market orders)
- Include API latency simulation

### Risk 4: Optimization Overfitting
**Symptom**: Best params are extreme values (threshold=0.75, kelly_clip=1.0)
**Mitigation**:
- Regularize objective (penalize extreme params)
- Test parameter stability (perturb ±5%, Sharpe should be similar)
- Prefer robust configs over peak performance

---

## What Success Looks Like (Realistic)

### After Stage 2 Optimization

**In-sample (validation folds)**:
- Mean Sharpe: 1.5-3.0
- Sharpe std: 0.5-1.5
- Trade frequency: 20-50 trades/year
- Max drawdown: 15-25%

**Out-of-sample (2024-2025)**:
- Mean Sharpe: 0.8-1.5 (50-70% of in-sample)
- Win rate: 50-55%
- Max drawdown: 20-30%

**Paper trading (30 days)**:
- Mean Sharpe: 0.5-1.0 (after all costs)
- No catastrophic failures
- Positions execute as expected

**Live trading (if deployed)**:
- Target Sharpe: 0.3-0.8 (long-term sustainable)
- Max drawdown: < 25%
- Position sizing: < 30% per trade

---

## Stage 2 Script Modifications

### Critical Changes Required

1. **Add consistency metrics** to objective function
2. **Test multiple models**, not just Stage 1 #1
3. **Log trade-level details** (frequency, positions, drawdowns)
4. **Implement sanity checks** before declaring "best"
5. **Create out-of-sample validation** script

### Suggested Approach

**Option A: Conservative (Recommended)**
- Optimize each of top 5 Stage 1 models separately
- 100 trials each = 500 total
- Select best model+strategy combo by out-of-sample Sharpe

**Option B: Aggressive**
- Optimize only Stage 1 best model (#1410)
- 500 trials
- Faster but riskier (may miss better combos)

**Recommendation**: Go with **Option A** given the overfitting concerns.

---

## Final Checklist Before Live Deployment

- [ ] Stage 2 optimization completed
- [ ] Best config identified (model + strategy params)
- [ ] Sanity checks passed (Sharpe < 5, trades > 10, etc.)
- [ ] Out-of-sample validation completed (Sharpe > 0.5)
- [ ] Walk-forward validation completed (consistent Sharpe)
- [ ] Parameter stability tested (robust to perturbations)
- [ ] Paper trading for 30+ days (Sharpe > 0.3)
- [ ] Risk management implemented (stop-loss, position limits)
- [ ] Monitoring system setup (alerts, daily reports)
- [ ] Incremental capital deployment (start with 5-10% of budget)

**DO NOT skip any of these steps.**

---

## Summary: We Are On the Same Page

### What We Agree On:

1. ✅ **Sharpe is likely biased/overfit** due to short windows and potential data issues
2. ✅ **High Sharpe values (3-10) are suspicious**, not to be trusted blindly
3. ✅ **Stage 2 should proceed**, but with extreme caution and validation
4. ✅ **Out-of-sample testing is mandatory** before any deployment considerations
5. ✅ **This is still prediction/simulation**, not real trading

### What Stage 2 Will Do:

1. Find optimal `threshold`, `confidence_margin`, `kelly_clip` for top Stage 1 models
2. Evaluate using Sharpe on validation folds (with skepticism)
3. Require extensive validation on out-of-sample data
4. Provide realistic expectations for production performance

### What Stage 2 Will NOT Do:

1. Deploy anything to live trading
2. Assume Sharpe values are real without validation
3. Skip out-of-sample testing
4. Ignore consistency/robustness metrics

---

**Status**: Ready to proceed with Stage 2, eyes wide open about overfitting risks.

**Next Step**: Create `optuna_optimize_v6_stage2.py` with conservative approach and extensive validation.

---

*Documented: 2025-11-28*
*Purpose: Context recovery for future sessions*
*User: Francisco*
*Project: He_NN_trading - Optuna V6 Two-Stage Optimization*
