# Hermite NN Trading – Probabilistic Pipeline

This repository implements a calibrated forecasting stack for crypto markets. A Hermite-based neural network consumes stationary, leak-safe features, produces calibrated Gaussian forecasts plus up-probabilities, and reports an extensive “randomness vs skill” diagnostic suite. The pipeline supports single chronological splits or rolling-origin cross-validation (ROCV), conformal intervals, per-trade p-values, and simple trading policy evaluation.

---

## Project layout

```
.
├── main.py                     # CLI entry point
├── inference.py                # Lightweight checkpoint inference helper
├── src/
│   ├── __init__.py             # Runtime environment guards (OpenMP-safe defaults)
│   ├── config/
│   │   ├── defaults.yaml       # YAML configuration with data/model/eval defaults
│   │   └── settings.py         # Dataclasses + loaders
│   ├── data/
│   │   ├── binance_fetcher.py  # REST client (synthetic-safe fallbacks)
│   │   └── dataset.py          # Stationary, windowed dataset builder
│   ├── eval/
│   │   ├── conformal.py        # Quantiles and p-values
│   │   ├── diagnostics.py      # Statistical tests and calibration metrics
│   │   └── strategy.py         # Threshold policy back-test
│   ├── features/
│   │   ├── stationary.py       # Leak-safe feature engineering
│   │   ├── liquidity.py        # Time-aligned liquidity proxies
│   │   └── orderbook.py        # Order-book proxies
│   ├── models/hermite.py       # Probabilistic Hermite forecaster
│   ├── pipeline/
│   │   ├── scaler.py           # Leak-guarded standardiser
│   │   ├── split.py            # ROCV splitter
│   │   └── training.py         # End-to-end training + diagnostics
│   └── reporting/plots.py      # Reliability diagram helper
└── tests/                      # Pytest suite (synthetic coverage)
```

---

## Environment

PyTorch, NumPy, and SciPy run CPU-only by default; CUDA is optional.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas scipy matplotlib requests pyyaml pytest
```

To avoid Intel OpenMP shared-memory issues on restricted systems, ensure the following variables are exported before running any scripts (the package sets these automatically when imported):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export KMP_AFFINITY=disabled
export KMP_INIT_AT_FORK=FALSE
```

---

## Configuration

Default hyper-parameters live in `src/config/defaults.yaml` and are loaded into structured dataclasses. The key sections are:

| Section     | Highlights                                                                                  |
|-------------|----------------------------------------------------------------------------------------------|
| `binance`   | symbol, interval, history limit, REST endpoints                                             |
| `data`      | window length, forecast horizon, validation split, stationary feature windows, extras flag  |
| `features`  | liquidity/order-book proxy parameters                                                       |
| `model`     | Hermite block size, degree, maps, dropout                                                   |
| `training`  | batch size, epochs, LR, optimiser, scheduler, gradient clip, BCE weight, LR range test flag |
| `evaluation`| alpha, ROCV folds, validation block size, threshold τ, per-trade cost, Markdown toggle      |
| `reporting` | output directory, legend metadata                                                           |

Override any subset by:

```bash
python main.py --config my_experiment.yaml
```

CLI flags supersede YAML:

| Flag            | Meaning                                        |
|-----------------|------------------------------------------------|
| `--cv`          | Enable ROCV (expanding folds) instead of single split |
| `--alpha`       | Override conformal miscoverage level (default 0.1)    |
| `--threshold`   | Trading threshold τ for strategy evaluation           |
| `--cost-bps`    | Transaction cost per trade (basis points)             |
| `--prob-source` | Choose between calibrated CDF (`cdf`) or sigmoid logits (`logit`) |
| `--use-lstm/--no-use-lstm` | Force-enable or disable the temporal LSTM encoder         |
| `--cv-folds`    | Override the number of rolling-origin folds           |
| `--save-md/--no-save-md` | Toggle Markdown report output                |
| `--seed`        | Deterministic seed override                           |
| `--results-dir` | Custom directory for reports and diagnostics          |

---

## Running training

### Single chronological split

```bash
python main.py --results-dir reports/run_single
```

### Rolling-origin cross-validation (5 folds by default)

```bash
python main.py --cv --results-dir reports/run_rocv
```

During each run the trainer:

1. Builds leak-safe stationary features:
   - `log_ret_close`, `hl_range`, `oc_gap`, train-only z-scored volume inside the window.
   - Rolling context (means/stds, autocorr, drawdown) and day-of-week one-hot.
   - Optional time-aligned liquidity/order-book extras when `data.use_extras` is true.
2. Splits chronologically and applies a leak-guarded scaler (raises if future indices leak into the fit slice).
3. Trains the probabilistic Hermite forecaster with Gaussian NLL + λ·BCE (configurable λ).
4. Applies dropout, weight decay, and gradient clipping; restores the best checkpoint via early stopping.
5. Reserves a calibration tail from each train fold, computes conformal quantiles, intervals, and p-values.
6. Logs calibrated diagnostics, probabilistic calibration, strategy metrics, and baseline comparisons.

Per-epoch logs include total loss, Gaussian NLL, classification BCE, Brier score, and the instantaneous learning rate; training halts early if any loss becomes non-finite or explodes beyond a safe threshold.

---

## Outputs

Every run produces:

- **Results CSV** `reports/results_<timestamp>.csv` – fold-aggregated metrics (`mean±std` when multiple folds).
- **Markdown report** *(optional)* mirroring the CSV plus a legend for every abbreviation.
- **Reliability diagrams** (raw + calibrated) saved per fold under `<results-dir>/plots/reliability_{raw|calibrated}_fold_{k}.png`.
- **Probability diagnostics plots** `<results-dir>/plots_fold_k_{prob_hist|pit_qq|sign_scatter}.png` (histogram of calibrated `p_up`, PIT QQ plot, and sign scatter).
- **Prediction exports** `reports/predictions/predictions_fold_k_<timestamp>.csv` containing time stamps, μ, σ, logits, calibrated/raw probabilities, conformal bands, p-values, and PIT z-scores.
- **LR range plot** `lr_range_fold_{k}.png` when `training.enable_lr_range_test` is enabled to visualise loss response vs learning rate.
- **Structured logs** (JSON) summarising device selection, fold hashes, and seeds.
- **Summary JSON** `reports/summary.json` with per-fold/average DirAcc, AUC, MZ stats, PT p-values, Brier components, conformal coverage/width, and strategy metrics for every threshold.

### Learning-rate range test

To explore stable learning rates, enable the built-in LR sweep in a YAML override:

```yaml
training:
  enable_lr_range_test: true
  lr_range_min: 1.0e-5
  lr_range_max: 5.0e-2
  lr_range_steps: 60
```

The trainer will perform a single pass (fold 0) across exponentially spaced LRs and emit `plots/lr_range_fold_0.png`, plotting loss against LR on a log scale.

Key columns in the results table:

- Return/price errors: `MAE_return`, `RMSE_return`, `MAE_price`, `sMAPE_price`.
- Classification diagnostics: `DirAcc`, `Binom_p`, `Brier`, `AUC`, `ECE`.
- Statistical tests: `DM_p_SE`, `DM_p_AE`, `MZ_intercept`, `MZ_slope`, `MZ_F_p`, `Runs_p`, `LjungBox_p`.
- Conformal metrics: `Conf_Coverage@X%`, `Conf_Width@X%`, per-prediction p-values in the detailed forecast frame.
- Strategy evaluation: `Sharpe_strategy`, `MDD_strategy`, `Turnover`, plus `Sharpe_naive_long` and `Sharpe_naive_flat`.

## Model architecture sketch

The Hermite forecaster follows the exact flow below (aligned with the implementation in `src/models/hermite.py`):

1. **Inputs** – stationary feature vector `x ∈ ℝ^d` (windowed features optionally encoded by the LSTM, plus contextual features). No bias is appended; scaling is handled upstream.
2. **Linear maps** – multiple learnable projections `Aᵢ, Bⱼ ∈ ℝ^{m×d}` generate `zᵢ = Aᵢ x`, `zⱼ = Bⱼ x`.
3. **Hermite activation** – each projection is passed through `h(z) = Σₙ cₙ Heₙ(z) + d₀ + d₁z` where:
   - `Heₙ` is the probabilist/physicist Hermite polynomial (configurable).
   - Coefficients `cₙ` are learnable (initialised positive/small); `(d₀, d₁)` provide optional affine terms.
   - The elementwise derivative `h′(z)` is available for the Jacobian trace.
4. **Symmetric transform** – symmetric features `F(x) = Σᵢ Aᵢᵀh(Aᵢx) − Σⱼ Bⱼᵀh(Bⱼx) + b` capture even/odd structure, while the Jacobian trace proxy `Tr(J(x))` sums row-norm–weighted derivatives.
5. **Feature concatenation** – `[x ; F(x) ; Tr(J(x))] ∈ ℝ^{2d+1}` becomes the input to the shared pre-head (LayerNorm + GELU + Dropout).
6. **Probabilistic heads** – three parallel MLPs output:
   - `μ` (mean log-return),
   - `logvar` (log variance for Gaussian NLL + conformal scaling),
   - `logits` (directional classification). Sigmoid logits are always available, and—when `model.prob_source='cdf'`—the code uses `p_up = Φ(μ / σ)` via the Gaussian CDF for probability-based tasks.
7. **Optional temporal encoder** – if `model.use_lstm=True`, the first `feature_window × len(window_feature_columns)` features are reshaped to `(B, T, F)` and encoded by the lightweight LSTM before being concatenated with the contextual remainder.
8. **Losses & calibration** – training minimises a weighted sum of Gaussian NLL, classification loss (BCE or focal), variance regulariser, and a gentle sign-consistency hinge. Probabilities are calibrated via temperature scaling + isotonic on a rolling calibration slice, then evaluated with conformal residuals for coverage/width, PT/MZ diagnostics, and multi-threshold Kelly-sized strategies with conformal gating.

The Markdown file ends with a legend explaining every abbreviation (p_up, DM, MZ, ECE, etc.).

---

## Probabilistic diagnostics

- **Gaussian mean/variance head** for heteroscedastic NLL.
- **`p_up` head** calibrated through BCE with configurable weight.
- **Conformal calibration**:
  - Symmetric absolute residual quantiles on calibration slices.
  - Per-prediction p-values via empirical conformal scores.
  - Coverage and width reporting in both return and price space.
- **Randomness vs skill tests**:
  - Exact two-sided binomial sign test.
  - Diebold–Mariano vs zero-return baseline (squared/absolute loss).
  - Mincer–Zarnowitz regression with joint F-test (converted to χ² for q=2).
  - Runs test for residual sign patterns.
  - Ljung–Box statistic on residual autocorrelation.
- **Probability calibration metrics**:
  - Brier score, ROC AUC, expected calibration error (ECE).
  - Reliability diagrams (saved per fold).
- **Strategy evaluation**:
  - Threshold-based long/short/flat policy with costs, annualised Sharpe, max drawdown, hit rate, turnover.
  - Always-long and always-flat baselines.

---

## Testing

Run the suite (set OpenMP variables if your system restricts shared memory):

```bash
pytest
```

Highlights:

- Dataset / scaler guards (no future leakage, non-zero variance, zero-mean stationary returns).
- Probabilistic head sanity (tensor shapes, finite NLL).
- Conformal coverage on synthetic noise.
- Diebold–Mariano, runs, Ljung–Box, and calibration metrics regression tests.
- End-to-end synthetic smoke test executed in a subprocess, skipping automatically when the host disallows Intel OpenMP shared memory (common in sandboxed CI).

---

## Notes

- The codebase defaults to CPU execution and caps BLAS threads to one to remain reproducible across constrained environments.
- Liquidity/order-book extras are optional; if real historical series are unavailable, keep `data.use_extras: false`.
- When running on machines with sparse `/dev/shm`, export the environment variables shown earlier to avoid Intel OpenMP crashes.

For questions or improvement ideas (e.g. SPA test variants, alternative benchmarks, or additional diagnostics), open an issue or submit a pull request. Contributions welcome!
