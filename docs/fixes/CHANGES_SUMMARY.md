Repository: He_NN_trading — Hermite activation NN for BTC forecasting.

Scope (implement only):

H1: Proper temporal encoder usage + data slicing.

H2: Stabilize uncertainty and conformal pipeline.

H3: p_up from μ/σ (CDF) + isotonic calibration.

M1: Loss weight rebalance + sign-consistency auxiliary loss (no MZ-as-loss).

M2: Rolling-origin CV with larger calibration blocks and fold stability.

M3: Strategy PnL alignment to horizon, clipped Kelly sizing, and threshold gating.

Global constraints

Python ≥ 3.10, PyTorch ≥ 2.0.

Keep existing CLI and config/dataclass flow. Add config keys only under model, conformal, training, strategy, evaluation sections in src/config/defaults.yaml. Mark added lines with # FIX:.

No data leakage. Calibration and validation/test splits must be strictly disjoint.

Reproducibility: call set_seed(42) at training and inference entry points. Add src/utils/utils.py if missing. Mark # FIX:.

For every file changed: include a docstring and # FIX: on changed lines.

Add unit tests under tests/ for each behavior change.

Files to edit (adapt names if different):

src/models/hermite.py (or model file) — temporal encoder and logvar clamp, p_up CDF output.

src/data/dataset.py — ensure window_feature_columns and feature_window are correct and documented.

src/eval/conformal.py (or conformal utility) — residual_kind support and enforced min calib size.

src/eval/evaluate.py (or predictions exporter) — isotonic calibration, p_up wiring, PIT export.

src/pipeline/training.py — loss weights, sign auxiliary loss, rolling CV config hook, calibration slice handling.

src/strategy/strategy.py — horizon-aligned PnL, clip Kelly, threshold gating.

src/config/defaults.yaml — add keys below.

src/utils/utils.py — set_seed, small helpers.

tests/ — add tests listed below.

Concrete required edits & behavior
H1 — Temporal encoder and correct slicing

Add small LSTM encoder class in src/models/hermite.py:

class LSTMTemporalEncoder(nn.Module): with signature (input_size:int, hidden_size:int=64, num_layers:int=1, bidirectional:bool=False) returning last-layer hidden of shape (B, hidden_size) on forward. Add docstring. # FIX:

In the model constructor, add config-driven flags:

model:
  use_lstm: true        # FIX:
  lstm_hidden: 64       # FIX:
  lstm_layers: 1        # FIX:


Wire these into the model instantiation. # FIX:

In forward(x) (where x is the flat feature vector):

Compute feats_per_step = len(window_feature_columns) and flat_seq_len = feature_window * feats_per_step. Validate x.shape[1] >= flat_seq_len, raise a clear error if not. # FIX:

Slice: x_seq = x[:, :flat_seq_len].view(B, feature_window, feats_per_step) and x_ctx = x[:, flat_seq_len:]. # FIX:

Pass x_seq through LSTMTemporalEncoder to produce h_last and concatenate h_last with x_ctx before the Hermite block / MLP. # FIX:

Add unit test tests/test_temporal_encoder.py:

Build a dummy tensor (B=8, W=64, F=4), run through encoder, assert output shape (8, lstm_hidden). # FIX:

H2 — Stabilize uncertainty and conformal pipeline

In src/models/hermite.py:

After logvar output, clamp:
logvar = torch.clamp(logvar, min=-12.0, max=5.0) # FIX:
sigma = torch.sqrt(torch.exp(logvar).clamp_min(1e-12)) # FIX:

Ensure sigma is returned in model outputs or made available to downstream. Keep logits and mu. # FIX:

In src/eval/conformal.py:

Add/ensure signature:
def conformal_interval(y_true, mu, sigma=None, residual_kind="abs", alpha=0.1, calib_resid=None, min_calib_size=256): with docstring. # FIX:

Behavior:

If residual_kind == "std_gauss": require sigma and calib_resid may be ((y_cal - mu_cal)/sigma_cal). Compute q = np.quantile(std_resids_cal, 1-alpha) and return L = mu - q*sigma, U = mu + q*sigma. # FIX:

If residual_kind == "abs": if calib_resid present (absolute residuals on calib), compute q = quantile(calib_resid, 1-alpha) and return mu ± q. If calib_resid missing but sigma provided, compute standardized residuals on calib then scale back. If both missing and calib_size < min_calib_size, derive fallback_scale from a larger training residual pool or np.std(y_train - mu_train) available via function param; log a warning and use q = fallback_scale * z_alpha where z_alpha is Gaussian quantile. # FIX:

If calib_size < min_calib_size, do not silently skip: set used_fallback=True and include this flag in returned metadata or saved report. # FIX:

Add tests/test_conformal.py:

Synthetic normal residuals: build y_cal and mu_cal with known σ; check that 90% intervals achieve empirical coverage within ±2%. # FIX:

H3 — p_up from μ/σ and isotonic post-calibration

In src/models/hermite.py or prediction wrapper:

Compute raw p_up_raw as Gaussian CDF:
p_up_raw = 0.5 * (1.0 + torch.erf(mu / (sigma * math.sqrt(2.0)))) # FIX:

Export p_up_raw (numpy) downstream for calibration. # FIX:

In src/eval/evaluate.py (prediction-to-report pipeline):

Fit sklearn.isotonic.IsotonicRegression(out_of_bounds="clip") on p_up_raw_cal vs y_bin_cal (calibration slice). Save calibrator object to reports/. # FIX:

Compute p_up_cal = isotonic.predict(p_up_raw_val) and use p_up_cal for strategy gating and probability metrics. # FIX:

Save p_up_raw and p_up_cal per-sample in reports/predictions_fold_*.csv. # FIX:

Add tests/test_isotonic_calibration.py:

Toy example where p_up_raw monotonic but miscalibrated; isotonic fit reduces Brier and ECE. Assert ECE reduced. # FIX:

M1 — Rebalance loss weights and add sign-consistency aux loss

Config defaults in src/config/defaults.yaml:

training:
  reg_weight: 1.0         # FIX:
  cls_weight: 1.0         # FIX:
  unc_weight: 0.5         # FIX:
  sign_hinge_weight: 0.03 # FIX:


In src/pipeline/training.py:

Compute losses as:
loss = reg_weight * nll_loss + cls_weight * cls_loss + unc_weight * unc_loss # FIX:

Implement sign auxiliary loss (light):

# safer around zero: target_sign in {-1,1}
sign_target = torch.sign(targets).clamp(min=-1, max=1)
pred_sign_approx = torch.tanh(5.0 * mu)  # small k to map μ to (-1,1)
loss_sign = F.mse_loss(pred_sign_approx, sign_target)
loss = loss + sign_hinge_weight * loss_sign  # FIX:


Do not implement MZ as a differentiable loss. Compute MZ only as post-epoch diagnostic on validation. # FIX:

Add tests/test_loss_balance.py:

Synthetic batch where targets have mixed signs; check loss_sign > 0 and that total loss decreases across a few gradient steps on a tiny model. # FIX:

M2 — Rolling CV with larger calibration blocks

Add config:

training:
  use_cv: true            # FIX:
  cv_folds: 5             # FIX:
  min_calib_size: 256     # FIX:


In src/pipeline/split.py or equivalent:

Implement/ensure rolling-origin splits produce train/calibration/validation/test with calib_block >= min_calib_size. If dataset too small, raise a clear error and exit with recommendation to increase data or reduce folds. # FIX:

In src/pipeline/training.py:

If use_cv, loop folds and save per-fold models/predictions. # FIX:

Aggregate metrics as mean±std across folds and write reports/summary.json including per-fold MZ, PT p, and conformal used_fallback flags. # FIX:

Add tests/test_cv_splits.py:

Given synthetic time series length N, create splits with cv_folds=5; verify each calib block ≥ min_calib_size and no overlapping between calib and val/test. # FIX:

M3 — Strategy: horizon alignment, clip Kelly, gate by calibrated prob & conformal p

In src/strategy/strategy.py:

Ensure PnL uses horizon-aligned returns: ret_t = price[t+H]/price[t] - 1 where H=config.data.forecast_horizon. If t+H exceeds series bound, drop those rows. # FIX:

Kelly fraction (raw): f_raw = 2 * p_up_cal - 1. Clip: f = float(np.clip(f_raw, -kelly_clip, kelly_clip)) with kelly_clip from config default 0.5. # FIX:

Gate trades only if abs(p_up_cal - 0.5) >= confidence_margin and (if enabled) conformal_p >= conformal_p_min. Add these config keys under strategy and use them. # FIX:

Config additions:

strategy:
  kelly_clip: 0.5          # FIX:
  confidence_margin: 0.10  # FIX:
  use_conformal_gate: true # FIX:
  conformal_p_min: 0.05    # FIX:


Tests tests/test_strategy_pnl.py:

Synthetic price array and p_up_cal values, test that Kelly clip applied, horizon alignment correct, and that gating blocks trades below margin. # FIX:

I/O, reports, and acceptance checks

Per-fold CSV: reports/predictions_fold_{fold}.csv must include columns timestamp,y,mu,sigma,logit,p_up_raw,p_up_cal,conformal_p,pit_z. # FIX:

Summary reports/summary.json must include per-fold DirAcc, AUC, MZ_slope, Brier, Conf_Coverage@90%, Conf_Width@90%, used_fallback_conformal booleans. # FIX:

Unit tests: run pytest -q tests/ — all added tests must pass. # FIX:

Streamlit smoke: streamlit run src/interface/app.py must not exit early with "Skipping conformal interval due to insufficient calibration sample" unless data truly lacks min_calib_size. If fallback is used, the UI must display an explicit st.warning with reason. # FIX:

Implementation notes and cautions (must follow)

Do not add MZ as a gradient loss. Use only as post-epoch diagnostic and for model selection. # FIX:

When computing sigma from logvar, clamp first to avoid NaNs / infinities: logvar = torch.clamp(logvar, -12.0, 5.0). # FIX:

Isotonic calibration must be fit on calibration slice only and then applied to validation/test. Save the calibrator (pickle) to reports/. # FIX:

Keep classification head logits for backward compatibility but prefer p_up_raw (CDF) for probabilities unless prob_source config overrides. Add model.prob_source: cdf default if not present. # FIX:

Tests to add (explicit list)

tests/test_temporal_encoder.py — shape check. # FIX:

tests/test_conformal.py — coverage on synthetic normals. # FIX:

tests/test_isotonic_calibration.py — isotope improves ECE on toy set. # FIX:

tests/test_loss_balance.py — sign-loss effect and training step decrease. # FIX:

tests/test_cv_splits.py — splits validity and calib size. # FIX:

tests/test_strategy_pnl.py — horizon alignment, Kelly clip, gating. # FIX:

Deliverables (explicit)

Full contents of each modified/new file (not diffs): model file(s), conformal, evaluate pipeline, training, strategy, utils, config updates, tests. Each changed line must have # FIX:.

A short commit-style summary listing modified files and a one-line reason for each change. # FIX:

Instructions to run tests and Streamlit app:

cd He_NN_trading
pytest -q tests/
streamlit run src/interface/app.py


# FIX:

Acceptance criteria (Claude must ensure)

pytest -q tests/ passes (new tests pass). # FIX:

streamlit run src/interface/app.py runs without aborting due to missing calibration unless dataset truly lacks min_calib_size; if fallback used, UI shows a clear warning. # FIX:

reports/predictions_fold_*.csv and reports/summary.json are created and include required fields. # FIX:

No change introduces data leakage: isotonic is fit only on calibration slice; conformal uses disjoint calib residuals. # FIX:
