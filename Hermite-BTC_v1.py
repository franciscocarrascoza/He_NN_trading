#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python3
# coding: utf-8

"""
Hermite NN for Bitcoin Close prediction (returns-based).
- Full script: loading, preprocessing (log returns), model, training, evaluation,
  reconstruction to prices, CSV export, plotting (with dates).
- Train target: future log returns (horizon steps).
- Final predictions are reconstructed to price units for metrics and plotting.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Tuple
from scipy.special import factorial


# In[9]:


# ---------------------------
# Hermite polynomial utilities
# ---------------------------

def hermite_polys(z: torch.Tensor, N: int, version: str = "probabilist") -> torch.Tensor:
    assert version in ("probabilist", "physicist"), "version must be 'probabilist' or 'physicist'"
    polys = []
    if N >= 0:
        polys.append(torch.ones_like(z))
    if N >= 1:
        polys.append(z if version == "probabilist" else 2.0 * z)
    for n in range(1, N):
        if version == "probabilist":
            nextp = z * polys[n] - n * polys[n-1]
        else:
            nextp = 2.0 * z * polys[n] - 2.0 * n * polys[n-1]
        polys.append(nextp)
    return torch.stack(polys, dim=0)

def hermite_derivative_coeff_factor(n: int, version: str = "probabilist") -> float:
    return float(n) if version == "probabilist" else float(2 * n)


# ---------------------------
# Hermite Activation
# ---------------------------

class HermiteActivation(nn.Module):
    def __init__(self,
                 degree: int = 5,
                 in_features: int = 1,
                 learn_coeffs: bool = True,
                 coeffs: Optional[torch.Tensor] = None,
                 d0: float = 0.0,
                 d1: float = 0.0,
                 version: str = "probabilist"):
        super().__init__()
        self.N = degree
        self.version = version
        if coeffs is not None:
            c_init = torch.tensor(coeffs, dtype=torch.get_default_dtype()).view(self.N + 1)
        else:
            # small random init
            c_init = 0.05 * torch.randn(self.N + 1, dtype=torch.get_default_dtype())
        if learn_coeffs:
            self.c = nn.Parameter(c_init.clone())
        else:
            self.register_buffer("c", c_init.clone())
        self.register_buffer("d0", torch.tensor(float(d0)))
        self.register_buffer("d1", torch.tensor(float(d1)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., )
        polys = hermite_polys(z, self.N, version=self.version)   # (N+1, *z.shape)
        c_view = self.c.view(self.N + 1, *([1] * z.dim()))
        return (c_view * polys).sum(dim=0) + self.d0 + self.d1 * z

    def derivative(self, z: torch.Tensor) -> torch.Tensor:
        if self.N == 0:
            return torch.zeros_like(z) + self.d1
        polys = hermite_polys(z, self.N - 1, version=self.version)  # (N, *z.shape)
        factors = [hermite_derivative_coeff_factor(n, self.version) for n in range(1, self.N + 1)]
        factors_t = torch.tensor(factors, dtype=self.c.dtype, device=self.c.device).view(-1, *([1] * z.dim()))
        c1 = self.c[1:].view(-1, *([1] * z.dim()))
        return (c1 * factors_t * polys).sum(dim=0) + self.d1


# ---------------------------
# SymmetricHermiteBlock (returns summary by default)
# ---------------------------

class SymmetricHermiteBlock(nn.Module):
    """
    F(s) = sum A^T h(A s) - sum B^T h(B s) + bF
    J(s) = sum A^T diag(h'(A s)) A - ...
    By default returns j_mode='summary' -> (trace, fro)
    """
    def __init__(self,
                 input_dim: int,
                 maps_A: int = 1,
                 maps_B: int = 0,
                 hidden_dim: Optional[int] = None,
                 hermite_degree: int = 5,
                 hermite_version: str = "probabilist",
                 learn_coeffs: bool = True,
                 coeffs: Optional[np.ndarray] = None):
        super().__init__()
        self.d = input_dim
        self.hdim = hidden_dim if hidden_dim is not None else input_dim
        self.A_list = nn.ModuleList([nn.Linear(self.d, self.hdim, bias=False) for _ in range(maps_A)])
        self.B_list = nn.ModuleList([nn.Linear(self.d, self.hdim, bias=False) for _ in range(maps_B)])
        coeffs_tensor = torch.tensor(coeffs) if coeffs is not None else None
        self.hermite = HermiteActivation(degree=hermite_degree,
                                         in_features=self.hdim,
                                         learn_coeffs=learn_coeffs,
                                         coeffs=coeffs_tensor,
                                         d0=0.0, d1=0.0,
                                         version=hermite_version)
        self.bF = nn.Parameter(torch.zeros(self.d))

    def forward(self, s: torch.Tensor, j_mode: str = "summary"):
        """
        s: (batch, d)
        returns: F_acc (batch,d), J_repr depending on mode
        - 'summary' -> (batch,2) [trace, fro]
        - 'full' -> (batch,d,d)
        - 'diag' -> (batch,d)
        - 'flatten' -> (batch, d*d)
        """
        batch = s.shape[0]
        device = s.device
        F_acc = torch.zeros_like(s, device=device)
        J_acc = torch.zeros(batch, self.d, self.d, device=device)

        # positive A terms
        for A in self.A_list:
            z = A(s)                        # (batch, hdim)
            h_z = self.hermite(z)           # (batch, hdim)
            F_acc = F_acc + torch.matmul(h_z, A.weight.t())  # (batch,d)
            hprime = self.hermite.derivative(z)             # (batch, hdim)
            M = A.weight                                   # (hdim, d)
            M_exp = M.unsqueeze(0).expand(batch, -1, -1)    # (batch, hdim, d)
            Mh = M_exp * hprime.unsqueeze(2)               # (batch, hdim, d)
            J_acc = J_acc + torch.matmul(M.t().unsqueeze(0).expand(batch, -1, -1), Mh)

        # negative B terms
        for B in self.B_list:
            z = B(s)
            h_z = self.hermite(z)
            F_acc = F_acc - torch.matmul(h_z, B.weight.t())
            hprime = self.hermite.derivative(z)
            M = B.weight
            M_exp = M.unsqueeze(0).expand(batch, -1, -1)
            Mh = M_exp * hprime.unsqueeze(2)
            J_acc = J_acc - torch.matmul(M.t().unsqueeze(0).expand(batch, -1, -1), Mh)

        F_acc = F_acc + self.bF.unsqueeze(0)

        if j_mode == "full":
            return F_acc, J_acc
        elif j_mode == "diag":
            return F_acc, torch.diagonal(J_acc, dim1=1, dim2=2)
        elif j_mode == "flatten":
            return F_acc, J_acc.view(batch, -1)
        else:
            trace = torch.einsum('bii->b', J_acc).unsqueeze(1)   # (batch,1)
            fro = torch.linalg.norm(J_acc.view(batch, -1), dim=1, keepdim=True)  # (batch,1)
            return F_acc, torch.cat([trace, fro], dim=1)  # (batch,2)


# ---------------------------
# HermiteNetwork with enriched head (uses J summary)
# ---------------------------

class HermiteNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 maps_A: int = 1,
                 maps_B: int = 0,
                 hermite_degree: int = 5,
                 hermite_version: str = "probabilist",
                 learn_coeffs: bool = True,
                 coeffs: Optional[np.ndarray] = None,
                 out_dim: int = 1):
        super().__init__()
        self.symblock = SymmetricHermiteBlock(input_dim=input_dim,
                                              maps_A=maps_A,
                                              maps_B=maps_B,
                                              hidden_dim=None,
                                              hermite_degree=hermite_degree,
                                              hermite_version=hermite_version,
                                              learn_coeffs=learn_coeffs,
                                              coeffs=coeffs)
        # features: s (d) + F_s (d) + J_summary (2) -> total = 2*d + 2
        feat_dim = input_dim * 2 + 2
        self.W = nn.Linear(feat_dim, out_dim, bias=True)
        self.last_jacobian = None

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (batch, d)
        F_s, J_repr = self.symblock(s, j_mode="summary")  # F_s: (b,d), J_repr: (b,2)
        features = torch.cat([s, F_s, J_repr], dim=1)     # (b, 2d+2)
        out = self.W(features)                            # (b, out_dim)
        return out

    def summary(self):
        print("HermiteNetwork summary:")
        print(self)


# ---------------------------
# Data / Sliding windows (returns)
# ---------------------------

def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    # returns length T-1: r[t] = ln(p[t+1]/p[t])
    eps = 1e-12
    return np.log(prices[1:] + eps) - np.log(prices[:-1] + eps)

def create_sliding_windows_returns(returns: np.ndarray, window_len: int, horizon: int):
    """
    returns: 1D array length R (R = T-1)
    returns:
      X: (n_windows, window_len)
      Y: (n_windows, horizon)
    where n_windows = R - window_len - horizon + 1
    """
    X, Y = [], []
    L = len(returns)
    for i in range(L - window_len - horizon + 1):
        X.append(returns[i:i + window_len])
        Y.append(returns[i + window_len:i + window_len + horizon])
    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)
    return X, Y

def load_bitcoin_returns(csv_path: str,
                         window: int,
                         horizon: int,
                         close_col: str = "Close",
                         date_col: Optional[str] = "Open time",
                         dropna: bool = True):
    """
    Load CSV, compute log returns, scale returns, build sliding windows.
    Returns:
      (X_train,Y_train), (X_test,Y_test),
      scaler (fitted on returns),
      closes (original price series),
      dates (array of datetimes or None),
      prev_closes_train, prev_closes_test,
      target_dates_test (dates for the first horizon target, or None)
    """
    df = pd.read_csv(csv_path)
    if dropna:
        df = df.dropna().reset_index(drop=True)

    if close_col not in df.columns:
        raise ValueError(f"Close column '{close_col}' not found in CSV.")

    dates = None
    if date_col is not None and date_col in df.columns:
        # parse datetimes robustly
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        dates = df[date_col].values

    closes = df[close_col].values.astype(np.float64)
    if len(closes) < (window + horizon + 2):
        raise ValueError("Not enough rows in CSV for requested window + horizon.")

    # compute log returns length T-1
    returns = compute_log_returns(closes)  # shape (T-1,)

    # scale returns
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns.reshape(-1, 1)).flatten()

    # sliding windows over scaled returns
    X, Y = create_sliding_windows_returns(returns_scaled, window, horizon)
    n_windows = X.shape[0]

    # prev_closes for each window: price index = i + window
    prev_indices = np.arange(window, window + n_windows)  # indexes into closes (0-indexed)
    prev_closes_all = closes[prev_indices]                # (n_windows,)
    # target date for first horizon: index prev_index + 1 (p at t+1)
    if dates is not None:
        target_dates_all = dates[prev_indices + 1]
    else:
        target_dates_all = None

    # train/test split
    split = int(0.8 * n_windows)
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]
    prev_closes_train = prev_closes_all[:split]
    prev_closes_test = prev_closes_all[split:]
    target_dates_test = target_dates_all[split:] if target_dates_all is not None else None

    return (X_train, Y_train), (X_test, Y_test), scaler, closes, dates, prev_closes_train, prev_closes_test, target_dates_test


# ---------------------------
# Training / Evaluate / Metrics
# ---------------------------

def train_model(model: nn.Module,
                train_loader: DataLoader,
                epochs: int,
                lr: float,
                device: torch.device,
                loss_fn: Optional[nn.Module] = None,
                print_every: int = 100):
    if loss_fn is None:
        loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)
        avg_loss = total_loss / (total_count + 1e-12)
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}  Train loss: {avg_loss:.6e}")
    return model

def evaluate_model(model: nn.Module, X_test: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(Xt).cpu().numpy()
    return preds  # scaled returns (as model was trained on scaled returns)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def directional_accuracy_with_prev(prev_closes: np.ndarray, y_true_prices: np.ndarray, y_pred_prices: np.ndarray) -> float:
    """
    prev_closes: (n_samples,)
    y_true_prices: (n_samples, horizon) in price units
    y_pred_prices: (n_samples, horizon)
    Compare sign of (first horizon price - prev_close)
    """
    true_dir = np.sign(y_true_prices[:, 0] - prev_closes)
    pred_dir = np.sign(y_pred_prices[:, 0] - prev_closes)
    correct = (true_dir == pred_dir).astype(float)
    return float(np.nanmean(correct))

def sharpe_ratio_from_pred_returns(pred_returns: np.ndarray, freq_per_year: float = 365.0) -> float:
    """
    pred_returns: (n_samples,) or (n_samples, horizon) -> we use first column
    """
    if pred_returns.ndim > 1:
        r = pred_returns[:, 0]
    else:
        r = pred_returns
    mean_r = np.nanmean(r)
    std_r = np.nanstd(r) + 1e-12
    return float((mean_r / std_r) * math.sqrt(freq_per_year))


# ---------------------------
# Plot utilities
# ---------------------------

def plot_predictions_indexed(x_axis, true_prices: np.ndarray, pred_prices: np.ndarray, title: str = "Prediction (first horizon)"):
    """
    x_axis: either numeric indices or array-like datetimes
    true_prices, pred_prices: (n_samples, horizon) in price units
    Plot only first horizon for clarity.
    """
    first_true = true_prices[:, 0]
    first_pred = pred_prices[:, 0]
    plt.figure(figsize=(12,5))
    if np.issubdtype(type(x_axis[0]), np.datetime64) or isinstance(x_axis[0], (pd.Timestamp, np.datetime64)):
        # x_axis are datetimes
        plt.plot(x_axis, first_true, label="True t+1")
        plt.plot(x_axis, first_pred, label="Pred t+1", linestyle="--")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
    else:
        plt.plot(x_axis, first_true, label="True t+1")
        plt.plot(x_axis, first_pred, label="Pred t+1", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_horizons_indexed(x_axis, true_prices: np.ndarray, pred_prices: np.ndarray, title: str = "Predictions (all horizons)"):
    plt.figure(figsize=(12,6))
    H = true_prices.shape[1]
    alpha = 0.9
    for h in range(H):
        label_t = f"True t+{h+1}"
        label_p = f"Pred t+{h+1}"
        plt.plot(x_axis, true_prices[:, h], label=label_t, alpha=alpha)
        plt.plot(x_axis, pred_prices[:, h], linestyle="--", label=label_p, alpha=alpha)
    if np.issubdtype(type(x_axis[0]), np.datetime64) or isinstance(x_axis[0], (pd.Timestamp, np.datetime64)):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[12]:


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    # -----------------------
    # User variables (edit these)
    # -----------------------
    csv_path = "/home/francisco/MEGA/work/trading/PRICE_HISTORY/BTCUSD/btc_1d_data_2018_to_2025.csv"   # <-- set this to your CSV file path
    close_col = "Close"
    date_col = "Open time"   # set to None to ignore dates
    window = 64              # sliding window length (in returns)
    horizon = 3              # predict next N returns (1 <= horizon < 15)
    assert 1 <= horizon < 15, "horizon must be between 1 and 14 inclusive"

    batch_size = 64
    epochs = 5000
    lr = 1e-5

    hermite_degree = 8
    maps_A = 2
    maps_B = 1
    learn_coeffs = True

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------
    # Load data
    # -----------------------
    print("Loading data from:", csv_path)
    (X_train, Y_train), (X_test, Y_test), scaler, closes, dates, prev_closes_train, prev_closes_test, target_dates_test = \
        load_bitcoin_returns(csv_path=csv_path,
                             window=window,
                             horizon=horizon,
                             close_col=close_col,
                             date_col=date_col,
                             dropna=True)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    print(f"Prepared windows: train {n_train}, test {n_test} (window={window}, horizon={horizon})")

    # DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # -----------------------
    # Initialize model
    # -----------------------
    model = HermiteNetwork(
        input_dim=window,
        maps_A=maps_A,
        maps_B=maps_B,
        hermite_degree=hermite_degree,
        hermite_version="probabilist",
        learn_coeffs=learn_coeffs,
        coeffs=None,
        out_dim=horizon
    ).to(device)

    print("Model initialized.")
    model.summary()

    # -----------------------
    # Train
    # -----------------------
    model = train_model(model, train_loader, epochs, lr, device, loss_fn=nn.HuberLoss(delta=1.0), print_every=100)

    # -----------------------
    # Evaluate (scaled returns), invert scaling to returns, reconstruct prices
    # -----------------------
    preds_scaled = evaluate_model(model, X_test, device)  # shape (n_test, horizon) scaled returns
    # inverse transform per-element
    preds_returns = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)  # (n_test,horizon)
    true_returns = scaler.inverse_transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)

    # Reconstruct predicted prices from prev_closes_test and predicted returns
    # vectorized: cumulative product of exp(returns) times prev_close
    exp_preds = np.exp(preds_returns)  # (n_test, horizon)
    cumprod_preds = np.cumprod(exp_preds, axis=1)  # cumulative multiplicative factor relative to prev_close
    preds_prices = (prev_closes_test.reshape(-1, 1)) * cumprod_preds  # (n_test, horizon)

    # Reconstruct true prices similarly
    exp_trues = np.exp(true_returns)
    cumprod_trues = np.cumprod(exp_trues, axis=1)
    true_prices = (prev_closes_test.reshape(-1, 1)) * cumprod_trues

    # -----------------------
    # Metrics (prices)
    # -----------------------
    mse_price = mse(true_prices, preds_prices)
    rmse_price = rmse(true_prices, preds_prices)
    dir_acc = directional_accuracy_with_prev(prev_closes_test, true_prices, preds_prices)
    # compute freq_per_year from dates if available
    if target_dates_test is not None and len(target_dates_test) >= 2:
        # compute median delta days
        deltas = np.diff(pd.to_datetime(target_dates_test)).astype('timedelta64[s]').astype(np.float64)
        median_seconds = np.median(deltas)
        median_days = median_seconds / 86400.0 if median_seconds > 0 else 1.0
        freq_per_year = 365.0 / max(median_days, 1e-6)
    else:
        freq_per_year = 365.0
    sharpe = sharpe_ratio_from_pred_returns(preds_returns[:, 0], freq_per_year=freq_per_year)

    print("\n--- Evaluation (test set) ---")
    print(f"MSE (price units, all horizons): {mse_price:.6f}")
    print(f"RMSE (price units, all horizons): {rmse_price:.6f}")
    print(f"Directional accuracy (first horizon vs prev): {dir_acc * 100:.2f}%")
    print(f"Predicted returns Sharpe-like (annualized): {sharpe:.4f}")

    # -----------------------
    # Save predictions CSV (prices)
    # -----------------------
    out_dict = {"prev_close": prev_closes_test}
    for h in range(horizon):
        out_dict[f"true_tplus{h+1}"] = true_prices[:, h]
    for h in range(horizon):
        out_dict[f"pred_tplus{h+1}"] = preds_prices[:, h]

    out_df = pd.DataFrame(out_dict)
    out_csv = "hermite_btc_predictions_1.csv"
    out_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Predictions saved to {out_csv}")

    # -----------------------
    # Plot (use dates if available)
    # -----------------------
    if target_dates_test is not None:
        x_axis = pd.to_datetime(target_dates_test)
    else:
        x_axis = np.arange(n_test)

    # first-horizon plot
    plot_predictions_indexed(x_axis, true_prices, preds_prices,
                             title=f"BTC Close predictions (first horizon out of {horizon})")
    # all horizons plot (optional)
    plot_all_horizons_indexed(x_axis, true_prices, preds_prices,
                              title=f"BTC Close predictions (all horizons, horizon={horizon})")

    # -----------------------
    # Print Hermite coefficients (if learnable)
    # -----------------------
    try:
        cvals = model.symblock.hermite.c.detach().cpu().numpy()
        print("Hermite coefficients c_n:", cvals)
    except Exception:
        print("Hermite coefficients not available.")

    print("\nModel structure:")
    print(model)
    try:
        model.summary()
    except Exception:
        pass

    # End of script


# In[ ]:





# In[ ]:





# In[ ]:




