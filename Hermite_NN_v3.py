#!/usr/bin/env python
# coding: utf-8
Hermite-based NN architecture (PyTorch)

Implements the architecture described by the user:

Inputs (x₁,...,x_d, bias)
    ⟶ Linear maps: z = A_i x, B_i x
    ⟶ Activation: h(z) = Σ_{n=0}^N c_n Heₙ(z) + d₀ + d₁ z
       (coefficients c_n can be offline-estimated via quadrature or learned)
    ⟶ Symmetric feature transform:
        F(s) = Σ Aᵀ h(A s) − Σ Bᵀ h(B s) + b
    ⟶ Jacobian:
        J(s) = Σ Aᵀ diag(h′(A s)) A − Σ Bᵀ diag(h′(B s)) B
    ⟶ Final head:
        ŷ = W · [s x F(s) x J_feat(s)] + b

Features:
 - probabilists vs physicists Hermite polynomials
 - analytic derivative of Hermite basis (d/dx Heₙ = n He_{n-1} for probabilists,
   d/dx Hₙ = 2 n H_{n-1} for physicists)
 - offline quadrature estimator for c_n
 - options to keep c_n learnable or fixed
 - multiple A/B linear maps
 - compact J(s) summary (trace, frobenius, diag, or flattened)
 - toy functions: toy_poly, toy_qho, toy_moment, toy_dho
 - plotting utilities
# In[6]:


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.special import factorial
from typing import List, Optional, Tuple
import pandas as pd

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
# Quadrature coefficient estimator
# ---------------------------

def estimate_coeffs_quadrature(x_nodes: np.ndarray,
                               w_nodes: np.ndarray,
                               psi_vals: np.ndarray,
                               N: int,
                               version: str = "probabilist") -> np.ndarray:
    c = np.zeros(N+1)
    K = x_nodes.size
    He = np.zeros((N+1, K))
    He[0, :] = 1.0
    if N >= 1:
        He[1, :] = x_nodes if version == "probabilist" else 2.0 * x_nodes
    for n in range(1, N):
        if version == "probabilist":
            He[n+1, :] = x_nodes * He[n, :] - n * He[n-1, :]
        else:
            He[n+1, :] = 2.0 * x_nodes * He[n, :] - 2.0 * n * He[n-1, :]
    for n in range(N+1):
        c[n] = np.sum(w_nodes * psi_vals * He[n, :])
    return c


# ---------------------------
# Hermite Activation Module
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
            c_init = torch.tensor(coeffs, dtype=torch.get_default_dtype())
        else:
            c_init = 0.1 * torch.randn(self.N + 1)
        if learn_coeffs:
            self.c = nn.Parameter(c_init.clone())
        else:
            self.register_buffer("c", c_init.clone())
        self.register_buffer("d0", torch.tensor(float(d0)))
        self.register_buffer("d1", torch.tensor(float(d1)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        polys = hermite_polys(z, self.N, version=self.version)
        c_view = self.c.view(self.N+1, *([1]*z.dim()))
        return (c_view * polys).sum(dim=0) + self.d0 + self.d1 * z

    def derivative(self, z: torch.Tensor) -> torch.Tensor:
        if self.N == 0:
            return torch.ones_like(z) * 0.0 + self.d1
        polys = hermite_polys(z, self.N-1, version=self.version)
        factors = [hermite_derivative_coeff_factor(n, self.version) for n in range(1, self.N+1)]
        factors_t = torch.tensor(factors, dtype=self.c.dtype, device=self.c.device).view(-1, *([1]*z.dim()))
        c1 = self.c[1:].view(-1, *([1]*z.dim()))
        return (c1 * factors_t * polys).sum(dim=0) + self.d1


# ---------------------------
# Symmetric Feature Transform + Jacobian
# ---------------------------

class SymmetricHermiteBlock(nn.Module):
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
        self.hermite = HermiteActivation(degree=hermite_degree,
                                         in_features=self.hdim,
                                         learn_coeffs=learn_coeffs,
                                         coeffs=torch.tensor(coeffs) if coeffs is not None else None,
                                         d0=0.0, d1=0.0,
                                         version=hermite_version)
        self.register_parameter("bF", nn.Parameter(torch.zeros(self.d)))

    def forward(self, s: torch.Tensor, j_mode: str = "summary") -> Tuple[torch.Tensor, torch.Tensor]:
        batch = s.shape[0]
        F_acc = torch.zeros_like(s)
        J_acc = torch.zeros(batch, self.d, self.d, device=s.device)

        # Jacobian computation section

        for A in self.A_list:
            z = A(s)
            h_z = self.hermite(z)
            At = A.weight.t()
            F_acc += torch.matmul(h_z, At)
            hprime = self.hermite.derivative(z)
            M = A.weight
            M_exp = M.unsqueeze(0).expand(batch, -1, -1)
            Mh = M_exp * hprime.unsqueeze(2)
            J_acc += torch.matmul(M.transpose(0,1).unsqueeze(0).expand(batch, -1, -1), Mh)

        for B in self.B_list:
            z = B(s)
            h_z = self.hermite(z)
            Bt = B.weight.t()
            F_acc -= torch.matmul(h_z, Bt)
            hprime = self.hermite.derivative(z)
            M = B.weight
            M_exp = M.unsqueeze(0).expand(batch, -1, -1)
            Mh = M_exp * hprime.unsqueeze(2)
            J_acc -= torch.matmul(M.transpose(0,1).unsqueeze(0).expand(batch, -1, -1), Mh)

        F_acc = F_acc + self.bF.unsqueeze(0)

        if j_mode == "full":
            return F_acc, J_acc
        elif j_mode == "diag":
            return F_acc, torch.diagonal(J_acc, dim1=1, dim2=2)
        elif j_mode == "flatten":
            return F_acc, J_acc.view(batch, -1)
        else:
            trace = torch.einsum('bii->b', J_acc).unsqueeze(1)
            fro = torch.linalg.norm(J_acc.reshape(batch, -1), dim=1, keepdim=True)
            return F_acc, torch.cat([trace, fro], dim=1)


# ---------------------------
# Overall Final Model
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
        self.W = nn.Linear(1, out_dim, bias=True)
        self.last_jacobian = None # store last computed Jacobian
        
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        F_s, J_s = self.symblock(s, j_mode="full")
        self.last_jacobian = J_s.detach().cpu() # save for inspection/printing
        bilinear_term = torch.einsum("bi,bij,bj->b", s, J_s, F_s)
        return self.W(bilinear_term.unsqueeze(-1))


    def summary(self):
        print("HermiteNetwork summary:")
        print(self)
        if self.last_jacobian is not None:
            print("Last computed Jacobian (sample 0):")
            print(self.last_jacobian[0])


# ---------------------------
# Toy functions
# ---------------------------

def toy_poly(x: np.ndarray, coeffs: Optional[np.ndarray] = None, sigma: float = 1.0) -> np.ndarray:
    if coeffs is None:
        coeffs = np.array([1.0, -0.5, 0.2])
    poly = np.polyval(coeffs[::-1], x)
    gauss = np.exp(-(x**2)/(2.0*sigma**2))
    return poly * gauss


def toy_qho(x: np.ndarray, n: int = 0) -> np.ndarray:
    N = n
    K = x.shape[0]
    H = np.zeros((N+1, K))
    H[0, :] = 1.0
    if N >= 1:
        H[1, :] = 2.0 * x
    for k in range(1, N):
        H[k+1, :] = 2.0 * x * H[k, :] - 2.0 * k * H[k-1, :]
    Hn = H[N, :]
    norm = 1.0 / (np.pi**0.25 * np.sqrt((2**N) * factorial(N)))
    return norm * Hn * np.exp(-0.5*x**2)


def toy_moment(x: np.ndarray, order: int = 1) -> float:
    return float(np.mean(x**order))


def toy_dho(t: np.ndarray, A: float = 1.0, gamma: float = 0.1, omega0: float = 1.0, phi: float = 0.0) -> np.ndarray:
    omega_sq = omega0**2 - gamma**2
    if omega_sq > 0:
        omega_p = np.sqrt(omega_sq)
        return A * np.exp(-gamma*t) * np.cos(omega_p*t + phi)
    else:
        r1 = -gamma + np.sqrt(max(0.0, gamma**2 - omega0**2))
        r2 = -gamma - np.sqrt(max(0.0, gamma**2 - omega0**2))
        return A * (np.exp(r1*t) + np.exp(r2*t))/2.0


# ---------------------------
# Plotting utilities
# ---------------------------

def plot_time_series(t: np.ndarray, y: np.ndarray, title: str = "time series"):
    plt.figure(figsize=(8,3))
    plt.plot(t, y)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_predictions(t: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray, title: str = "prediction"):
    plt.figure(figsize=(8,3))
    plt.plot(t, y_true, label="true")
    plt.plot(t, y_hat, label="pred", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.grid(False)
    plt.savefig('prediction_.png')
    plt.show()

    # Save to CSV if filename is provided
    df = pd.DataFrame({
        "time": t,
        "y_true": y_true,
        "y_pred": y_hat
    })
    df.to_csv("./prediction_data.csv", index=False)


def plot_hermite_basis(x: np.ndarray, N: int = 5, version: str = "probabilist"):
    xs = torch.tensor(x, dtype=torch.get_default_dtype())
    polys = hermite_polys(xs, N, version=version).cpu().numpy()
    plt.figure(figsize=(8,4))
    for n in range(N+1):
        plt.plot(x, polys[n,:], label=f"{'He' if version=='probabilist' else 'H'}_{n}(x)")
    plt.legend()
    plt.title(f"Hermite basis ({version})")
    plt.grid(True)
    plt.ylim(-25, 25)   # restrict y-axis range
    plt.savefig('Hermite_basis_.png')
    plt.show()

# ---------------------------
#  CORE MODEL AND FUNCTIONS
# ---------------------------

def generate_synthetic_wave(t: np.ndarray):
    """A more complex synthetic function with step-wise changing amplitude/frequency."""
    freq0 = 0.5
    T0 = 2 * np.pi / freq0
    step_factor = ((t // (2 * T0)) % 2) * 0.5 + 3.0
    amp = (1.0 + 0.5 * np.sin(0.05 * t)) * step_factor
    freq = freq0 * (1.0 + 0.2 * ((t // (2 * T0)) % 3))
    y = amp * np.cos(2.0 * np.pi * freq * t) * np.exp(-0.001 * t)
    return y

def prepare_dataset(y: np.ndarray, window: int):
    X, Y = [], []
    for i in range(len(y) - window):
        X.append(y[i:i + window])
        Y.append(y[i + window])
    X = np.stack(X)
    Y = np.array(Y)
    split = int(X.shape[0] * 0.8)
    return (X[:split], Y[:split]), (X[split:], Y[split:])

def train_model(model, train_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_loader.dataset)
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}   train loss: {total_loss:.6f}")
    return model

def evaluate_model(model, X_test, Y_test, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        Yt = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1).to(device)
        Ypred = model(Xt).cpu().numpy().reshape(-1)
    return Ypred


# In[ ]:





# In[ ]:





# In[7]:


# ---------------------------
# Example usage and training loop
# ---------------------------

def example_usage():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)


    epochs = 1500
    lr = 1e-3
    n_samples = 5000
    window = 16
    input_dim = window
    batch_size = 64

    hermite_degree = 6
    hermite_version = "probabilist"
    
    
    #learn_coeffs → do we optimize the Hermite polynomial coefficients?

    #maps_A → how many positive projection/interaction channels?

    #maps_B → how many negative balancing channels?
    
    learn_coeffs = True
    maps_A = 2
    maps_B = 1
    out_dim = 1

    t = np.linspace(0, 30, n_samples)
    amp = 1.0 + 0.5*np.sin(0.05*t)
    freq = 1.0 + 0.2*np.sin(0.02*t+0.3)
    y = amp * np.cos(2*np.pi*freq*t) * np.exp(-0.001*t)

    ### A changing synthetic function
    # base frequency and period
    #freq0 = 1.0
    #T0 = 2 * np.pi / freq0
    
    # step function factor: changes every 2 base periods
    #step_factor = ((t // (2 * T0)) % 2) * 0.5 + 1.0
    # values alternate between 1.0 and 1.5 every 2 periods
    
    # amplitude and frequency now modulated step-wise
    #amp = (1.0 + 0.5 * np.sin(0.05 * t)) * step_factor
    #freq = freq0 * (1.0 + 0.2 * ((t // (2 * T0)) % 3))

    # final signal
    #y = amp * np.cos(2.0 * np.pi * freq * t) * np.exp(-0.001 * t)
    ####################################################################

    ### Other toy functions
    #y = toy_poly(t, coeffs=np.array([1.0, -0.5, 0.2]), sigma=1.0)
    # or
    #y = toy_qho(t, n=2)
    # or
    #y = toy_dho(t, A=1.0, gamma=0.05, omega0=1.2)
    

    X, Y = [], []
    for i in range(len(y)-window):
        X.append(y[i:i+window])
        Y.append(y[i+window])
    X = np.stack(X)
    Y = np.array(Y)

    split = int(X.shape[0]*0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                  torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    nodes = np.linspace(-3,3,81)
    weights = np.ones_like(nodes)*(nodes[1]-nodes[0])
    psi_on_nodes = toy_poly(nodes, coeffs=np.array([1.0,-0.2,0.05]), sigma=1.0)
    init_coeffs = estimate_coeffs_quadrature(nodes, weights, psi_on_nodes, hermite_degree, version=hermite_version)

    model = HermiteNetwork(input_dim=input_dim,
                           maps_A=maps_A,
                           maps_B=maps_B,
                           hermite_degree=hermite_degree,
                           hermite_version=hermite_version,
                           learn_coeffs=learn_coeffs,
                           coeffs=init_coeffs,
                           out_dim=out_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_loader.dataset)
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}   train loss: {total_loss:.6f}")

    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        Yt = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1).to(device)
        Ypred = model(Xt).cpu().numpy().reshape(-1)

    t_test = t[window+split:window+split+len(Y_test)]
    #plot_time_series(t_test, Y_test.flatten(), title="True target (test segment)")
    plot_predictions(t_test, Y_test.flatten(), Ypred, title="Symple Sinus Wave Function")
    plot_hermite_basis(np.linspace(-4,4,100), N=hermite_degree, version=hermite_version)

    print("Model summary:")
    print(model)
    print("Hermite coefficients c_n:", model.symblock.hermite.c.detach().cpu().numpy())
    return model, (X_test, Y_test, Ypred, t_test)


if __name__ == "__main__":
    example_usage()


# In[8]:


# ---------------------------
# Example usage and training loop
# ---------------------------

def example_usage():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)


    epochs = 1500
    lr = 1e-3
    n_samples = 5000
    window = 16
    input_dim = window
    batch_size = 64
    
    hermite_degree = 6
    hermite_version = "probabilist"
        
    #learn_coeffs → do we optimize the Hermite polynomial coefficients?
    #maps_A → how many positive projection/interaction channels?
    #maps_B → how many negative balancing channels?


    
    
    learn_coeffs = True
    maps_A = 2
    maps_B = 1
    out_dim = 1

    #t = np.linspace(0, 30, n_samples)
    #amp = 1.0 + 0.5*np.sin(0.05*t)
    #freq = 1.0 + 0.2*np.sin(0.02*t+0.3)
    #y = amp * np.cos(2*np.pi*freq*t) * np.exp(-0.001*t)

    ### A changing synthetic function
    t = np.linspace(0, 30, n_samples)

    # base frequency and period
    freq0 = 1.0
    T0 = 2 * np.pi / freq0
    
    # step function factor: changes every 2 base periods
    step_factor = ((t // (2 * T0)) % 2) * 0.5 + 1.0
    # values alternate between 2.0 and 1.5 every 2 periods
    
    # amplitude and frequency now modulated step-wise
    amp = (1.0 + 0.5 * np.sin(0.05 * t)) * step_factor
    freq = freq0 * (1.0 + 0.2 * ((t // (2 * T0)) % 3))

    # final signal
    y = amp * np.cos(2.0 * np.pi * freq * t) * np.exp(-0.001 * t)
    ####################################################################

    ### Other toy functions
    #y = toy_poly(t, coeffs=np.array([1.0, -0.5, 0.2]), sigma=1.0)
    # or
    #y = toy_qho(t, n=2)
    # or
    #y = toy_dho(t, A=1.0, gamma=0.05, omega0=1.2)
    

    X, Y = [], []
    for i in range(len(y)-window):
        X.append(y[i:i+window])
        Y.append(y[i+window])
    X = np.stack(X)
    Y = np.array(Y)

    split = int(X.shape[0]*0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                  torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    nodes = np.linspace(-3,3,81)
    weights = np.ones_like(nodes)*(nodes[1]-nodes[0])
    psi_on_nodes = toy_poly(nodes, coeffs=np.array([1.0,-0.2,0.05]), sigma=1.0)
    init_coeffs = estimate_coeffs_quadrature(nodes, weights, psi_on_nodes, hermite_degree, version=hermite_version)

    model = HermiteNetwork(input_dim=input_dim,
                           maps_A=maps_A,
                           maps_B=maps_B,
                           hermite_degree=hermite_degree,
                           hermite_version=hermite_version,
                           learn_coeffs=learn_coeffs,
                           coeffs=init_coeffs,
                           out_dim=out_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_loader.dataset)
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}   train loss: {total_loss:.6f}")

    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        Yt = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1).to(device)
        Ypred = model(Xt).cpu().numpy().reshape(-1)

    t_test = t[window+split:window+split+len(Y_test)]
    #plot_time_series(t_test, Y_test.flatten(), title="True target (test segment)")
    plot_predictions(t_test, Y_test.flatten(), Ypred, title="Symple Sinus Wave Function")
    plot_hermite_basis(np.linspace(-4,4,100), N=hermite_degree, version=hermite_version)

    print("Model summary:")
    print(model)
    print("Hermite coefficients c_n:", model.symblock.hermite.c.detach().cpu().numpy())
    return model, (X_test, Y_test, Ypred, t_test)


if __name__ == "__main__":
    example_usage()


# In[ ]:




