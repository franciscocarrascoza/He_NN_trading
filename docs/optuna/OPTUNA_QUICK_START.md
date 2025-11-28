# Optuna Optimization - Quick Start Guide

## ✅ CORRECTED: Using `conda activate binance_env`

The environment **binance_env** exists and is activated using standard conda commands.

---

## 🚀 Quick Start (Copy-Paste Ready)

### Option 1: Run with tmux (Recommended)

```bash
# In your terminal (NOT Claude Code)
cd /home/francisco/work/AI/He_NN_trading

# Start tmux session
tmux new -s optuna

# Inside tmux, activate and run
conda activate binance_env
python optuna_optimize_v2.py

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t optuna
```

### Option 2: Run with nohup (Background)

```bash
cd /home/francisco/work/AI/He_NN_trading
conda activate binance_env
nohup python optuna_optimize_v2.py > optuna_v2.log 2>&1 &

# Check progress
tail -f optuna_v2.log

# Or check PID
ps aux | grep optuna_optimize_v2
```

### Option 3: Simple Terminal Run

```bash
cd /home/francisco/work/AI/He_NN_trading
conda activate binance_env
python optuna_optimize_v2.py
```

---

## 📊 Monitor Progress (Separate Terminal)

```bash
# Activate environment
conda activate binance_env

# Launch dashboard
optuna-dashboard sqlite:///optuna_hermite_v2.db

# Open browser: http://localhost:8080
```

---

## ⏱️ Expected Runtime

- **150 trials** with **50 epochs** per trial
- **~6 hours total** (depends on GPU/CPU)
- **Resumable** if interrupted (progress saved in database)

---

## 📁 Output Files

After completion:
1. **optuna_hermite_v2.db** - All trial history
2. **best_config_optuna_v2.yaml** - Best hyperparameters
3. **optuna_runs/** - Trial artifacts

---

## 🔍 Check Progress Anytime

```bash
# In Python
conda activate binance_env
python -c "
import optuna
study = optuna.load_study('hermite_v2_comprehensive', 'sqlite:///optuna_hermite_v2.db')
print(f'Completed: {len([t for t in study.trials if t.state.name == \"COMPLETE\"])} trials')
print(f'Best score: {study.best_value:.4f}')
"
```

---

## ⚙️ What's Being Optimized

**31 hyperparameters** across:
- **Model architecture** (9 params): Hermite version, degree, maps, hidden dims, dropout, LSTM, prob source
- **Training dynamics** (13 params): Learning rate, batch size, loss weights, optimizer, scheduler, regularization
- **Data config** (2 params): Feature window, forecast horizon
- **Strategy execution** (7 params): Threshold, confidence margin, Kelly clip, conformal p-min, filters

**Objective**: Maximize composite score = 50% Sharpe + 25% AUC + 15% DirAcc + 10% (1-Brier)

---

## ✅ After Optimization Completes

### 1. View Results

```bash
# Show best config
cat best_config_optuna_v2.yaml
```

### 2. Run Final Training with Best Config

```bash
conda activate binance_env

# Full 5-fold cross-validation
python main.py --config best_config_optuna_v2.yaml --cv

# Or quick single-fold test
python main.py --config best_config_optuna_v2.yaml
```

### 3. Analyze Study

```bash
# Launch dashboard
optuna-dashboard sqlite:///optuna_hermite_v2.db

# Or generate plots in Python
python -c "
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

study = optuna.load_study('hermite_v2_comprehensive', 'sqlite:///optuna_hermite_v2.db')

# Save plots
plot_optimization_history(study).write_html('optuna_history.html')
plot_param_importances(study).write_html('optuna_importance.html')
print('✓ Plots saved: optuna_history.html, optuna_importance.html')
"
```

---

## 🛑 Stop Optimization

```bash
# If running in tmux
tmux attach -t optuna
# Then press Ctrl+C

# If running with nohup
ps aux | grep optuna_optimize_v2
kill <PID>

# Progress is automatically saved!
# Resume later by running the same command
```

---

## 📝 Resuming Interrupted Study

```bash
# Just run the same command - it auto-resumes from database
conda activate binance_env
python optuna_optimize_v2.py

# It will print: "Using an existing study with name 'hermite_v2_comprehensive'"
```

---

## 🔧 Troubleshooting

### "conda: command not found"

```bash
# Initialize conda
eval "$(conda shell.bash hook)"
conda activate binance_env
```

### "No module named 'optuna'"

```bash
conda activate binance_env
pip install optuna optuna-dashboard plotly
```

### "Trial failed" repeatedly

- Check data path: `ls ~/trading/he_nn_data/`
- Verify config: `cat config/app.yaml`
- Check logs: `tail optuna_runs/*/logs/*.log`

### Out of memory

Edit `optuna_optimize_v2.py` line 87:
```python
num_epochs=30  # Reduce from 50
```

And line 228:
```python
n_trials=50  # Reduce from 150
```

---

## 🎯 Ready to Start?

```bash
# COPY-PASTE THIS:
cd /home/francisco/work/AI/He_NN_trading && \
tmux new -s optuna -d "conda activate binance_env && python optuna_optimize_v2.py" && \
echo "✓ Optimization started in tmux session 'optuna'" && \
echo "  Attach: tmux attach -t optuna" && \
echo "  Detach: Ctrl+B, then D"
```

**Monitor dashboard:**
```bash
conda activate binance_env && optuna-dashboard sqlite:///optuna_hermite_v2.db
```

---

For full documentation, see: **OPTUNA_GUIDE.md**
