"""
PART 5 (Python) - LSTM and GRU with walk-forward validation, evaluated on rolling test forecasts
Run from the PROJECT ROOT:  python python/05_lstm_gru.py
Quick smoke test (3 min):   QUICK=1 python python/05_lstm_gru.py      (Git Bash)   |   set QUICK=1  then run (cmd)
Needs: data/prices.csv (python/01) OR data/prices_R.csv (R/01). pip install tensorflow pandas numpy matplotlib statsmodels
"""
import os, random, time
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # silences the oneDNN notice, makes CPU numerics more reproducible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")    # hides TensorFlow info messages
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from statsmodels.stats.diagnostic import acorr_ljungbox

# ------------------------------------------------------------------ SETTINGS (each one justified)
SEED, H = 42, 10                                       # H=10: longest horizon in the brief (5..10)
QUICK = os.environ.get("QUICK", "0") == "1"
K_FOLDS = 3 if QUICK else 4                            # walk-forward folds inside the TRAIN period
WINDOW_GRID = [20] if QUICK else [20, 60]              # 20 d ~ 1 trading month; 60 d ~ 1 quarter. The GARCH half-life of a volatility shock was 21 d, so memory beyond ~60 d is unlikely to matter
UNITS_GRID = [16] if QUICK else [16, 32]               # small nets: only ~1,700 training windows -> large nets overfit
N_LAYERS = 1                                           # 1 recurrent layer: depth buys nothing at this sample size (can be changed and re-run as an ablation)
DROPOUT = 0.2                                          # regularisation against overfitting a very noisy target
MAX_EPOCHS, BATCH, PATIENCE = (30 if QUICK else 80), 32, 8   # early stopping decides the real number of epochs
os.makedirs("data", exist_ok=True); os.makedirs("figures", exist_ok=True)
print("tensorflow", tf.__version__, "| numpy", np.__version__, "| pandas", pd.__version__, "| QUICK =", QUICK)

def seed_all():
    random.seed(SEED); np.random.seed(SEED); keras.utils.set_random_seed(SEED)

# ------------------------------------------------------------------ 1. DATA (same alignment and 90/10 split as the R scripts)
def load_prices():
    if os.path.exists("data/prices.csv"):
        d = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)[["AdjClose_AAPL", "AdjClose_XLK"]]
        d.columns = ["AAPL", "XLK"]
    else:
        d = pd.read_csv("data/prices_R.csv", index_col=0, parse_dates=True)[["AAPL", "XLK"]]
    return d.dropna()
px = load_prices()
lpf = np.log(px.values)                                # log prices, n+1 rows
R = np.diff(lpf, axis=0); dates = px.index[1:]         # log returns, n rows (return i = change during day i)
n = len(R); cut = int(0.9 * n)                         # train = returns 0..cut-1, test = cut..n-1
r_a = R[:, 0]
# Inputs: AAPL return, XLK return (the multivariate part of the project), 10-day rolling volatility of AAPL (uses only past data)
vol10 = pd.Series(r_a).rolling(10, min_periods=1).std().fillna(0).values
F = np.column_stack([R[:, 0], R[:, 1], vol10])
print(f"n={n} train={cut} test={n-cut} | features: AAPL return, XLK return, 10-d volatility")
# TARGET = the next H daily returns (not prices): prices are non-stationary and a network cannot extrapolate beyond the
# price range it saw in training; returns are stationary. Price forecasts are rebuilt by cumulating predicted returns.

def make_samples(origins, W):
    """origin t = last observed day: input = features of days t-W+1..t, target = returns of days t+1..t+H."""
    X = np.stack([F[t - W + 1:t + 1] for t in origins]); Y = np.stack([r_a[t + 1:t + 1 + H] for t in origins])
    return X, Y

class Scaler:                                          # always fitted on the TRAINING part only (no look-ahead)
    def __init__(s, F_tr, r_tr): s.mu, s.sd, s.ysd = F_tr.mean(0), F_tr.std(0) + 1e-8, r_tr.std() + 1e-12
    def x(s, X): return (X - s.mu) / s.sd
    def y(s, Y): return Y / s.ysd

# ------------------------------------------------------------------ 2. MODEL
def build(arch, W, units):
    cell = L.LSTM if arch == "LSTM" else L.GRU         # the ONLY difference between the two models
    m = keras.Sequential([keras.Input((W, F.shape[1]))])
    for i in range(N_LAYERS):
        m.add(cell(units, dropout=DROPOUT, return_sequences=(i < N_LAYERS - 1)))
    m.add(L.Dropout(DROPOUT)); m.add(L.Dense(H))       # one output per forecast horizon (direct multi-horizon forecast)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m

def fit_rnn(arch, W, units, Xs, Ys):
    seed_all(); keras.backend.clear_session()
    m = build(arch, W, units)
    n_val = int(0.15 * len(Xs)); tr_end = len(Xs) - n_val - H   # validation = LAST 15% in time, with a gap of H so target windows do not overlap
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    hist = m.fit(Xs[:tr_end], Ys[:tr_end], validation_data=(Xs[-n_val:], Ys[-n_val:]),
                 epochs=MAX_EPOCHS, batch_size=BATCH, callbacks=[es], verbose=0)   # shuffling windows INSIDE the train block is not look-ahead
    return m, hist.history

# ------------------------------------------------------------------ 3. WALK-FORWARD (expanding-window) CROSS-VALIDATION inside the training period
# Never k-fold: each fold trains on the past only and is scored on the block right after it.
def cv_config(arch, W, units):
    n0 = int(0.5 * cut); B = (cut - n0) // K_FOLDS; out = []
    for k in range(K_FOLDS):
        fs, fe = n0 + k * B, n0 + (k + 1) * B                 # fold k: train on days < fs, evaluate on days fs..fe-1
        sc = Scaler(F[:fs], r_a[:fs])
        Xtr, Ytr = make_samples(np.arange(W - 1, fs - H), W)  # targets end at fs-1: nothing from the evaluation block
        Xev, Yev = make_samples(np.arange(fs - 1, fe - H), W) # first target = day fs
        m, _ = fit_rnn(arch, W, units, sc.x(Xtr), sc.y(Ytr))
        P = m.predict(sc.x(Xev), verbose=0); Ye = sc.y(Yev)
        mse = np.mean((P - Ye) ** 2)
        base = np.mean((sc.y(Ytr).mean(0) - Ye) ** 2)         # benchmark: always predict the average training return
        out.append(dict(arch=arch, W=W, units=units, fold=k, mse=mse, base_mse=base, ratio=mse / base))
    return out

t0 = time.time(); rows = []
for arch in ["LSTM", "GRU"]:
    for W in WINDOW_GRID:
        for u in UNITS_GRID:
            rows += cv_config(arch, W, u); print(f"  CV done: {arch} W={W} units={u}  ({time.time()-t0:.0f}s)")
cv = pd.DataFrame(rows); cv.to_csv("data/rnn_cv_folds_py.csv", index=False)
summ = cv.groupby(["arch", "W", "units"]).agg(mse=("mse", "mean"), ratio=("ratio", "mean"), ratio_sd=("ratio", "std")).reset_index()
print("\nWalk-forward CV (scaled MSE; ratio = model MSE / 'predict the average return' MSE; ratio < 1 means the net beats the benchmark):")
print(summ.round(4).to_string(index=False)); summ.to_csv("data/rnn_cv_summary_py.csv", index=False)

# ------------------------------------------------------------------ 4. FINAL MODELS: best config per architecture, trained on the whole training period
sc = Scaler(F[:cut], r_a[:cut]); test_orig = np.arange(cut - 1, n - H)       # rolling forecast origins across the test set
hists, preds = {}, {}
for arch in ["LSTM", "GRU"]:
    b = summ[summ.arch == arch].sort_values("mse").iloc[0]; W, u = int(b.W), int(b.units)
    print(f"\n{arch}: chosen window={W}, units={u}")
    Xtr, Ytr = make_samples(np.arange(W - 1, cut - H), W)
    m, hists[arch] = fit_rnn(arch, W, u, sc.x(Xtr), sc.y(Ytr))
    print(f"  parameters: {m.count_params()} | epochs run: {len(hists[arch]['loss'])} | best val loss: {min(hists[arch]['val_loss']):.4f}")
    Xte, _ = make_samples(test_orig, W)            # the model is NOT refitted on test data; its inputs are the actual past returns at each origin
    preds[arch] = m.predict(sc.x(Xte), verbose=0) * sc.ysd   # back to raw log returns

# ------------------------------------------------------------------ 5. LOSS CURVES (training procedure documentation)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, arch in zip(ax, hists):
    a.plot(hists[arch]["loss"], label="train"); a.plot(hists[arch]["val_loss"], label="validation")
    a.set_title(f"{arch}: loss (MSE, scaled)"); a.set_xlabel("epoch"); a.legend()
plt.tight_layout(); plt.savefig("figures/06_rnn_loss.png", dpi=130); plt.close()

# ------------------------------------------------------------------ 6. ROLLING TEST FORECASTS -> price scale, same metrics as the R scripts
lp_t = lpf[test_orig + 1, 0]                                   # log price at each origin
steps = np.arange(1, H + 1)
actual = np.stack([lpf[test_orig + 1 + h, 0] for h in steps], axis=1)
mu = r_a[:cut].mean()
fc = {"lstm_lp": lp_t[:, None] + np.cumsum(preds["LSTM"], axis=1),
      "gru_lp": lp_t[:, None] + np.cumsum(preds["GRU"], axis=1),
      "rw0_lp": np.repeat(lp_t[:, None], H, axis=1),           # random walk without drift
      "rwdrift_lp": lp_t[:, None] + mu * steps}                # random walk with the TRAIN drift
def metrics(y_lp, p_lp):
    y, p = np.exp(y_lp), np.exp(p_lp); e = y - p
    return dict(MSE=np.mean(e**2), RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), MAPE=100 * np.mean(np.abs(e / y)))
tab = pd.DataFrame({(name.replace("_lp", ""), f"h={h}"): metrics(actual[:, h-1], fc[name][:, h-1])
                    for name in fc for h in (1, 5, 10)}).T
print(f"\nRolling forecasts over {len(test_orig)} test origins (price scale):\n", tab.round(4)); tab.to_csv("data/metrics_rnn_py.csv")

# One long CSV, one row per (origin, horizon): the R evaluation script merges ARIMA / GARCH forecasts on the same origins
long = pd.DataFrame({"origin": np.repeat(np.arange(len(test_orig)), H), "origin_date": np.repeat(dates[test_orig].astype(str), H),
                     "h": np.tile(steps, len(test_orig)), "actual_lp": actual.ravel(), **{k: v.ravel() for k, v in fc.items()}})
long.to_csv("data/forecast_rnn_py.csv", index=False)

# ------------------------------------------------------------------ 7. DOES THE NETWORK ACTUALLY HAVE SKILL? (1-day-ahead returns)
a1 = r_a[test_orig + 1]
print("\n1-day-ahead return check on the test set:")
for arch in ["LSTM", "GRU"]:
    p1 = preds[arch][:, 0]; err = a1 - p1
    print(f"  {arch}: corr(pred, actual) = {np.corrcoef(p1, a1)[0,1]:.3f} | direction hit-rate = {np.mean(np.sign(p1) == np.sign(a1)):.1%}"
          f" | Ljung-Box p on errors (lag 10) = {acorr_ljungbox(err, lags=[10])['lb_pvalue'].iloc[0]:.3f}")
print(f"  share of up-days in the test set = {np.mean(a1 > 0):.1%} (a hit-rate near this value means no directional skill)")
fig, ax = plt.subplots(figsize=(10, 3.5)); ax.plot(a1, lw=.6, label="actual"); ax.plot(preds["LSTM"][:, 0], label="LSTM"); ax.plot(preds["GRU"][:, 0], label="GRU")
ax.set_title("1-day-ahead return forecasts, test set"); ax.legend(); plt.tight_layout(); plt.savefig("figures/07_rnn_h1.png", dpi=130); plt.close()
print(f"\nTotal time: {time.time()-t0:.0f}s. Saved data/forecast_rnn_py.csv, data/metrics_rnn_py.csv, figures/06_rnn_loss.png, 07_rnn_h1.png")
