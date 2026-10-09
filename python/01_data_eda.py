"""
PART 1 (Python) - Data import, preparation and exploratory analysis
Pair: AAPL (stock) + XLK (its sector ETF). Daily, 2019-01-01 -> 2026-09-30.
Install: pip install yfinance pandas numpy matplotlib statsmodels scipy
Run (from the PROJECT ROOT): python python/01_data_eda.py
"""
import os, random, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed: figures are saved to files
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", category=UserWarning)

# Fixed seed + fixed date window: same results on every run (reproducibility criterion)
SEED = 42; random.seed(SEED); np.random.seed(SEED)
TICKERS, START, END = ["AAPL", "XLK"], "2019-01-01", "2026-09-30"
os.makedirs("data", exist_ok=True); os.makedirs("figures", exist_ok=True)
print("statsmodels", statsmodels.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)

# ---------------------------------------------------------------- 1. IMPORT
# Cache to CSV so the notebook re-runs offline and the data cannot silently change.
CSV = "data/prices.csv"
if os.path.exists(CSV):
    df = pd.read_csv(CSV, index_col=0, parse_dates=True)
else:
    parts = []
    for t in TICKERS:
        # auto_adjust=False keeps BOTH Close and Adj Close (dividend/split adjusted by Yahoo)
        d = yf.download(t, start=START, end=END, auto_adjust=False, progress=False)
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]  # flatten MultiIndex
        d.columns = [f"{c.replace(' ', '')}_{t}" for c in d.columns]
        parts.append(d)
    df = pd.concat(parts, axis=1).dropna()  # inner alignment: same trading days for both assets
    df.index = pd.DatetimeIndex(df.index, name="Date")
    df.to_csv(CSV)
print(df.shape, df.index.min().date(), "->", df.index.max().date())

# Frequency note: trading days are NOT a regular calendar (weekends + holidays missing).
# We keep the real trading-day DatetimeIndex rather than forward-filling holidays,
# because filling would create fake zero returns and bias volatility downward.

# ------------------------------------------------- 2. DERIVED INDICATORS
for t in TICKERS:
    px = df[f"AdjClose_{t}"]
    df[f"lp_{t}"] = np.log(px)                                   # log price: stabilises variance, additive returns
    df[f"r_{t}"] = df[f"lp_{t}"].diff()                          # log return: the stationary object used for GARCH
    df[f"ma20_{t}"] = px.rolling(20).mean()                      # 20-day moving average (trend indicator)
    df[f"rv20_{t}"] = df[f"r_{t}"].rolling(20).std() * np.sqrt(252)  # annualised realised volatility
df = df.dropna(subset=[f"r_{t}" for t in TICKERS])

# --------------------------------------------- 3. CHRONOLOGICAL 90/10 SPLIT
n = len(df); cut = int(0.9 * n)
train, test = df.iloc[:cut], df.iloc[cut:]
print(f"Train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} obs)")
print(f"Test : {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} obs)")
# ALL exploration/identification below uses TRAIN ONLY: looking at the test set would leak information.
tr = train
MAIN = "AAPL"

# -------------------------------------------------- 4. EXPLORATORY PLOTS
fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for t in TICKERS:  # rebased to 100 so two different price scales are comparable
    ax[0].plot(tr.index, 100 * tr[f"AdjClose_{t}"] / tr[f"AdjClose_{t}"].iloc[0], label=t)
ax[0].plot(tr.index, 100 * tr[f"ma20_{MAIN}"] / tr[f"AdjClose_{MAIN}"].iloc[0], "k--", lw=.8, label=f"MA20 {MAIN}")
ax[0].set_title("Adjusted close (base 100)"); ax[0].legend()
for t in TICKERS: ax[1].plot(tr.index, tr[f"r_{t}"], lw=.6, label=t)
ax[1].set_title("Log returns"); ax[1].legend()
for t in TICKERS: ax[2].plot(tr.index, tr[f"rv20_{t}"], label=t)
ax[2].set_title("20-day realised volatility (annualised)"); ax[2].legend()
plt.tight_layout(); plt.savefig("figures/01_overview.png", dpi=130); plt.close()
print("Correlation of returns (train):", round(tr["r_AAPL"].corr(tr["r_XLK"]), 3))

# ------------------------------------------------ 5. ACF / PACF
# Raw log price (expect slowly decaying ACF = unit root) vs first difference (expect ~ white noise).
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
x_lvl, x_ret = tr[f"lp_{MAIN}"], tr[f"r_{MAIN}"]
plot_acf(x_lvl, lags=40, ax=ax[0, 0], title="ACF log price");  plot_pacf(x_lvl, lags=40, ax=ax[0, 1], title="PACF log price", method="ywm")
plot_acf(x_ret, lags=40, ax=ax[1, 0], title="ACF log return"); plot_pacf(x_ret, lags=40, ax=ax[1, 1], title="PACF log return", method="ywm")
plt.tight_layout(); plt.savefig("figures/02_acf_pacf.png", dpi=130); plt.close()

# -------------------------------------------------- 6. STL DECOMPOSITION
# period=252 ~ one trading year. STL (not classical) is robust to outliers such as March 2020.
stl = STL(x_lvl.reset_index(drop=True), period=252, robust=True).fit()
fig = stl.plot(); fig.set_size_inches(11, 7); plt.tight_layout()
plt.savefig("figures/03_stl.png", dpi=130); plt.close()
print("STL: share of variance of detrended series explained by seasonal component =",
      round(1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal), 3))

# ----------------------------------------------------- 7. OUTLIERS
q1, q3 = x_ret.quantile([.25, .75]); iqr = q3 - q1
mask = (x_ret < q1 - 3 * iqr) | (x_ret > q3 + 3 * iqr)  # 3*IQR = "extreme" fence (1.5 would flag ~8% of fat-tailed returns)
print(f"\nExtreme returns (3*IQR rule): {mask.sum()} of {len(x_ret)}")
print(x_ret[mask].sort_values(key=abs, ascending=False).head(10).round(4))
# Treatment decision (to justify in the report): KEEP. These are genuine market events (e.g. Covid crash
# March 2020), not data errors; removing them would erase exactly the volatility GARCH must model.
# Check that none is a data glitch (e.g. an unadjusted split) before keeping.

# ------------------------------------- 8. DESCRIPTIVE STATS (reused for LLM section 5)
desc = pd.DataFrame({t: {
    "mean": tr[f"r_{t}"].mean(), "variance": tr[f"r_{t}"].var(),
    "skewness": stats.skew(tr[f"r_{t}"]), "excess_kurtosis": stats.kurtosis(tr[f"r_{t}"]),
    "JarqueBera_p": stats.jarque_bera(tr[f"r_{t}"])[1]} for t in TICKERS})
print("\nDescriptive statistics of log returns (train):\n", desc.round(6))
desc.to_csv("data/descriptive_stats.csv")

# ---------------------------------------- 9. STATIONARITY: ADF + KPSS
def stationarity(x, name, reg="c"):
    adf_p = adfuller(x.dropna(), regression=reg, autolag="AIC")[1]   # H0: unit root (non-stationary)
    kpss_p = kpss(x.dropna(), regression=reg, nlags="auto")[1]       # H0: stationary (opposite!)
    if adf_p < .05 and kpss_p >= .05: verdict = "STATIONARY (both agree)"
    elif adf_p >= .05 and kpss_p < .05: verdict = "NON-STATIONARY (both agree)"
    else: verdict = "INCONCLUSIVE / conflicting -> inspect, try differencing or a trend term"
    print(f"{name:<14} ADF p={adf_p:.4f} | KPSS p={kpss_p:.4f} (p reported is capped at 0.01-0.10) -> {verdict}")
print("\nStationarity tests (train):")
for t in TICKERS:
    stationarity(tr[f"lp_{t}"], f"log price {t}", reg="ct")  # 'ct' = constant + trend, plausible for a drifting price
    stationarity(tr[f"r_{t}"], f"log return {t}")
# Expected: log price I(1) -> d = 1; log return stationary -> d = 0 on returns.
# Write the conclusion in the report: "required order of differencing d = 1 on log prices".

df.to_csv("data/prepared.csv"); print("\nSaved data/prepared.csv, figures/*.png")
