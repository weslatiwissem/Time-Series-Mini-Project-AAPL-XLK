"""
PART 2 (Python) - ARIMA / SARIMA identification, residual diagnostics, h=10 forecast
Run AFTER python/01_data_eda.py (or the R version: it falls back to data/prices_R.csv).
Install: pip install pmdarima statsmodels pandas numpy scipy matplotlib
"""
import os, random, warnings, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
import pmdarima as pm

warnings.filterwarnings("ignore")
SEED = 42; random.seed(SEED); np.random.seed(SEED)   # reproducibility
os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)
H = 10                                               # forecast horizon (brief: h = 5..10)

# ---- 1. LOAD + SAME 90/10 SPLIT AS PART 1 (identical n and cut, so train/test never differ between scripts)
if os.path.exists("data/prices.csv"):
    px = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)["AdjClose_AAPL"]
else:
    px = pd.read_csv("data/prices_R.csv", index_col=0, parse_dates=True)["AAPL"]
lp = np.log(px).iloc[1:]                              # drop first obs: aligned with the return series of Part 1
n = len(lp); cut = int(0.9 * n)
train, test = lp.iloc[:cut].values, lp.iloc[cut:].values   # plain arrays: avoids irregular-frequency warnings
print(f"n={n} train={cut} test={n-cut}")

# ---- 2. MANUAL IDENTIFICATION from ACF/PACF of the differenced series (d=1 from ADF/KPSS in Part 1)
r = np.diff(train)
bound = 1.96 / np.sqrt(len(r))                        # 95% white-noise band
a, pa = acf(r, nlags=30)[1:], pacf(r, nlags=30)[1:]
print("\nSignificant ACF lags :", [i+1 for i, v in enumerate(a) if abs(v) > bound])
print("Significant PACF lags:", [i+1 for i, v in enumerate(pa) if abs(v) > bound])
print("(~5% of 30 lags exceed the band by chance alone: isolated lags are not evidence of structure)")
print(acorr_ljungbox(r, lags=[10, 20]).round(4))     # H0: returns are white noise

# ---- 3. SEARCH: (a) information-criterion grid, (b) pmdarima auto_arima -> compare both approaches
rows = []
for p, q in itertools.product(range(4), range(4)):
    try:
        f = ARIMA(train, order=(p, 1, q), trend="t").fit()   # trend='t' = drift on the differenced series
        rows.append((p, 1, q, f.aic, f.bic))
    except Exception: pass
grid = pd.DataFrame(rows, columns=["p", "d", "q", "AIC", "BIC"]).sort_values("BIC")
print("\nTop 5 by BIC (BIC penalises complexity more: favours parsimony):\n", grid.head(5).round(2).to_string(index=False))
bp, bq = int(grid.iloc[0].p), int(grid.iloc[0].q)

auto = pm.auto_arima(train, d=1, seasonal=False, information_criterion="bic",
                     stepwise=False, max_p=5, max_q=5, with_intercept=True, suppress_warnings=True)
print("\nauto_arima (non-seasonal):", auto.order, "| BIC =", round(auto.bic(), 2))

# SARIMA check: weekly cycle s=5 is the only plausible seasonality in daily trading data
sauto = pm.auto_arima(train, d=1, seasonal=True, m=5, D=0, information_criterion="bic",
                      stepwise=True, suppress_warnings=True)
print("auto_arima (seasonal m=5):", sauto.order, sauto.seasonal_order, "| BIC =", round(sauto.bic(), 2))
print("-> SARIMA only justified if its BIC is clearly lower than the non-seasonal model.")

# ---- 4. FINAL MODEL + FULL RESIDUAL DIAGNOSTICS
def fit_and_diagnose(order, name):
    fit = ARIMA(train, order=order, trend="t").fit()
    res = fit.resid[1:]                                # first residual is an initialisation artefact for d=1
    k = order[0] + order[2]
    lb = acorr_ljungbox(res, lags=[10, 20], model_df=k)     # H0: no residual autocorrelation
    jb = stats.jarque_bera(res); sh = stats.shapiro(res)    # H0: normal (will be rejected: fat tails)
    a5, a10 = het_arch(res, nlags=5), het_arch(res, nlags=10)  # H0: no ARCH effect (constant variance)
    print(f"\n=== Diagnostics {name} {order} ===")
    print("Ljung-Box p (lag10, lag20):", lb["lb_pvalue"].round(4).tolist())
    print(f"Jarque-Bera p={jb.pvalue:.4g} | Shapiro p={sh.pvalue:.4g} | excess kurt={stats.kurtosis(res):.2f}")
    print(f"ARCH-LM p (5 lags)={a5[1]:.4g} | (10 lags)={a10[1]:.4g}")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(res, lw=.5); ax[0, 0].set_title("Residuals")
    plot_acf(res, lags=30, ax=ax[0, 1], title="ACF residuals")
    plot_acf(res**2, lags=30, ax=ax[1, 0], title="ACF squared residuals (ARCH check)")
    stats.probplot(res, plot=ax[1, 1]); ax[1, 1].set_title("QQ-plot")
    plt.tight_layout(); plt.savefig(f"figures/04_diag_{name}.png", dpi=130); plt.close()
    return fit
fit_grid = fit_and_diagnose((bp, 1, bq), "grid_BIC")
fit_auto = fit_and_diagnose(tuple(auto.order), "auto")
FIT = fit_grid                                        # choose one for forecasting; justify in report

# ---- 5. h=10 FORECAST vs RANDOM-WALK-WITH-DRIFT BASELINE (an ARIMA that cannot beat it adds nothing)
print(FIT.summary())   # coefficients, needed in the report
fc = FIT.get_forecast(H)
mu, ci = np.asarray(fc.predicted_mean), np.asarray(fc.conf_int(alpha=0.05))
drift = r.mean(); rw = train[-1] + drift * np.arange(1, H + 1)
actual = test[:H]
def metrics(y, yhat):                                  # computed on PRICE scale (exp of log price), the interpretable one
    y, yhat = np.exp(y), np.exp(yhat); e = y - yhat
    return dict(MSE=np.mean(e**2), RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), MAPE=100*np.mean(np.abs(e/y)))
tab = pd.DataFrame({f"ARIMA{(bp,1,bq)}": metrics(actual, mu), "RW+drift": metrics(actual, rw)}).T
print(f"\nForecast accuracy, h={H}, price scale:\n", tab.round(4))
cover = np.mean((actual >= ci[:, 0]) & (actual <= ci[:, 1]))
print(f"95% interval coverage on these {H} points: {cover:.0%}")
print("Caution: 10 points = very noisy comparison. Section 6 will use a rolling-origin evaluation + Diebold-Mariano.")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(-30, 0), np.exp(train[-30:]), label="train"); ax.plot(range(H), np.exp(actual), "k", label="actual")
ax.plot(range(H), np.exp(mu), "r", label="ARIMA"); ax.plot(range(H), np.exp(rw), "g--", label="RW+drift")
ax.fill_between(range(H), np.exp(ci[:, 0]), np.exp(ci[:, 1]), color="r", alpha=.15); ax.legend()
plt.tight_layout(); plt.savefig("figures/05_arima_forecast.png", dpi=130); plt.close()
pd.DataFrame({"actual_lp": actual, "arima_lp": mu, "lo": ci[:, 0], "hi": ci[:, 1], "rw_lp": rw}).to_csv("data/forecast_arima.csv", index=False)
tab.to_csv("data/metrics_arima.csv")
