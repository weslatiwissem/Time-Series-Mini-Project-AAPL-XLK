"""
PART 4 (Python) - ARCH/GARCH: ARCH test, GARCH variants, standardized-residual diagnostics, volatility plot, volatility-aware intervals
Same study as R/04_garch.R (uses the `arch` package). Run from the PROJECT ROOT:  python python/04_garch.py
Needs data/prices.csv (python/01) OR data/prices_R.csv (R/01).  pip install arch pandas numpy scipy matplotlib statsmodels
NOTE: conventions differ from rugarch (e.g. arch's p = ARCH order, q = GARCH order, o = asymmetry); compare CONCLUSIONS, not every decimal.
"""
import os, random, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arch as arch_pkg
from arch import arch_model
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")
SEED, H = 42, 10; random.seed(SEED); np.random.seed(SEED)
os.makedirs("data", exist_ok=True); os.makedirs("figures", exist_ok=True)
print("arch", arch_pkg.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)

# ---- 1. LOAD + SAME SPLIT (returns in PERCENT: GARCH optimisers are much more stable on a ~1 scale than on ~0.01)
if os.path.exists("data/prices.csv"):
    px = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)["AdjClose_AAPL"]
else:
    px = pd.read_csv("data/prices_R.csv", index_col=0, parse_dates=True)["AAPL"]
lpf = np.log(px.dropna().values); dates = px.dropna().index[1:]
ret = np.diff(lpf) * 100; n = len(ret); cut = int(0.9 * n)
r_tr, dates_tr = ret[:cut], dates[:cut]
lp_last = lpf[cut]; actual = lpf[cut + 1:cut + 1 + H]                 # same alignment as R/04 and python/02

# ---- 2. ARCH-LM on the residuals of the mean equation ARMA(1,0) (the mean of the ARIMA(1,1,0) chosen in Part 2)
res_mean = ARIMA(r_tr, order=(1, 0, 0)).fit().resid
print("ARCH-LM on ARMA(1,0) residuals, p (5, 10, 20 lags):", [float(f"{het_arch(res_mean, nlags=k)[1]:.3g}") for k in (5, 10, 20)])

# ---- 3. CANDIDATES: GARCH(1,1) is the required baseline; the others are fitted ONLY to see whether diagnostics justify more
cands = {"GARCH(1,1)-norm": dict(p=1, o=0, q=1, dist="normal"),     # normal errors: expected too thin-tailed
         "GARCH(1,1)-std":  dict(p=1, o=0, q=1, dist="t"),          # Student-t: kurtosis was 6.6
         "GARCH(1,1)-sstd": dict(p=1, o=0, q=1, dist="skewt"),      # skewed t: skewness was slightly negative
         "GARCH(2,1)-std":  dict(p=2, o=0, q=1, dist="t"),
         "GARCH(1,2)-std":  dict(p=1, o=0, q=2, dist="t"),
         "GJR(1,1)-std":    dict(p=1, o=1, q=1, dist="t")}           # leverage effect: bad news raises volatility more
fits = {nm: arch_model(r_tr, mean="AR", lags=1, vol="GARCH", **c).fit(disp="off") for nm, c in cands.items()}
ic = pd.DataFrame({nm: dict(loglik=f.loglikelihood, AIC=f.aic, BIC=f.bic, converged=f.convergence_flag == 0) for nm, f in fits.items()}).T
print("\nGARCH comparison (lower AIC/BIC is better):\n", ic.round(2))

# ---- 4. DIAGNOSTICS of the standardized residuals z = e / sigma (must look like white noise AND have constant variance)
def diagnose(res, name, garch_df):
    z = np.asarray(res.std_resid); z = z[~np.isnan(z)]
    lbz = acorr_ljungbox(z, lags=[10, 20], model_df=1)["lb_pvalue"].values             # mean equation (AR(1): 1 parameter)
    lbz2 = acorr_ljungbox(z**2, lags=[10, 20], model_df=garch_df)["lb_pvalue"].values  # variance equation
    print(f"\n=== Standardized residuals: {name} ===")
    print("Ljung-Box on z   p (lag10, lag20):", np.round(lbz, 4)); print("Ljung-Box on z^2 p (lag10, lag20):", np.round(lbz2, 4))
    print("ARCH-LM on z     p (5, 10 lags)   :", [round(float(het_arch(z, nlags=k)[1]), 4) for k in (5, 10)], "(should now be NOT significant)")
    print(f"Jarque-Bera on z p = {stats.jarque_bera(z).pvalue:.3g} (normality is NOT expected: errors are modelled as t; read the QQ-plot against the fitted t)")
    pp = (np.arange(1, len(z) + 1) - 0.5) / len(z)
    if "nu" in res.params.index and "lambda" not in res.params.index:        # standardized Student-t quantiles (unit variance, as in arch)
        nu = res.params["nu"]; q_th = stats.t.ppf(pp, nu) * np.sqrt((nu - 2) / nu); lab = f"t({nu:.1f})"
    else: q_th, lab = stats.norm.ppf(pp), "normal"
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(z, lw=.5); ax[0, 0].set_title("Standardized residuals")
    plot_acf(z, lags=30, ax=ax[0, 1], title="ACF z"); plot_acf(z**2, lags=30, ax=ax[1, 0], title="ACF z^2 (should be flat)")
    ax[1, 1].plot(q_th, np.sort(z), ".", ms=3); lim = [q_th.min(), q_th.max()]; ax[1, 1].plot(lim, lim, "r"); ax[1, 1].set_title(f"QQ vs fitted {lab}")
    plt.tight_layout(); plt.savefig("figures/09_garch_diag_" + "".join(ch for ch in name if ch.isalnum()) + ".png", dpi=130); plt.close()
MAIN = "GARCH(1,1)-std"; fit_main = fits[MAIN]
diagnose(fit_main, MAIN, garch_df=2)
best = ic["BIC"].astype(float).idxmin()
if best != MAIN: diagnose(fits[best], best, garch_df=cands[best]["p"] + cands[best]["q"] + cands[best]["o"])
print("\nBest by BIC:", best)
print(fit_main.summary())

# ---- 5. PERSISTENCE AND UNCONDITIONAL VOLATILITY
P = fit_main.params; pers = P["alpha[1]"] + P["beta[1]"]
print(f"\nPersistence alpha+beta = {pers:.4f} | half-life of a volatility shock = {np.log(0.5)/np.log(pers):.1f} days"
      f" | long-run vol = {np.sqrt(P['omega']/(1-pers))*np.sqrt(252):.1f}% annualised")
if "gamma[1]" in fits["GJR(1,1)-std"].params.index:
    g = fits["GJR(1,1)-std"]; print(f"GJR leverage coefficient gamma = {g.params['gamma[1]']:.4f} (p = {g.pvalues['gamma[1]']:.4f}); a positive, significant gamma would mean bad news raises volatility more")

# ---- 6. CONDITIONAL VOLATILITY OVER TIME (annualised), linked to events -> label the red lines yourself after checking the news
vol = np.asarray(fit_main.conditional_volatility) * np.sqrt(252); d_vol = dates_tr[-len(vol):]   # first value is NaN with an AR(1) mean
plt.figure(figsize=(11, 4)); plt.plot(d_vol, vol, lw=.8)
for e in ["2020-03-16", "2022-11-10", "2025-04-09"]:                    # extreme-return dates found in Part 1
    plt.axvline(pd.Timestamp(e), ls="--", c="r", lw=.8); plt.text(pd.Timestamp(e), np.nanmax(vol) * 1.01, e, color="r", fontsize=7, ha="center")
plt.ylabel("% annualised"); plt.title("AAPL conditional volatility, GARCH(1,1)-t"); plt.tight_layout(); plt.savefig("figures/10_cond_vol.png", dpi=130); plt.close()
print(f"Peak conditional volatility: {np.nanmax(vol):.1f}% on {d_vol[np.nanargmax(vol)].date()}")
pd.DataFrame({"date": d_vol, "sigma_ann": vol}).to_csv("data/garch_condvol_py.csv", index=False)

# ---- 7. FORECAST h=10: the mean is almost the ARIMA one; the GARCH contribution is the volatility-aware INTERVAL
f = fit_main.forecast(horizon=H, reindex=False)
mu_r = f.mean.values[-1] / 100; sg = np.sqrt(f.variance.values[-1]) / 100
lp_hat = lp_last + np.cumsum(mu_r)
half = 1.96 * np.sqrt(np.cumsum(sg**2))                                # sum of daily variances; ~normal for multi-day sums
nu = P["nu"]; half[0] = stats.t.ppf(0.975, nu) * np.sqrt((nu - 2) / nu) * sg[0]   # day 1: exact standardized-t quantile
lo, hi = lp_hat - half, lp_hat + half
print("\nForecast sigma (annualised %) day 1 -> day 10:", round(sg[0] * np.sqrt(252) * 100, 1), "->", round(sg[-1] * np.sqrt(252) * 100, 1))
def metrics(y, p):
    y, p = np.exp(y), np.exp(p); e = y - p
    return dict(MSE=np.mean(e**2), RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), MAPE=100 * np.mean(np.abs(e / y)))
rw = lp_last + (r_tr.mean() / 100) * np.arange(1, H + 1)
tab = {"RW_drift": metrics(actual, rw), "ARMA_GARCH": metrics(actual, lp_hat)}
cover = {"GARCH": np.mean((actual >= lo) & (actual <= hi)), "GARCH_width": np.mean(np.exp(hi) - np.exp(lo))}
if os.path.exists("data/forecast_arima.csv"):                          # written by python/02_arima.py
    a2 = pd.read_csv("data/forecast_arima.csv"); tab["ARIMA"] = metrics(actual, a2["arima_lp"].values)
    cover["ARIMA"] = np.mean((actual >= a2["lo"].values) & (actual <= a2["hi"].values)); cover["ARIMA_width"] = np.mean(np.exp(a2["hi"].values) - np.exp(a2["lo"].values))
print(f"\nForecast accuracy, h={H} (price scale):\n", pd.DataFrame(tab).T.round(4))
print("Interval coverage / width (10 points, not informative on its own):", {k: round(float(v), 3) for k, v in cover.items()})
print("Interpretation: ARIMA assumes the AVERAGE past volatility forever; GARCH uses the CURRENT volatility state.")
pd.DataFrame({"actual_lp": actual, "garch_lp": lp_hat, "lo": lo, "hi": hi}).to_csv("data/forecast_garch_py.csv", index=False)
pd.DataFrame(tab).T.to_csv("data/metrics_garch_py.csv")

# ---- 8. SENSITIVITY: does the AR(1) term / the very first observation (2019-01-03, a -10.5% day) matter?
rows = {}
for label, y, mean in [("AR(1), all obs", r_tr, "AR"), ("constant mean, all obs", r_tr, "Constant"), ("constant mean, drop first", r_tr[1:], "Constant")]:
    kw = dict(lags=1) if mean == "AR" else {}
    r_ = arch_model(y, mean=mean, vol="GARCH", p=1, q=1, dist="t", **kw).fit(disp="off"); rows[label] = r_.params.rename({"mu": "Const"})
print("\nSensitivity of the GARCH(1,1)-t parameters:\n", pd.DataFrame(rows).T.round(4).to_string())
