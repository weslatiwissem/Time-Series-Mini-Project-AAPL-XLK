"""
PART 3 (Python) - VAR, Granger causality, cointegration (Engle-Granger + Johansen), VECM
Same study as R/03_var_cointegration.R. Run from the PROJECT ROOT:  python python/03_var_cointegration.py
Needs data/prices.csv (python/01) OR data/prices_R.csv (R/01).  pip install pandas numpy scipy matplotlib statsmodels
NOTE: package conventions differ between R and Python (lag selection in ADF, deterministic terms in Johansen, critical values):
compare CONCLUSIONS between the two languages, not every decimal.
"""
import os, random, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.stats.diagnostic import het_arch
import statsmodels.api as sm

warnings.filterwarnings("ignore")
SEED, H = 42, 10; random.seed(SEED); np.random.seed(SEED)            # fixed seed; H = longest horizon of the brief
os.makedirs("data", exist_ok=True); os.makedirs("figures", exist_ok=True)
print("statsmodels", statsmodels.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)

# ---- 1. LOAD + SAME 90/10 SPLIT AS THE R SCRIPTS
if os.path.exists("data/prices.csv"):
    px = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)[["AdjClose_AAPL", "AdjClose_XLK"]]; px.columns = ["AAPL", "XLK"]
else:
    px = pd.read_csv("data/prices_R.csv", index_col=0, parse_dates=True)[["AAPL", "XLK"]]
lp2 = np.log(px.dropna()).iloc[1:]                                   # drop first obs: same alignment as the return series
n = len(lp2); cut = int(0.9 * n)
Y = pd.DataFrame(lp2.iloc[:cut].values, columns=["AAPL", "XLK"])     # RangeIndex: avoids irregular-date warnings in VAR
Ytest = lp2.iloc[cut:].values
R = Y.diff().dropna()                                                # returns: the stationary representation used for the VAR
print(f"n={n} train={cut} test={n-cut}")

# ---- 2. PRE-CONDITION: both log prices must be I(1)
for v in Y.columns:
    print(v, "| ADF p (level) =", round(adfuller(Y[v], autolag="AIC")[1], 3), "| ADF p (first diff) =", round(adfuller(Y[v].diff().dropna(), autolag="AIC")[1], 4))

# ---- 3. COINTEGRATION, step A: Engle-Granger (residual-based test with the CORRECT critical values: statsmodels.coint)
ols = sm.OLS(Y["AAPL"], sm.add_constant(Y["XLK"])).fit()
print("\nLong-run regression AAPL ~ XLK:", ols.params.round(4).to_dict(), "(t-statistics are NOT valid for non-stationary series)")
eg_stat, eg_p, eg_cv = coint(Y["AAPL"], Y["XLK"], trend="c", autolag="aic")   # H0: NO cointegration
print(f"Engle-Granger: statistic = {eg_stat:.3f}, p = {eg_p:.3f} (R used the Phillips-Ouliaris variant of this test)")
plt.figure(figsize=(10, 4)); plt.plot(ols.resid.values, lw=.7); plt.axhline(0, ls="--", c="k")
plt.title("Long-run spread: AAPL - a - b*XLK (log prices)"); plt.tight_layout(); plt.savefig("figures/08_spread.png", dpi=130); plt.close()

# ---- 4. COINTEGRATION, step B: Johansen trace test
sel = VAR(Y).select_order(10).selected_orders; print("\nLag selection on the level VAR:", sel)
K = max(2, sel["bic"])                                               # K lags in levels <=> K-1 lagged differences in the VECM (BIC, as in R)
def johansen_rank(det, k_diff):
    j = coint_johansen(Y, det_order=det, k_ar_diff=k_diff)           # j.lr1 = trace statistics [r=0, r<=1]; j.cvt columns = 90/95/99%
    rank = (2 if j.lr1[1] > j.cvt[1, 1] else 1) if j.lr1[0] > j.cvt[0, 1] else 0   # sequential decision from r = 0, 5% level
    return j, rank
jo, rank = johansen_rank(0, K - 1)
print(f"Johansen (det_order=0, K={K}): r=0 trace = {jo.lr1[0]:.2f} vs 5% cv {jo.cvt[0,1]:.2f} | r<=1 trace = {jo.lr1[1]:.2f} vs 5% cv {jo.cvt[1,1]:.2f}")
print("Johansen cointegration rank at 5% =", rank)
if rank == 2: print("Rank 2 = full rank = levels stationary: contradicts the I(1) result, check the data / deterministic term.")
print("\nRobustness (deterministic term x lags): r=0 trace statistic vs its own 5% critical value")
for det, name in [(-1, "no-const"), (0, "const"), (1, "trend")]:
    for Kk in (2, 3, 5, 10):
        j, _ = johansen_rank(det, Kk - 1)
        flag = "  [no constant at all: misspecified for log prices, ignore]" if det == -1 else ""
        print(f"  det={name:<8} K={Kk:>2} | stat {j.lr1[0]:6.2f} vs 5% cv {j.cvt[0,1]:6.2f} -> {'REJECT no-cointegration' if j.lr1[0] > j.cvt[0,1] else 'cannot reject'}{flag}")
# Mapping to R (verified on this data): det_order=0 gives EXACTLY the statistics of R's ecdet="none" (unrestricted constant), but statsmodels uses
# different critical-value tables (15.49 vs 17.95). det_order=-1 has no constant at all: inappropriate for log prices (non-zero level), shown for completeness only.
# statsmodels has no restricted-constant option, so R's ecdet="const" has no exact counterpart here. Compare CONCLUSIONS across rows that include a constant.

# ---- 5. VAR ON RETURNS (stationary; correct whether or not the series are cointegrated)
selR = VAR(R).select_order(10).selected_orders; print("\nVAR lag selection on returns:", selR)
pR = max(1, selR["bic"])                                             # BIC order (AIC usually picks more: report both, justify)
var_r = VAR(R).fit(pR, trend="c")
print(pd.concat({"coef": var_r.params, "p-value": var_r.pvalues}, axis=1).round(4))

# Residual diagnostics
wh = var_r.test_whiteness(nlags=10, adjusted=False)                  # portmanteau, H0: no residual autocorrelation
print(f"\nPortmanteau (10 lags): stat = {wh.test_statistic:.2f}, df = {wh.df}, p = {wh.pvalue:.3g}")
nm = var_r.test_normality()                                          # H0: residuals multivariate normal
print(f"Multivariate Jarque-Bera: stat = {nm.test_statistic:.1f}, p = {nm.pvalue:.3g}")
for c in R.columns:                                                  # per-equation ARCH-LM (R reported a multivariate version)
    print(f"ARCH-LM {c} residuals: p(5 lags) = {het_arch(var_r.resid[c], nlags=5)[1]:.3g} | p(10 lags) = {het_arch(var_r.resid[c], nlags=10)[1]:.3g}")
print("Stable VAR:", var_r.is_stable(), "| moduli of companion eigenvalues:", np.round(np.abs(1 / var_r.roots), 3))

# ---- 6. GRANGER CAUSALITY (on the stationary VAR)
for cause, effect in [("AAPL", "XLK"), ("XLK", "AAPL")]:
    g = var_r.test_causality(effect, [cause], kind="f"); i = var_r.test_inst_causality(cause)
    print(f"Granger: {cause} -> {effect}  p = {g.pvalue:.4f} | instantaneous p = {i.pvalue:.4f}")
# Reading: Granger = does the PAST of one help predict the other beyond its own past (predictive, NOT structural causation).
# 'Instantaneous' is expected ~0: same-day returns are 0.83 correlated (comovement, not lead-lag).

# ---- 7. VECM (only if cointegration was found)
vecm_fc = None
if rank >= 1:
    vecm = VECM(Y, k_ar_diff=K - 1, coint_rank=1, deterministic="ci").fit()   # 'ci' = constant inside the long-run relation
    alpha = vecm.alpha[:, 0]
    print("\nCointegrating vector beta:", np.round(vecm.beta[:, 0], 4)); print("Adjustment speeds alpha (AAPL, XLK):", np.round(alpha, 5))
    hl = [np.log(0.5) / np.log(1 + a) if -1 < a < 0 else np.nan for a in alpha]
    print("Half-life (days) per equation (nan = that variable does not correct):", np.round(hl, 1))
    vecm_fc = vecm.predict(steps=H, alpha=0.05)                      # (forecast, lower, upper) in log prices
else:
    print("\nNo cointegration at 5%: VECM not justified. The VAR on returns is the correct multivariate model.")

# ---- 8. FORECAST h=10 AND COMPARISON (same price-scale metrics as R)
fc = var_r.forecast(R.values[-pR:], steps=H)
fc_var = Y["AAPL"].iloc[-1] + np.cumsum(fc[:, 0])                    # cumulate forecast returns back to log price
rw = Y["AAPL"].iloc[-1] + Y["AAPL"].diff().dropna().mean() * np.arange(1, H + 1)
actual = Ytest[:H, 0]
def metrics(y, p):
    y, p = np.exp(y), np.exp(p); e = y - p
    return dict(MSE=np.mean(e**2), RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), MAPE=100 * np.mean(np.abs(e / y)))
tab = {"RW_drift": metrics(actual, rw), "VAR_returns": metrics(actual, fc_var)}
out = pd.DataFrame({"actual_lp": actual, "var_lp": fc_var, "rw_lp": rw})
if vecm_fc is not None:
    tab["VECM"] = metrics(actual, vecm_fc[0][:, 0]); out["vecm_lp"] = vecm_fc[0][:, 0]
print(f"\nForecast accuracy, h={H} (price scale):\n", pd.DataFrame(tab).T.round(4))
print("Caution: 10 points only; the rolling-origin evaluation + Diebold-Mariano (R/06_evaluation.R) is the real comparison.")
out.to_csv("data/forecast_var_py.csv", index=False); pd.DataFrame(tab).T.to_csv("data/metrics_var_py.csv")
