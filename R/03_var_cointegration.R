# PART 3 (R) - VAR, Granger causality, cointegration (Phillips-Ouliaris + Johansen), VECM
# Pair: AAPL & XLK (log prices are I(1) per Part 1; their returns are I(0)).
# Run AFTER 01 and 02. Install: install.packages(c("vars","urca","tseries","xts","zoo"))
suppressPackageStartupMessages({library(vars); library(urca); library(tseries); library(xts); library(zoo)})
set.seed(42); H <- 10
dir.create("figures", showWarnings=FALSE)

# ---- 1. LOAD + SAME 90/10 SPLIT AS PARTS 1-2
px  <- as.xts(read.zoo("data/prices_R.csv", header=TRUE, sep=",", index.column=1))
lp2 <- log(px[, c("AAPL", "XLK")])[-1, ]             # drop first obs: same alignment as the return series
n <- nrow(lp2); cut <- floor(0.9 * n)
Y <- as.matrix(lp2[1:cut, ]); Ytest <- as.matrix(lp2[(cut+1):n, ])
R <- diff(Y)                                         # returns: the stationary representation used for the VAR

# ---- 2. PRE-CONDITION: both log prices must be I(1) (cointegration is only defined between I(1) series)
for (v in colnames(Y))
  cat(v, "| ADF p (level) =", round(adf.test(Y[, v])$p.value, 3), "| ADF p (first diff) =", round(adf.test(diff(Y[, v]))$p.value, 3), "\n")

# ---- 3. COINTEGRATION, step A: Engle-Granger idea with the proper test (Phillips-Ouliaris)
# A plain ADF on regression residuals uses wrong critical values (residuals are estimated), so po.test is used instead.
eg <- lm(AAPL ~ XLK, data = as.data.frame(Y)); print(round(summary(eg)$coefficients, 4))
po <- po.test(Y)                                     # H0: NO cointegration (small p => cointegrated)
cat("Phillips-Ouliaris p =", po$p.value, "(tseries caps p at 0.01-0.15)\n")
png("figures/R05_spread.png", 1000, 450); plot(zoo(residuals(eg), index(lp2)[1:cut]), main="Long-run spread: AAPL - a - b*XLK (log prices)", ylab="", xlab=""); abline(h=0, lty=2); dev.off()

# ---- 4. COINTEGRATION, step B: Johansen trace test (handles >2 series, no normalisation choice)
sel <- VARselect(Y, lag.max=10, type="const"); print(sel$selection)
K <- max(2, sel$selection[["SC(n)"]])                # BIC lag of the level VAR (ca.jo needs K >= 2)
jo <- ca.jo(Y, type="trace", ecdet="const", K=K)     # 'const' = intercept inside the long-run relation; re-run with "trend"/"none" as a robustness check
print(summary(jo))
ts_ <- jo@teststat; cv <- jo@cval[, "5pct"]          # order: [1] = (r <= 1), [2] = (r = 0)
rank <- if (ts_[2] > cv[2]) { if (ts_[1] > cv[1]) 2 else 1 } else 0   # sequential decision, starting from r = 0
cat("\nJohansen cointegration rank at 5% =", rank, "\n")
if (rank == 2) cat("Rank 2 = full rank = levels stationary: contradicts I(1) result, check the data / deterministic spec.\n")

# ---- 5. VAR ON RETURNS (stationary, correctly specified whether or not the series are cointegrated)
selR <- VARselect(R, lag.max=10, type="const"); print(selR$selection)
pR <- selR$selection[["SC(n)"]]                      # BIC order (AIC often picks more lags: report both, justify the choice)
var_r <- VAR(R, p=pR, type="const"); print(coef(var_r))

# Multivariate residual diagnostics
print(serial.test(var_r, lags.pt=10, type="PT.asymptotic"))   # H0: no residual autocorrelation (needs lags.pt > p)
print(normality.test(var_r))                                  # H0: residuals multivariate normal
print(arch.test(var_r, lags.multi=5, multivariate.only=TRUE)) # H0: no multivariate ARCH -> input to GARCH (Part 4)
print(roots(var_r))                                           # stability: all moduli must be < 1

# ---- 6. GRANGER CAUSALITY (on the stationary VAR)
for (cz in c("AAPL", "XLK")) {
  g <- causality(var_r, cause = cz)
  cat(sprintf("Granger: %s -> other  p = %.4f | instantaneous p = %.4f\n", cz, g$Granger$p.value, g$Instant$p.value))
}
# Reading: Granger = does the PAST of one help predict the other beyond its own past? (predictive, NOT causal in the structural sense.)
# 'Instantaneous' is expected to be ~0 here: same-day returns are 0.83 correlated, which is contemporaneous comovement, not lead-lag.

# ---- 7. VECM (only if cointegration was found)
vecm_fc <- NULL
if (rank >= 1) {
  r1 <- 1                                            # use rank 1 (one long-run relation between two series)
  vecm <- cajorls(jo, r = r1)
  cat("\nCointegrating vector (normalised on AAPL):\n"); print(vecm$beta)
  cat("\nVECM coefficients (look at the 'ect1' row = error-correction terms alpha):\n"); print(round(vecm$rlm$coefficients, 5))
  alpha <- vecm$rlm$coefficients["ect1", ]
  hl <- ifelse(alpha < 0 & alpha > -1, log(0.5) / log(1 + alpha), NA)
  cat("\nHalf-life (days) of a deviation, per equation (NA = that variable does not correct):\n"); print(round(hl, 1))
  # Interpretation: alpha<0 in an equation = that variable moves back toward the long-run relation; alpha ~ 0 = it does not adjust.
  vv <- vec2var(jo, r = r1)
  pv <- predict(vv, n.ahead = H, ci = 0.95)
  vecm_fc <- pv$fcst$AAPL                            # matrix: fcst, lower, upper (log price)
} else cat("\nNo cointegration at 5%: VECM not justified. The VAR on returns is the correct multivariate model.\n")

# ---- 8. FORECAST h=10 AND COMPARISON (same price-scale metrics as Part 2)
pf <- predict(var_r, n.ahead = H)
fc_var <- Y[cut, "AAPL"] + cumsum(pf$fcst$AAPL[, "fcst"])   # cumulate forecast returns back to log price (no interval: would need accumulated variance)
a2 <- read.csv("data/forecast_arima_R.csv"); actual <- Ytest[1:H, "AAPL"]
metrics <- function(y, yhat) { y <- exp(y); yhat <- exp(yhat); e <- y - yhat
  c(MSE = mean(e^2), RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)), MAPE = 100 * mean(abs(e / y))) }
tab <- rbind(RW_drift = metrics(actual, a2$rw_lp), ARIMA = metrics(actual, a2$arima_lp), VAR_returns = metrics(actual, fc_var))
if (!is.null(vecm_fc)) tab <- rbind(tab, VECM = metrics(actual, vecm_fc[, "fcst"]))
cat("\nForecast accuracy, h =", H, "(price scale):\n"); print(round(tab, 4))
cat("Caution: 10 points only; the rolling-origin evaluation + Diebold-Mariano (Section 6) is the real comparison.\n")
out <- data.frame(actual_lp = actual, var_lp = fc_var)
if (!is.null(vecm_fc)) out <- cbind(out, vecm_lp = vecm_fc[, "fcst"], vecm_lo = vecm_fc[, "lower"], vecm_hi = vecm_fc[, "upper"])
write.csv(out, "data/forecast_var_R.csv", row.names = FALSE); write.csv(tab, "data/metrics_var_R.csv")

for (ec in c("none","const","trend")) for (K in c(2,3,5,10)) {
  j <- ca.jo(Y, type="trace", ecdet=ec, K=K)
  cat(sprintf("ecdet=%-5s K=%2d | r=0 stat %.2f vs 5%% cv %.2f\n", ec, K, j@teststat[2], j@cval[2,"5pct"]))
}

