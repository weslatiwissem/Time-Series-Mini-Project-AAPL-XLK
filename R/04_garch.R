# PART 4 (R) - ARCH/GARCH: ARCH test, GARCH(1,1) vs higher orders / asymmetry / distributions,
#              standardized-residual diagnostics, conditional volatility plot, volatility-based forecast intervals
# Install: install.packages(c("rugarch","xts","zoo","tseries"))
suppressPackageStartupMessages({library(rugarch); library(xts); library(zoo); library(tseries)})
set.seed(42); H <- 10
dir.create("figures", showWarnings=FALSE)

# ---- 1. LOAD + SAME SPLIT AS PARTS 1-3 (returns in PERCENT: GARCH optimisers are far more stable on a ~1 scale than on ~0.01)
px <- as.xts(read.zoo("data/prices_R.csv", header=TRUE, sep=",", index.column=1))
lpf <- as.numeric(log(px$AAPL)); dates <- index(px)[-1]
ret <- diff(lpf) * 100; n <- length(ret); cut <- floor(0.9 * n)
r_tr <- ret[1:cut]; dates_tr <- dates[1:cut]
lp_last <- lpf[cut + 1]; actual <- lpf[(cut + 2):(cut + 1 + H)]     # identical alignment to Part 2

# ---- 2. ARCH-LM on the mean-model residuals (ARMA(1,0) = the mean equation of ARIMA(1,1,0) chosen in Part 2)
arch_lm <- function(e, k) { e2 <- e^2; m <- embed(e2, k + 1)
  1 - pchisq(nrow(m) * summary(lm(m[, 1] ~ m[, -1]))$r.squared, k) }  # H0: constant variance
res_mean <- residuals(arima(r_tr, order = c(1, 0, 0)))
cat("ARCH-LM on ARMA(1,0) residuals, p (5, 10, 20 lags):", arch_lm(res_mean, 5), arch_lm(res_mean, 10), arch_lm(res_mean, 20), "\n")

# ---- 3. CANDIDATE GARCH MODELS: GARCH(1,1) is the required baseline; others are fitted ONLY to see whether diagnostics justify more
fit_g <- function(model, order, dist) {
  spec <- ugarchspec(variance.model = list(model = model, garchOrder = order),
                     mean.model = list(armaOrder = c(1, 0), include.mean = TRUE), distribution.model = dist)
  tryCatch(ugarchfit(spec, r_tr, solver = "hybrid"), error = function(e) NULL) }
cands <- list("GARCH(1,1)-norm" = list("sGARCH", c(1,1), "norm"),   # normal errors: expected to be too thin-tailed
              "GARCH(1,1)-std"  = list("sGARCH", c(1,1), "std"),    # Student-t: kurtosis was 6.6
              "GARCH(1,1)-sstd" = list("sGARCH", c(1,1), "sstd"),   # skewed t: skewness was slightly negative
              "GARCH(2,1)-std"  = list("sGARCH", c(2,1), "std"),
              "GARCH(1,2)-std"  = list("sGARCH", c(1,2), "std"),
              "GJR(1,1)-std"    = list("gjrGARCH", c(1,1), "std"))  # leverage effect: bad news raises volatility more
fits <- lapply(cands, function(a) fit_g(a[[1]], a[[2]], a[[3]]))
ic <- do.call(rbind, lapply(names(fits), function(nm) { f <- fits[[nm]]
  if (is.null(f)) return(data.frame(model = nm, AIC = NA, BIC = NA, loglik = NA, converged = FALSE))
  i <- infocriteria(f); data.frame(model = nm, AIC = i[1], BIC = i[2], loglik = likelihood(f), converged = convergence(f) == 0) }))
cat("\nGARCH comparison (AIC/BIC are per observation: only their ORDER matters):\n"); print(ic, row.names = FALSE)

# ---- 4. DIAGNOSTICS OF THE STANDARDIZED RESIDUALS z = e / sigma (must look like white noise AND have constant variance)
diagnose_g <- function(fit, name, garch_df) {
  z <- as.numeric(residuals(fit, standardize = TRUE))
  cat("\n=== Standardized residuals:", name, "===\n")
  cat("Ljung-Box on z   p (lag10, lag20):", Box.test(z, 10, "Ljung-Box", fitdf = 1)$p.value, Box.test(z, 20, "Ljung-Box", fitdf = 1)$p.value, "\n")      # mean equation
  cat("Ljung-Box on z^2 p (lag10, lag20):", Box.test(z^2, 10, "Ljung-Box", fitdf = garch_df)$p.value, Box.test(z^2, 20, "Ljung-Box", fitdf = garch_df)$p.value, "\n") # variance equation
  cat("ARCH-LM on z     p (5, 10 lags)   :", arch_lm(z, 5), arch_lm(z, 10), "\n")                                          # should now be NOT significant
  cat("Jarque-Bera on z p =", jarque.bera.test(z)$p.value, "(normality is NOT expected: errors are modelled as t; check the QQ-plot against the fitted t)\n")
  cf <- coef(fit)
  q_th <- tryCatch(qdist(fit@model$modeldesc$distribution, p = ppoints(length(z)),
                         skew = if ("skew" %in% names(cf)) cf["skew"] else 1, shape = if ("shape" %in% names(cf)) cf["shape"] else 5), error = function(e) qnorm(ppoints(length(z))))
  png(paste0("figures/R06_garch_diag_", gsub("[^A-Za-z0-9]", "", name), ".png"), 1100, 700); par(mfrow = c(2, 2))
  plot(z, type = "l", main = "Standardized residuals"); acf(z, 30, main = "ACF z"); acf(z^2, 30, main = "ACF z^2 (should be flat)")
  qqplot(q_th, z, main = paste("QQ vs fitted", fit@model$modeldesc$distribution), xlab = "theoretical", ylab = "sample"); abline(0, 1, col = "red"); dev.off()
}
MAIN <- "GARCH(1,1)-std"; fit_main <- fits[[MAIN]]               # the required GARCH(1,1), with t errors
diagnose_g(fit_main, MAIN, garch_df = 2)
best <- ic$model[which.min(ic$BIC)]
if (best != MAIN) diagnose_g(fits[[best]], best, garch_df = sum(cands[[best]][[2]]) + (grepl("GJR", best)))
cat("\nBest by BIC:", best, "\n")
show(fit_main)       # includes the Sign-Bias test (significant => asymmetry => GJR justified) and Nyblom stability test

# ---- 5. PERSISTENCE AND UNCONDITIONAL VOLATILITY
pers <- persistence(fit_main)
cat(sprintf("\nPersistence alpha+beta = %.4f | half-life of a volatility shock = %.1f days | long-run vol = %.1f%% annualised\n",
            pers, log(0.5) / log(pers), sqrt(uncvariance(fit_main)) * sqrt(252)))

# ---- 6. CONDITIONAL VOLATILITY OVER TIME (annualised), linked to events -> label the red lines yourself after checking the news
vol <- as.numeric(sigma(fit_main)) * sqrt(252)
png("figures/R07_cond_vol.png", 1100, 450)
plot(dates_tr, vol, type = "l", xlab = "", ylab = "% annualised", main = "AAPL conditional volatility, GARCH(1,1)-t")
ev <- as.Date(c("2020-03-16", "2022-11-10", "2025-04-09"))        # extreme-return dates found in Part 1
abline(v = ev, lty = 2, col = "red"); mtext(format(ev), side = 3, at = ev, cex = 0.7, col = "red"); dev.off()
cat("Peak conditional volatility:", round(max(vol), 1), "% on", format(dates_tr[which.max(vol)]), "\n")
write.csv(data.frame(date = dates_tr, sigma_ann = vol), "data/garch_condvol_R.csv", row.names = FALSE)

# ---- 7. FORECAST h=10: the mean is almost the ARIMA one; the GARCH contribution is the volatility-aware INTERVAL
fc <- ugarchforecast(fit_main, n.ahead = H)
mu_r <- as.numeric(fitted(fc)) / 100; sg <- as.numeric(sigma(fc)) / 100
lp_hat <- lp_last + cumsum(mu_r)
half <- 1.96 * sqrt(cumsum(sg^2))          # approx: sums variances of daily returns (ignores the small AR(1) correlation); ~normal for a 10-day sum
lo <- lp_hat - half; hi <- lp_hat + half
cat("\nForecast sigma (annualised %) day 1 -> day 10:", round(sg[1] * sqrt(252) * 100, 1), "->", round(sg[H] * sqrt(252) * 100, 1), "\n")

a2 <- read.csv("data/forecast_arima_R.csv")
metrics <- function(y, yhat) { y <- exp(y); yhat <- exp(yhat); e <- y - yhat
  c(MSE = mean(e^2), RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)), MAPE = 100 * mean(abs(e / y))) }
tab <- rbind(RW_drift = metrics(actual, a2$rw_lp), ARIMA = metrics(actual, a2$arima_lp), ARMA_GARCH = metrics(actual, lp_hat))
cat("\nForecast accuracy, h =", H, "(price scale):\n"); print(round(tab, 4))
cat(sprintf("Interval coverage (10 pts): ARIMA %.0f%% | GARCH %.0f%%   Mean width (price): ARIMA %.2f | GARCH %.2f\n",
            100 * mean(actual >= a2$lo & actual <= a2$hi), 100 * mean(actual >= lo & actual <= hi),
            mean(exp(a2$hi) - exp(a2$lo)), mean(exp(hi) - exp(lo))))
cat("Interpretation: ARIMA assumes the AVERAGE past volatility forever; GARCH uses the CURRENT volatility state, so its interval is narrower after calm periods and wider after turbulent ones.\n")
write.csv(data.frame(actual_lp = actual, garch_lp = lp_hat, lo = lo, hi = hi), "data/forecast_garch_R.csv", row.names = FALSE)
write.csv(tab, "data/metrics_garch_R.csv")
