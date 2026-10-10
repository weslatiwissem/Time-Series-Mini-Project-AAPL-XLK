# PART 6 (R) - Section 6: rolling-origin evaluation of ALL models, one summary table, Diebold-Mariano tests, interval coverage
# Run AFTER R/01-04 and python/05_lstm_gru.py (needs data/forecast_rnn_py.csv). Expected runtime: a few minutes (about 370 model refits).
# Install: install.packages(c("forecast","vars","rugarch","xts","zoo"))
suppressPackageStartupMessages({library(forecast); library(vars); library(rugarch); library(xts); library(zoo)})
set.seed(42); H <- 10; HS <- c(1, 5, 10)
dir.create("figures", showWarnings = FALSE)

# ---- 1. ORIGINS: identical to python/05 (origin k = last observed day; K = 186 origins covering the whole test period)
px  <- as.xts(read.zoo("data/prices_R.csv", header = TRUE, sep = ",", index.column = 1))
lpf <- as.numeric(log(px$AAPL)); lpx <- as.numeric(log(px$XLK)); dates <- index(px)[-1]
ret <- diff(lpf); retx <- diff(lpx)                    # daily log returns (decimals)
n <- length(ret); cut <- floor(0.9 * n); K <- n - H - cut + 1
orig <- cut + 0:(K - 1)                                # 1-based index of the last observed return at each origin
lp_t <- lpf[orig + 1]                                  # log price at each origin
actual <- t(sapply(orig, function(o) lpf[o + 1 + (1:H)]))   # K x H matrix of realised log prices

py <- read.csv("data/forecast_rnn_py.csv"); stopifnot(nrow(py) == K * H)
if (!all(as.character(dates[orig]) == py$origin_date[py$h == 1])) stop("Origin dates differ between R and Python: check the split.")
gap <- max(abs(matrix(py$actual_lp, K, H, byrow = TRUE) - actual))
cat("Max |actual log price R - Python| =", signif(gap, 3), "\n")
if (gap > 1e-2) stop("data/prices.csv (Python) and data/prices_R.csv (R) differ: delete data/prices.csv and re-run python/05.")
if (gap > 1e-3) warning("Small price-file mismatch between R and Python downloads.")

# ---- 2. FORECASTS (each K x H matrix of predicted LOG prices)
FC <- list()
FC$RW       <- matrix(rep(lp_t, H), K, H)              # random walk, no drift
mu <- mean(ret[1:cut])                                 # drift estimated on TRAIN only (same as the Python file)
FC$RW_drift <- outer(lp_t, mu * (1:H), "+")            # random walk with drift = the benchmark to beat

# ARIMA(1,1,0)+drift: ORDER fixed from Part 2 (chosen on train), PARAMETERS re-estimated at every origin on all data up to that origin
FC$ARIMA <- matrix(NA_real_, K, H); LO_a <- HI_a <- matrix(NA_real_, K, H)
t_arima <- system.time(for (k in 1:K) {
  fit <- tryCatch(Arima(lpf[2:(orig[k] + 1)], order = c(1, 1, 0), include.drift = TRUE), error = function(e) NULL)
  if (is.null(fit)) { FC$ARIMA[k, ] <- FC$RW_drift[k, ]; next }
  fc <- forecast(fit, h = H, level = 95)
  FC$ARIMA[k, ] <- fc$mean; LO_a[k, ] <- fc$lower; HI_a[k, ] <- fc$upper })[["elapsed"]]

# VAR(1) on AAPL/XLK returns (BIC order from Part 3), re-estimated at every origin; returns are cumulated back to log price
rr <- cbind(AAPL = ret, XLK = retx); FC$VAR <- matrix(NA_real_, K, H)
t_var <- system.time(for (k in 1:K) {
  fit <- VAR(rr[1:orig[k], ], p = 1, type = "const")
  FC$VAR[k, ] <- lp_t[k] + cumsum(predict(fit, n.ahead = H)$fcst$AAPL[, "fcst"]) })[["elapsed"]]

# GARCH(1,1) with Student-t errors and CONSTANT mean (the AR(1) term vanished in Part 4), re-estimated at every origin
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = TRUE), distribution.model = "std")
FC$GARCH <- matrix(NA_real_, K, H); LO_g <- HI_g <- matrix(NA_real_, K, H); n_fail <- 0
t_garch <- system.time(for (k in 1:K) {
  fit <- tryCatch(ugarchfit(spec, ret[1:orig[k]] * 100, solver = "hybrid"), error = function(e) NULL)
  if (is.null(fit) || convergence(fit) != 0) { n_fail <- n_fail + 1; FC$GARCH[k, ] <- FC$RW_drift[k, ]; next }   # fallback, counted
  fc <- ugarchforecast(fit, n.ahead = H)
  mu_r <- as.numeric(fitted(fc)) / 100; sg <- as.numeric(sigma(fc)) / 100
  FC$GARCH[k, ] <- lp_t[k] + cumsum(mu_r)
  half <- 1.96 * sqrt(cumsum(sg^2))                    # sum of daily variances; ~normal for multi-day sums
  half[1] <- qdist("std", p = 0.975, shape = coef(fit)["shape"]) * sg[1]   # day 1: exact t quantile
  LO_g[k, ] <- FC$GARCH[k, ] - half; HI_g[k, ] <- FC$GARCH[k, ] + half })[["elapsed"]]
cat("GARCH fits that failed to converge (replaced by RW+drift):", n_fail, "of", K, "\n")

# Networks: trained ONCE on the training period (not refitted), forecasts read from Python
FC$LSTM <- matrix(py$lstm_lp, K, H, byrow = TRUE); FC$GRU <- matrix(py$gru_lp, K, H, byrow = TRUE)

# ---- 3. ONE SUMMARY TABLE (price scale), h = 1, 5, 10
metr <- function(y_lp, p_lp) { y <- exp(y_lp); p <- exp(p_lp); e <- y - p
  c(MSE = mean(e^2), RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)), MAPE = 100 * mean(abs(e / y))) }
tab <- do.call(rbind, lapply(HS, function(h) { m <- t(sapply(names(FC), function(nm) metr(actual[, h], FC[[nm]][, h])))
  data.frame(h = h, model = rownames(m), round(m, 4), row.names = NULL) }))
for (h in HS) { cat("\n=== Horizon h =", h, "(", K, "rolling origins ) ===\n"); print(tab[tab$h == h, -1][order(tab$RMSE[tab$h == h]), ], row.names = FALSE) }
write.csv(tab, "data/metrics_all_R.csv", row.names = FALSE)

# ---- 4. DIEBOLD-MARIANO TESTS (loss = squared error of the LOG price, which is scale-free; dm.test applies the Harvey small-sample
#         correction and accounts for the overlap of h-step errors). H0: equal forecast accuracy. Ratio < 1 => first model more accurate.
dm_cmp <- function(a, b, h) {
  e1 <- actual[, h] - FC[[a]][, h]; e2 <- actual[, h] - FC[[b]][, h]
  r <- tryCatch(dm.test(e1, e2, alternative = "two.sided", h = h, power = 2), error = function(e) NULL)
  data.frame(h = h, model = a, versus = b, MSE_ratio = round(mean(e1^2) / mean(e2^2), 4),
             DM = if (is.null(r)) NA else round(unname(r$statistic), 3), p = if (is.null(r)) NA else round(r$p.value, 4)) }
bench <- "RW_drift"
dm1 <- do.call(rbind, lapply(HS, function(h) do.call(rbind, lapply(setdiff(names(FC), bench), dm_cmp, b = bench, h = h))))
dm1$p_holm <- round(p.adjust(dm1$p, "holm"), 4)        # many tests at once: some "significant" p-values arise by chance, so also report Holm-adjusted ones
cat("\n=== Diebold-Mariano: each model vs the RW+drift benchmark ===\n"); print(dm1, row.names = FALSE)
pairs <- list(c("LSTM", "GRU"), c("GARCH", "ARIMA"), c("VAR", "ARIMA"), c("RW_drift", "RW"))
dm2 <- do.call(rbind, lapply(HS, function(h) do.call(rbind, lapply(pairs, function(p) dm_cmp(p[1], p[2], h)))))
cat("\n=== Diebold-Mariano: selected pairs ===\n"); print(dm2, row.names = FALSE)
write.csv(rbind(cbind(dm1[, 1:6], p_holm = dm1$p_holm), cbind(dm2, p_holm = NA)), "data/dm_tests_R.csv", row.names = FALSE)

# ---- 5. INTERVAL CALIBRATION (ARIMA vs GARCH): is the nominal 95% close to the empirical coverage?
cov <- do.call(rbind, lapply(HS, function(h) data.frame(h = h,
  cover_ARIMA = mean(actual[, h] >= LO_a[, h] & actual[, h] <= HI_a[, h], na.rm = TRUE),
  cover_GARCH = mean(actual[, h] >= LO_g[, h] & actual[, h] <= HI_g[, h], na.rm = TRUE),
  width_ARIMA = mean(exp(HI_a[, h]) - exp(LO_a[, h]), na.rm = TRUE), width_GARCH = mean(exp(HI_g[, h]) - exp(LO_g[, h]), na.rm = TRUE))))
cat("\n=== 95% interval coverage and mean width (price units) ===\n"); print(round(cov, 3), row.names = FALSE)
write.csv(cov, "data/coverage_R.csv", row.names = FALSE)

# ---- 6. RMSE BY HORIZON, relative to the benchmark (below 1 = better than RW+drift)
rm_h <- sapply(names(FC), function(nm) sapply(1:H, function(h) sqrt(mean((exp(actual[, h]) - exp(FC[[nm]][, h]))^2))))
png("figures/R08_rmse_by_horizon.png", 900, 500)
matplot(1:H, sweep(rm_h, 1, rm_h[, bench], "/"), type = "b", pch = 1:ncol(rm_h), lty = 1, col = 1:ncol(rm_h),
        xlab = "horizon (days)", ylab = "RMSE / RMSE of RW+drift", main = "Forecast accuracy relative to the benchmark")
abline(h = 1, lty = 2); legend("topleft", names(FC), col = 1:ncol(rm_h), pch = 1:ncol(rm_h), lty = 1, ncol = 2); dev.off()

# ---- 7. COST AND INTERPRETABILITY (complete the 'neural' runtimes from the Python log)
cost <- data.frame(model = names(FC),
  parameters = c(0, 1, 3, 6, 5, 4938, 1178),
  interpretability = c("total", "total", "high", "high", "high (variance equation)", "low (black box)", "low (black box)"),
  seconds_for_all_origins = c(0, 0, round(t_arima), round(t_var), round(t_garch), NA, NA))
cat("\n=== Complexity / cost ===\n"); print(cost, row.names = FALSE); write.csv(cost, "data/cost_R.csv", row.names = FALSE)
cat("\nReminder: ARIMA / VAR / GARCH are re-estimated at EVERY origin; LSTM / GRU are trained once. Say so in the report.\n")
