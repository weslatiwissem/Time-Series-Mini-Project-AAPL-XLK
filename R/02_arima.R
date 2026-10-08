# PART 2 (R) - ARIMA / SARIMA identification, residual diagnostics, h=10 forecast
# Run AFTER part1_data_eda.R. Install: install.packages(c("forecast","tseries","xts","zoo"))
suppressPackageStartupMessages({library(forecast); library(tseries); library(xts); library(zoo)})
set.seed(42); H <- 10                                  # fixed seed; horizon within the brief's 5..10
dir.create("figures", showWarnings=FALSE)

# ---- 1. LOAD + SAME 90/10 SPLIT AS PART 1 (same n, same floor(0.9 n))
px <- as.xts(read.zoo("data/prices_R.csv", header=TRUE, sep=",", index.column=1))
lp <- as.numeric(log(px$AAPL))[-1]                     # drop first obs: aligned with the return series of Part 1
n <- length(lp); cut <- floor(0.9 * n)
train <- lp[1:cut]; test <- lp[(cut+1):n]

# ---- 2. MANUAL IDENTIFICATION from ACF/PACF of the differenced series (d=1 from ADF/KPSS)
r <- diff(train); bound <- 1.96 / sqrt(length(r))      # 95% white-noise band
ac <- acf(r, 30, plot=FALSE)$acf[-1]; pc <- pacf(r, 30, plot=FALSE)$acf
cat("Significant ACF lags :", which(abs(ac) > bound), "\n")
cat("Significant PACF lags:", which(abs(pc) > bound), "\n")
cat("(~5% of 30 lags exceed the band by chance: isolated lags are not evidence of structure)\n")
print(Box.test(r, 10, "Ljung-Box")$p.value); print(Box.test(r, 20, "Ljung-Box")$p.value)  # H0: white noise

# ---- 3. SEARCH: (a) IC grid, (b) auto.arima -> compare both approaches
grid <- do.call(rbind, lapply(0:3, function(p) do.call(rbind, lapply(0:3, function(q) {
  f <- tryCatch(Arima(train, order=c(p,1,q), include.drift=TRUE), error=function(e) NULL)
  if (is.null(f)) NULL else data.frame(p=p, d=1, q=q, AIC=f$aic, BIC=f$bic) }))))
grid <- grid[order(grid$BIC), ]; cat("\nTop 5 by BIC (favours parsimony):\n"); print(head(grid, 5), row.names=FALSE)
bp <- grid$p[1]; bq <- grid$q[1]

auto <- auto.arima(train, d=1, seasonal=FALSE, stepwise=FALSE, approximation=FALSE, ic="bic")
cat("\nauto.arima (non-seasonal):", paste(arimaorder(auto)[1:3], collapse=","), "| BIC =", round(auto$bic, 2), "\n")
# SARIMA check: only plausible seasonality in daily trading data is the weekly cycle s=5
sauto <- auto.arima(ts(train, frequency=5), d=1, D=0, seasonal=TRUE, ic="bic")
cat("auto.arima (seasonal s=5):", as.character(sauto), "| BIC =", round(sauto$bic, 2), "\n")
cat("-> SARIMA only justified if its BIC is clearly lower than the non-seasonal model.\n")

# ---- 4. FINAL MODEL + FULL RESIDUAL DIAGNOSTICS
arch_lm <- function(e, k) {                            # ARCH-LM by hand (no extra package): regress e^2 on k lags, n*R2 ~ chi2(k)
  e2 <- e^2; m <- embed(e2, k + 1); R2 <- summary(lm(m[, 1] ~ m[, -1]))$r.squared
  1 - pchisq(nrow(m) * R2, k) }
diagnose <- function(fit, name) {
  res <- as.numeric(residuals(fit)); k <- length(fit$coef) - ("drift" %in% names(fit$coef))
  cat("\n=== Diagnostics", name, "===\n")
  cat("Ljung-Box p (lag10, lag20):", Box.test(res, 10, "Ljung-Box", fitdf=k)$p.value,
      Box.test(res, 20, "Ljung-Box", fitdf=k)$p.value, "\n")       # H0: no residual autocorrelation
  cat("Jarque-Bera p:", jarque.bera.test(res)$p.value, "| Shapiro p:", shapiro.test(res)$p.value, "\n")  # H0: normal
  cat("ARCH-LM p (5, 10 lags):", arch_lm(res, 5), arch_lm(res, 10), "\n")                           # H0: constant variance
  png(paste0("figures/R04_diag_", name, ".png"), 1100, 700); par(mfrow=c(2,2))
  plot(res, type="l", main="Residuals"); Acf(res, 30, main="ACF residuals")
  Acf(res^2, 30, main="ACF squared residuals (ARCH check)"); qqnorm(res); qqline(res); dev.off()
}
fit_grid <- Arima(train, order=c(bp,1,bq), include.drift=TRUE); diagnose(fit_grid, "grid_BIC")
diagnose(auto, "auto"); FIT <- fit_grid                # choose one for forecasting; justify in report

# ---- 5. h=10 FORECAST vs RANDOM-WALK-WITH-DRIFT BASELINE
fc <- forecast(FIT, h=H, level=95)
mu <- as.numeric(fc$mean); lo <- as.numeric(fc$lower); hi <- as.numeric(fc$upper)
rw <- train[cut] + mean(r) * (1:H); actual <- test[1:H]
metrics <- function(y, yhat) { y <- exp(y); yhat <- exp(yhat); e <- y - yhat   # on PRICE scale (interpretable)
  c(MSE=mean(e^2), RMSE=sqrt(mean(e^2)), MAE=mean(abs(e)), MAPE=100*mean(abs(e/y))) }
tab <- rbind(ARIMA=metrics(actual, mu), RW_drift=metrics(actual, rw))
cat("\nForecast accuracy, h =", H, "(price scale):\n"); print(round(tab, 4))
cat("95% interval coverage:", mean(actual >= lo & actual <= hi), "\n")
cat("Caution: 10 points = very noisy comparison. Section 6 uses rolling-origin evaluation + Diebold-Mariano.\n")
write.csv(data.frame(actual_lp=actual, arima_lp=mu, lo=lo, hi=hi, rw_lp=rw), "data/forecast_arima_R.csv", row.names=FALSE)
write.csv(tab, "data/metrics_arima_R.csv")
