# PART 1 (R) - Data import, preparation and exploratory analysis
# Pair: AAPL (stock) + XLK (sector ETF). Daily, 2019-01-01 -> 2026-09-30.
# Install: install.packages(c("quantmod","xts","tseries","forecast","urca","moments"))
suppressPackageStartupMessages({library(quantmod); library(xts); library(tseries)
  library(forecast); library(moments)})

set.seed(42)                                   # fixed seed for reproducibility
tickers <- c("AAPL","XLK"); start <- "2019-01-01"; end <- "2026-09-30"
dir.create("data", showWarnings=FALSE); dir.create("figures", showWarnings=FALSE)

# ---- 1. IMPORT (cached to CSV so re-runs are offline and identical)
if (file.exists("data/prices_R.csv")) {
  px <- as.xts(read.zoo("data/prices_R.csv", header=TRUE, sep=",", index.column=1))
} else {
  getSymbols(tickers, src="yahoo", from=start, to=end, auto.assign=TRUE)
  # Ad() = adjusted close (dividends/splits); inner join keeps common trading days only
  px <- na.omit(merge(Ad(AAPL), Ad(XLK))); colnames(px) <- tickers
  write.zoo(px, "data/prices_R.csv", sep=",")
}

# ---- 2. DERIVED INDICATORS
lp  <- log(px)                                          # log prices: variance-stabilising
ret <- na.omit(diff(lp))                                # log returns
ma20 <- rollapply(px$AAPL, 20, mean, align="right")     # 20-day MA (trend indicator)
rv20 <- rollapply(ret$AAPL, 20, sd, align="right") * sqrt(252)  # annualised realised vol

# ---- 3. CHRONOLOGICAL 90/10 SPLIT (never random)
n <- nrow(ret); cut <- floor(0.9 * n)
tr_lp <- lp[2:(cut+1)]; tr_ret <- ret[1:cut]; te_ret <- ret[(cut+1):n]
cat("Train:", format(index(tr_ret)[1]), "->", format(index(tr_ret)[cut]), "\n")
cat("Test :", format(index(te_ret)[1]), "->", format(index(te_ret)[nrow(te_ret)]), "\n")
# Everything below uses TRAIN ONLY, to avoid leaking test information into model choice.

# ---- 4. PLOTS
png("figures/R01_overview.png", 1100, 800)
par(mfrow=c(3,1)); plot.zoo(tr_lp, screens=1, col=1:2, main="Log adjusted close"); legend("topleft", tickers, col=1:2, lty=1)
plot.zoo(tr_ret, screens=1, col=1:2, main="Log returns")
plot.zoo(rv20[index(tr_ret)], main="20-day realised volatility AAPL"); dev.off()
cat("Return correlation (train):", round(cor(tr_ret$AAPL, tr_ret$XLK), 3), "\n")

# ---- 5. ACF / PACF: level (unit root pattern) vs differenced (near white noise)
png("figures/R02_acf_pacf.png", 1100, 700); par(mfrow=c(2,2))
Acf(tr_lp$AAPL, 40, main="ACF log price"); Pacf(tr_lp$AAPL, 40, main="PACF log price")
Acf(tr_ret$AAPL, 40, main="ACF log return"); Pacf(tr_ret$AAPL, 40, main="PACF log return"); dev.off()

# ---- 6. STL decomposition (robust=TRUE resists outliers such as March 2020; ~252 days/year)
y <- ts(as.numeric(tr_lp$AAPL), frequency=252)
png("figures/R03_stl.png", 1100, 700); plot(stl(y, s.window="periodic", robust=TRUE)); dev.off()

# ---- 7. OUTLIERS (3*IQR fence for extreme returns; decision = KEEP, they are real events)
r <- as.numeric(tr_ret$AAPL); q <- quantile(r, c(.25,.75)); iqr <- diff(q)
out <- r < q[1]-3*iqr | r > q[2]+3*iqr
cat("Extreme returns:", sum(out), "of", length(r), "\n")
print(head(sort(abs(setNames(r[out], format(index(tr_ret)[out]))), decreasing=TRUE), 10))

# ---- 8. DESCRIPTIVE STATS (reused in LLM section)
for (t in tickers) { x <- as.numeric(tr_ret[, t])
  cat(sprintf("%s mean=%.6f var=%.6f skew=%.3f exkurt=%.3f JB p=%.4g\n", t, mean(x), var(x),
      skewness(x), kurtosis(x)-3, jarque.bera.test(x)$p.value)) }

# ---- 9. STATIONARITY: ADF (H0 unit root) + KPSS (H0 stationary): opposite nulls, so use both
st <- function(x, name, kpss_null="Level") {
  x <- as.numeric(x)
  a <- adf.test(x)$p.value; k <- kpss.test(x, null=kpss_null)$p.value
  v <- if (a < .05 && k >= .05) "STATIONARY" else if (a >= .05 && k < .05) "NON-STATIONARY" else "INCONCLUSIVE"
  cat(sprintf("%-16s ADF p=%.4f | KPSS p=%.4f -> %s\n", name, a, k, v)) }
for (t in tickers) { st(tr_lp[, t], paste("log price", t), "Trend"); st(tr_ret[, t], paste("log return", t)) }
# Expected: log price I(1) => d = 1; log return stationary.
write.zoo(merge(lp, ret), "data/prepared_R.csv", sep=",")

install.packages("png")   # once
library(png)
for (f in list.files("figures", "^R0.*png$", full.names = TRUE)) {
  grid::grid.newpage(); grid::grid.raster(readPNG(f))
}
