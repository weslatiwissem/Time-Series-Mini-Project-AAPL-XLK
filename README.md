# Time Series Mini-Project: AAPL & XLK (TEK-UP, Data Science & AI)

Modeling, forecasting and diagnostics of two correlated financial series: **Apple (AAPL)** and its sector ETF **XLK**.
Theme: Finance (Option A). Data: Yahoo Finance adjusted close, daily, 2019-01-01 to 2026-09-30.
Brief: `Mini_Project_Time_Series2_1st_periode_2026_SDIAE.pdf` (not in the repo; ask the group).

> **Deadline / contact**: the brief contradicts itself (p.1: 12 Nov 2025, p.8: **12 Oct 2026, 23:55**; e-mail `tekup.de` vs `tek-up.de`). Confirm with the instructor before sending.
> **Oral defense**: projects are drawn at random; EVERY member must be able to explain ALL code and choices.

## What is this project about? (plain-language version)

**The question.** Can we predict where a stock's price goes next, and, just as important, can we measure *how uncertain* that prediction is? We study two things that move together: **Apple's stock (AAPL)** and the **technology-sector fund (XLK)** that Apple is part of. We use about 8 years of daily prices.

**The idea of the course.** A *time series* is a list of values ordered in time (here, one price per trading day). Forecasting it is not about applying one magic formula. It is about trying different families of models, **checking whether each one is actually appropriate** (the "diagnostics"), and comparing them fairly on data they have never seen. The grade rewards honest checking more than the number of models.

**How we work, step by step:**

1. **Prepare the data.** Download the prices, compute daily returns (the percentage change from one day to the next), and split the history in time order: the first 90% to build the models, the last 10% (about 195 days) to test them. We never mix the two, otherwise we would be "predicting" what we already saw.
2. **Explore.** Plot the series, look for trends and repeating patterns, and test whether the series is *stationary*. A stationary series behaves the same way over time (stable average and spread). Prices are not stationary because they wander upward, but daily returns are. This tells us we should model returns, or differenced prices.
3. **Classical statistical models.**
   - **ARIMA** predicts tomorrow from today's and yesterday's values. We found that the best version is almost a *random walk with drift*: tomorrow's price is today's plus a small upward push. That is typical of stocks, because they are hard to predict.
   - **VAR** lets AAPL and XLK predict each other. **Granger causality** asks: does yesterday's XLK help predict today's AAPL, beyond AAPL's own history? (It means "useful for prediction", not "is the cause of".)
   - **Cointegration** asks whether two wandering series are tied together by a long-run "leash", so that they can drift apart for a while but always come back. We tested it for AAPL and XLK and did not find it.
   - **GARCH** models the *volatility* (how wildly the price swings). Volatility comes in clusters: calm periods followed by turbulent ones, such as the March 2020 crash. GARCH follows this and lets us give prediction ranges that are narrow in calm times and wide in turbulent times.
4. **Neural networks (still to do).** LSTM and GRU are networks designed for sequences. We will test them with *walk-forward validation* (train on the past, predict the next period, move forward, repeat), as the correct form of cross-validation for time series.
5. **Compare everything fairly.** Same test data, same error measures (MSE, RMSE, MAE, MAPE), and a **Diebold-Mariano test** to check whether one model's advantage is real or just luck.
6. **Use an AI assistant (LLM) critically.** We ask a chatbot to suggest hypotheses and explain our results, **after** writing our own interpretation, then check where it is wrong or invents things.

**What we found so far (in simple words).**
- Apple's daily returns are close to unpredictable. The best simple model is barely better than "tomorrow equals today plus a small drift".
- What *is* predictable is **volatility**: after a big move, big moves tend to follow. A GARCH model captures this and its checks pass.
- Returns have **fat tails**: extreme days happen far more often than a bell curve would suggest.
- AAPL and XLK move almost together day by day (correlation 0.83), but there is no stable long-run "leash" between their prices.
- On a 10-day test, the simplest forecast beat the fancier ones. This is not conclusive yet because 10 days is too few.

### Mini-glossary
| Term | Meaning |
|---|---|
| Return | Percentage change in price from one day to the next |
| Stationary | Statistical behavior (average, spread) does not change over time |
| Differencing | Replacing each value by its change since the previous one, to remove a trend |
| Residuals | What the model failed to explain; they should look like pure noise |
| Ljung-Box test | Checks whether the residuals still contain a pattern |
| ARCH test | Checks whether the size of the swings changes over time |
| Volatility | How strongly the price fluctuates |
| Training / test set | Data used to build the model / data kept aside to judge it |
| Walk-forward validation | Repeatedly train on the past and test on the next period |

## Language split
- **R (RStudio) is the lead language**: all classical modeling (sections 1, 2, 3, 6).
- **Python**: advanced models (LSTM/GRU, hybrid, optional Prophet) AND a Python version of the classical core (the brief requires the classical core in BOTH languages).

## How to run
1. Open `timeseries_project.Rproj` in RStudio (sets the working directory = project root; all paths are relative).
2. R packages: `install.packages(c("quantmod","xts","zoo","tseries","forecast","urca","vars","moments","rugarch","png"))`
3. Run the R scripts in order: `R/01_data_eda.R` -> `02_arima.R` -> `03_var_cointegration.R` -> `04_garch.R`. After the Python step below, run `R/06_evaluation.R` (it needs `data/forecast_rnn_py.csv`).
4. Python (run from the project root, with a virtual environment active):
   - `pip install yfinance pandas numpy matplotlib statsmodels scipy pmdarima tensorflow`
   - `python python/01_data_eda.py`, `python python/02_arima.py` (optional: `02` also works from the R price cache)
   - **`python python/05_lstm_gru.py`** = LSTM and GRU. Smoke test first with `QUICK=1 python python/05_lstm_gru.py` (about 1 min). The full run takes about **7 min on a weak laptop CPU** (no GPU needed).
5. The first run downloads data and caches it in `data/` (R: `prices_R.csv`, Python: `prices.csv`). **Commit the cached CSVs** so everyone uses identical data. If both exist, `data/prices.csv` and `data/prices_R.csv` must contain the same prices (the evaluation script checks this).
6. Figures are written to `figures/` as files (they are not shown in the Plots pane).

### Windows setup tips (learned the hard way)
- Git Bash: activate the environment with `source .venv/Scripts/activate` (forward slashes). In cmd/PowerShell use `.venv\Scripts\activate`.
- Keep the project OUT of OneDrive if you can (syncing thousands of `.venv` files slows installs and can freeze them).
- If a `pip install` freezes and `Ctrl+C` does nothing in Git Bash: close the window, use cmd, and retry with `pip install --default-timeout=300 --no-cache-dir tensorflow`.
- TensorFlow 2.21 runs on CPU on native Windows; that is enough for this project.

## Conventions (please keep)
- Chronological 90/10 split: n = 1945 returns, train = 1750 (2019-01-03 to 2025-12-17), test = 195 (2025-12-18 to 2026-09-29). All scripts use the same alignment.
- Exploration and model selection use **train only**.
- Fixed seed `42`. One comment per block explaining *why*, not *what*.
- Every script saves its forecasts to `data/` as CSV; the evaluation script (section 6) reads them to build one comparison table.

## Status (9 Oct 2026)

| Script | Content | State |
|---|---|---|
| `R/01_data_eda.R` | data, indicators, split, ACF/PACF, STL, outliers, ADF/KPSS | run on real data |
| `R/02_arima.R` | ARIMA/SARIMA identification, diagnostics, h=10 forecast | run on real data |
| `R/03_var_cointegration.R` | VAR, Granger, Phillips-Ouliaris, Johansen (12 specifications), VECM if needed | run on real data (no cointegration found) |
| `R/04_garch.R` | ARCH test, GARCH variants, standardized-residual diagnostics, volatility plot | run on real data (section 8 sensitivity check not run yet) |
| `python/01_data_eda.py`, `02_arima.py` | Python versions of the exploration and ARIMA | written; 02 only tested on simulated data |
| **`python/05_lstm_gru.py`** | **LSTM and GRU, walk-forward validation, rolling test forecasts** | **run on real data (full version)** |
| `R/06_evaluation.R` | one metrics table for ALL models, Diebold-Mariano tests, interval coverage | **written, NOT run yet** |
| `python/03_*.py`, `python/04_*.py` | Python VAR/Granger/Johansen and GARCH (`arch` package) | **not written yet** (classical core is required in both languages) |

### LSTM / GRU in one paragraph (details and numbers in `docs/PROGRESS.md`)
Both networks predict the **next 10 daily AAPL returns** from the last 20 or 60 days of AAPL return, XLK return and 10-day volatility; price forecasts are rebuilt by cumulating the predicted returns. Hyperparameters (window, units) were chosen by **walk-forward (expanding window) validation inside the training period**, then each network was trained once and evaluated on **186 rolling forecast origins** over the test period. **Result: neither network beats a random walk with drift.** In validation their error is 0.5-1.2% *worse* than "always predict the average return", and on the test set all models are within 1-2% of each other. The 1-day direction hit rate (54-55%) is not distinguishable from the share of up-days (53%). This is an honest negative result, consistent with the weak predictability found by ARIMA and VAR.

### How to use the files
- `data/forecast_rnn_py.csv`: one row per (origin, horizon) with the actual log price and the LSTM, GRU and random-walk forecasts. `R/06_evaluation.R` merges ARIMA, VAR and GARCH forecasts at the same origins.
- Task list and owners: `docs/PROGRESS.md`.