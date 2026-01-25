# Copula.Ops

**Statistical Arbitrage Engine for NSE Pairs Trading.**

Full-stack quantitative analysis engine for identifying, analyzing, and backtesting mean-reverting stock pairs. Processes the **Nifty 500 & Microcap 250** universe (~750 stocks) to detect statistical inefficiencies using advanced econometrics.

Implements a multi-stage filtering process involving Cointegration tests, Hurst Exponents, and Ornstein-Uhlenbeck process fitting. Eliminates look-ahead bias via rolling window calculations for Z-Scores and Hedge Ratios.

## Interface

![Dashboard Demo](Demo.png)
*Real-time analysis dashboard showing rolling Z-Scores, Cointegration metrics, and interactive backtest equity curve.*

---

## Quantitative Methodology

### 1. Liquidity Filtering
Ensures execution viability before statistical analysis.
* **Criterion:** Rejects assets with median daily turnover (`Price * Volume`) < ₹5,000,000.
* **Data Quality:** Excludes assets with >20% zero-return days to filter out circuit-limited or illiquid instruments.

### 2. Pair Selection
Scans the liquid universe for high-probability mean-reversion candidates using a tiered approach:
1.  **Correlation:** Pearson correlation > 0.90.
2.  **Cointegration:** Engle-Granger two-step test. Pairs accepted only if residuals are stationary ($p < 0.05$).
3.  **Hurst Exponent:** Filters for mean-reverting random walks ($H < 0.5$).

### 3. Signal Generation
Calculated using strictly historical data windows to prevent look-ahead bias.
* **Rolling OLS:** 60-day window for dynamic Hedge Ratio ($\beta$).
* **Z-Score Calculation:** Standardized spread using rolling mean and deviation:
    $$Z_t = \frac{Spread_t - \mu_{60}}{\sigma_{60}}$$

### 4. Ornstein-Uhlenbeck Modeling
Fits spread data to the OU stochastic differential equation ($dx_t = \theta(\mu - x_t)dt + \sigma dW_t$) to derive the **Half-Life** (expected time to mean reversion).

---

## Project Structure

```text
Copula.Ops/
├── app.py              # Application entry point (FastAPI + Logic)
├── index.html          # Dashboard interface
├── requirements.txt    # Dependencies
├── Cache.csv           # Local market data storage
└── README.md           # Documentation
