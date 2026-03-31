# Copula.Ops | Statistical Arbitrage Terminal

A high-fidelity research environment designed for alpha discovery within the Nifty 750 universe. This terminal identifies high-probability mean-reversion opportunities by moving beyond static models toward state-space estimation and tail-dependency analysis.

![Terminal Dashboard](Demo.png)

## Core Quantitative Architecture

### 1. Dynamic State Estimation (Kalman Filter)
The engine utilizes a Recursive Kalman Filter to estimate the hedge ratio between two assets. Unlike static Ordinary Least Squares (OLS) regressions, this approach treats the hedge ratio as a hidden state that evolves over time. This allows the model to adjust for changing market regimes and maintain accurate residuals.

* **State Transition**: $$x_t = A x_{t-1} + w_t$$
* **Observation**: $$z_t = H x_t + v_t$$

### 2. Tail-Dependency Modeling (Empirical Copula)
To account for non-normal distributions and fat-tails in financial spreads, the model applies a Probability Integral Transform (PIT) to the residuals. 
* **Uniform Mapping**: Residuals are mapped into a uniform space [0, 1].
* **Copula Score**: Calculated using rank-order distributions to determine the "Copula Probability".
* **Objective**: This provides a rigorous measure of divergence by identifying statistical extremes regardless of the underlying distribution's shape.

### 3. Hurst Exponent Analysis
To quantify the strength of the mean-reverting behavior, the pipeline calculates the Hurst exponent ($H$) on the spread before ranking.
* **Thresholds**: $H < 0.5$ indicates a mean-reverting series, while $H > 0.5$ suggests a trending series.
* **Implementation**: Computed using the variance of differences at multiple lag windows (from 2 to 19 days) via log-log regression.
* **Weighting**: This metric accounts for up to 30% of the pair's final validity score, heavily penalizing spreads that wander rather than revert.

### 4. Kurtosis Scoring (Tail Magnitude)
Financial spreads often exhibit non-normal distributions. The system actively rewards pairs with high kurtosis (fat tails) in their spread distribution.
* **Objective**: Higher kurtosis implies that when the spread deviates, the magnitude of the divergence (and therefore the potential arbitrage profit) is larger.
* **Weighting**: The kurtosis metric contributes up to 20% of the final validity score, prioritizing explosive mean-reversion setups over tight, low-volatility tracking.

### 5. Walk-Forward Validation
To mitigate look-ahead bias and overfitting, the engine implements a 75/25 temporal split across the two-year lookback:
* **Formation Period (75%)**: Identifies initial cointegration via the Engle-Granger test.
* **Validation Period (25%)**: Verifies that the cointegration relationship survives a regime shift.
* **Filtering**: Pairs failing to maintain stationarity in the out-of-sample period are discarded.

### 6. Lead-Lag Detection
The system calculates cross-correlation of log returns across 11 lags from -5 to +5 days.
* **Formula**: $$Corr(r_{1,t-l}, r_{2,t})$$
* **Outcome**: This identifies whether one asset leads the other, providing a directional bias for entries.

---

## Operational Infrastructure

### Data Pipeline and Universe
* **Construction**: Merges the Nifty 500 and Nifty Smallcap 250 lists directly from NSE archives.
* **Lookback**: Fetches 2 years of daily data (approximately 500 trading days).
* **Batching**: Downloads via yfinance in 50-stock chunks to manage rate limits and memory.
* **Background Sync**: A monitor loop refreshes the entire universe and re-calculates cointegration every 4 hours.

### Execution Logic
* **Signal Generation**: Entry signals are based on a 30-day rolling Z-score of the Kalman residuals.
* **Monte Carlo Projection**: Runs 500-path simulations using the Euler-Maruyama method to estimate reversion probability within a 15-day window.
* **Friction Model**: Simulations incorporate a "Dynamic Friction" penalty based on historical volatility to account for slippage and costs.
* **Dynamic Sizing**: Calculates position sizes (BUY X / SELL Y) to neutralize the spread based on the live Kalman hedge ratio.

### Multiple Testing Bias
Given the complexity of a 750-stock matrix, the system applies a Bonferroni Correction to prevent Type I errors.
* **Adjusted Alpha**: $$\alpha = 0.05 / N_{unique\_idx}$$

---

## Technical Stack

* **Backend**: FastAPI for asynchronous endpoint handling.
* **Processing**: NumPy, Pandas, Statsmodels, and SciPy for vectorized math and statistics.
* **Frontend**: Vanilla JS, Plotly.js, and Bootstrap 5.
* **API Stability**: Custom SafeJSON class to handle NaN and Inf values inherent in quant data.
* **Deployment**: Containerized via Docker using Python 3.11-slim.

## Deployment
```bash
docker build -t copula-ops .
docker run -p 7860:7860 copula-ops
