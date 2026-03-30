# Copula.Ops | Statistical Arbitrage Terminal

A high-fidelity research environment designed for alpha discovery within the Nifty 750 universe. This terminal identifies high-probability mean-reversion opportunities by moving beyond static models toward state-space estimation and tail-dependency analysis.

![Terminal Dashboard](Demo.png)

## Core Quantitative Architecture

### 1. Dynamic State Estimation (Kalman Filter)
[cite_start]The engine utilizes a Recursive Kalman Filter to estimate the hedge ratio between two assets. [cite_start]Unlike static Ordinary Least Squares (OLS) regressions, this approach treats the hedge ratio as a hidden state that evolves over time. [cite_start]This allows the model to adjust for changing market regimes and maintain accurate residuals.

* [cite_start]**State Transition**: $$x_t = A x_{t-1} + w_t$$ 
* [cite_start]**Observation**: $$z_t = H x_t + v_t$$ 

### 2. Tail-Dependency Modeling (Empirical Copula)
[cite_start]To account for non-normal distributions and fat-tails in financial spreads, the model applies a Probability Integral Transform (PIT) to the residuals. 
* [cite_start]**Uniform Mapping**: Residuals are mapped into a uniform space $[0, 1]$.
* [cite_start]**Copula Score**: Calculated using rank-order distributions to determine the "Copula Probability".
* [cite_start]**Objective**: This provides a rigorous measure of divergence by identifying statistical extremes regardless of the underlying distribution's shape.

### 3. Walk-Forward Validation
[cite_start]To mitigate look-ahead bias and overfitting, the engine implements a 75/25 temporal split across the two-year lookback:
* [cite_start]**Formation Period (75%)**: Identifies initial cointegration via the Engle-Granger test.
* [cite_start]**Validation Period (25%)**: Verifies that the cointegration relationship survives a regime shift.
* [cite_start]**Filtering**: Pairs failing to maintain stationarity in the out-of-sample period are discarded.

### 4. Lead-Lag Detection
[cite_start]The system calculates cross-correlation of log returns across 11 lags from -5 to +5 days.
* [cite_start]**Formula**: $$Corr(r_{1,t-l}, r_{2,t})$$ 
* [cite_start]**Outcome**: This identifies whether one asset leads the other, providing a directional bias for entries.

---

## Operational Infrastructure

### Data Pipeline and Universe
* [cite_start]**Construction**: Merges the Nifty 500 and Nifty Smallcap 250 lists directly from NSE archives.
* [cite_start]**Lookback**: Fetches 2 years of daily data (approximately 500 trading days).
* [cite_start]**Batching**: Downloads via yfinance in 50-stock chunks to manage rate limits and memory.
* [cite_start]**Background Sync**: A monitor loop refreshes the entire universe and re-calculates cointegration every 4 hours.

### Execution Logic
* [cite_start]**Signal Generation**: Entry signals are based on a 30-day rolling Z-score of the Kalman residuals.
* [cite_start]**Monte Carlo Projection**: Runs 500-path simulations using the Euler-Maruyama method to estimate reversion probability within a 15-day window.
* [cite_start]**Friction Model**: Simulations incorporate a "Dynamic Friction" penalty based on historical volatility to account for slippage and costs.
* [cite_start]**Dynamic Sizing**: Calculates position sizes (BUY X / SELL Y) to neutralize the spread based on the live Kalman hedge ratio.

### Multiple Testing Bias
[cite_start]Given the $N^2$ complexity of a 750-stock matrix, the system applies a **Bonferroni Correction** to prevent Type I errors.
* [cite_start]**Adjusted Alpha**: $$\alpha = 0.05 / N_{unique\_idx}$$ 

---

## Technical Stack

* [cite_start]**Backend**: FastAPI for asynchronous endpoint handling.
* [cite_start]**Processing**: NumPy, Pandas, Statsmodels, and SciPy for vectorized math and statistics[cite: 1, 3].
* **Frontend**: Vanilla JS, Plotly.js, and Bootstrap 5.
* [cite_start]**API Stability**: Custom `SafeJSON` class to handle `NaN` and `Inf` values inherent in quant data.
* [cite_start]**Deployment**: Containerized via Docker using Python 3.11-slim[cite: 2].

## Deployment
```bash
docker build -t copula-ops .
docker run -p 7860:7860 copula-ops
