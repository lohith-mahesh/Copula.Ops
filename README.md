# Copula.Ops | Statistical Arbitrage Terminal

A high-frequency research environment designed for alpha discovery within the Nifty 750 universe. This project moves beyond static OLS (Ordinary Least Squares) models by utilizing state-space estimation and tail-dependency modeling to identify high-probability mean-reversion opportunities.

![Terminal Dashboard](Demo.png)

## Core Technical Architecture

### 1. Dynamic State Estimation (Kalman Filter)
The engine utilizes a Recursive Kalman Filter to estimate the hedge ratio ($\beta$) between two assets. Unlike static regressions, this approach treats $\beta$ as a hidden state that evolves over time. 
* **State Transition:** $x_t = A x_{t-1} + w_t$
* **Observation:** $z_t = H x_t + v_t$
This allows the model to adjust for changing market conditions and maintain a more accurate residual calculation by accounting for non-stationary relationships between the stock pair.

### 2. Tail-Dependency Modeling (Gaussian Copula)
To account for the non-normal distribution and fat-tails often found in financial spreads, the model applies a Probability Integral Transform (PIT) to the residuals. 
* Residuals are mapped into a uniform space $[0, 1]$.
* The engine uses a Gaussian Copula to determine the "Copula Probability."
* This provides a more rigorous measure of divergence than a standard Z-score by identifying true statistical extremes regardless of the underlying distribution's shape.

### 3. Walk-Forward Validation
To mitigate look-ahead bias and overfitting, the engine implements a 75/25 temporal split across the two-year lookback:
* **Formation Period (75%):** Identifies initial cointegration via the Engle-Granger test ($ADF$ test on residuals).
* **Validation Period (25%):** Verifies that the cointegration relationship survives a regime shift.
Pairs that fail to maintain stationarity in the out-of-sample period are automatically discarded, ensuring the identified alpha is not a result of "statistical ghosts."

### 4. Lead-Lag Detection (Cross-Correlation)
The system calculates the cross-correlation of log returns across a range of 11 lags ($-5$ to $+5$ days). 
* $Corr(r_{1,t-l}, r_{2,t})$
This identifies whether one asset consistently leads the other, providing a directional bias for entries and exits rather than assuming a simultaneous reaction.

### 5. Monte Carlo Projection & Friction
The terminal runs 500-path simulations for selected pairs to estimate the probability of reversion within a 15-day window. These simulations incorporate a "Dynamic Friction" penalty, which scales transaction costs relative to the historical volatility of the spread to ensure realistic projections of net profit potential.

## Data Pipeline & Rigor

### Universe Construction
The scanner builds a ~730+ stock universe by merging the Nifty 500 and the Nifty Smallcap 250 indices. Data is pulled via the YFinance API with a 50-stock batching protocol to manage memory consumption and API rate limits.

### Multiple Testing Bias
Given the $N^2$ complexity of a 750-stock matrix, the engine tests over 260,000 potential combinations. To prevent Type I errors, the system applies a **Bonferroni Correction** to the cointegration p-values:
* Adjusted $\alpha = \frac{0.05}{N_{unique\_idx}}$

## Logic Stack
* **Language:** Python 3.11+
* **Backend:** FastAPI (Asynchronous endpoint handling)
* **Frontend:** Vanilla JS, Plotly.js, Bootstrap 5 (Responsive Dark Theme)
* **Statistics:** Statsmodels (ADF/Coint), SciPy (Copula/Kurtosis), NumPy (Vectorized Math)

## Deployment
The project is containerized via Docker and optimized for environments with high-RAM availability (16GB+) to handle the computational overhead of the correlation matrix and the iterative Kalman recursion.
