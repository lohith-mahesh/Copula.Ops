# Copula.Ops | Statistical Arbitrage Terminal

A high-frequency research environment designed for alpha discovery within the Nifty 750 universe. This project moves beyond static OLS (Ordinary Least Squares) models by utilizing state-space estimation and tail-dependency modeling to identify high-probability mean-reversion opportunities.

![Terminal Dashboard](Demo.png)

## Core Technical Architecture

### 1. Dynamic State Estimation (Kalman Filter)
The engine utilizes a Recursive Kalman Filter to estimate the hedge ratio ($\beta$) between two assets. Unlike static regressions, this approach treats $\beta$ as a hidden state that evolves over time. This allows the model to adjust for changing market conditions and maintain a more accurate residual (spread) calculation.

### 2. Tail-Dependency Modeling (Gaussian Copula)
To account for the non-normal distribution and fat-tails often found in financial spreads, the model applies a Probability Integral Transform (PIT) to the residuals. 
* It maps residuals into a uniform space $[0, 1]$.
* It uses a Gaussian Copula to determine the "Copula Probability."
* This provides a more rigorous measure of divergence than a standard Z-score by identifying true statistical extremes.

### 3. Walk-Forward Validation
To mitigate look-ahead bias and overfitting, the engine implements a 75/25 temporal split:
* **Formation Period (75%):** Identifies initial cointegration via the Engle-Granger test.
* **Validation Period (25%):** Verifies that the cointegration relationship survives a regime shift.
Pairs that fail to maintain stationarity in the out-of-sample period are automatically discarded.

### 4. Monte Carlo Projection & Friction
The terminal runs 500-path simulations for selected pairs to estimate the probability of reversion within a 15-day window. These simulations incorporate a "Dynamic Friction" penalty, which scales transaction costs relative to the historical volatility of the spread to ensure realistic projections.

## Logic Stack
* **Language:** Python 3.11+
* **Backend:** FastAPI (Asynchronous endpoint handling)
* **Frontend:** Vanilla JS, Plotly.js, Bootstrap 5 (Responsive Dark Theme)
* **Data Source:** YFinance API (NSE India)
* **Statistics:** Statsmodels (ADF/Coint), SciPy (Copula/Kurtosis)

## Execution Parameters
* **Universe:** Nifty 500 + Nifty Smallcap 250 (De-duplicated)
* **Lookback:** 2 Years of historical daily close data
* **Search Rigor:** Bonferroni Correction applied to p-values to account for multiple testing bias across the 750-stock matrix.
* **Sizing:** Real-time dollar-neutral position sizing based on user-defined portfolio budget and current Kalman hedge ratio.

## Deployment
The project is containerized via Docker and optimized for environments with high-RAM availability (16GB+) to handle the $N^2$ complexity of the 750-stock correlation matrix.
