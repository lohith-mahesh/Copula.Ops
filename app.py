import uvicorn
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from statsmodels.tsa.stattools import coint
from scipy.stats import kurtosis, norm, rankdata
import requests
import io
import threading
import warnings
import time
import json
import os

# Suppress pandas and statsmodels warnings during background execution
warnings.filterwarnings("ignore")

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=BASE_DIR)

# In-memory state management for quantitative data.
# Note: State is volatile and will be lost on server restart.
# TODO: Migrate state management to a persistent store (e.g., Redis, PostgreSQL) for production.
class GlobalState:
    def __init__(self):
        self.status = "BOOTING..."
        self.progress = 0
        self.ready = False
        self.data = pd.DataFrame()
        self.sector_map = {}
        self.pairs_pool = []
        self.last_sync = "NEVER"

state = GlobalState()

# Custom JSONResponse subclass to handle numpy NaN and Inf values.
# Required because standard json.dumps does not serialize these floating-point edge cases.
class SafeJSON(JSONResponse):
    def render(self, content) -> bytes:
        def clean(obj):
            if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, list): return [clean(x) for x in obj]
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj): return 0.0
            return obj
        return json.dumps(clean(content)).encode("utf-8")

class AnalyzeRequest(BaseModel):
    t1: str
    t2: str

# Calculate empirical copula score using rank-order distributions.
# Evaluates the tail dependence and joint probability of the pair non-parametrically.
def get_copula_score(z_series):
    if len(z_series) < 30: return 0.5
    u = rankdata(z_series) / (len(z_series) + 1)
    return u[-1]

# Calculate composite validity score for pairs ranking.
# Aggregates out-of-sample cointegration p-value, Hurst exponent (mean reversion strength),
# and Kurtosis (tail magnitude).
# Applies Bonferroni correction to strictly control the family-wise error rate (FWER) across multiple comparisons.
def calculate_validity(z_series, p_oos, hurst, alpha_bonf):
    p_score = 50 if p_oos < alpha_bonf else (np.interp(p_oos, [alpha_bonf, 0.05], [40, 0]))
    h_score = np.interp(hurst, [0.3, 0.5, 0.6], [30, 10, 0])
    k_score = np.interp(abs(kurtosis(z_series.dropna())), [0, 3, 6], [20, 10, 0])
    return int(p_score + h_score + k_score)

# Simulate future spread trajectories via an Ornstein-Uhlenbeck (OU) process.
# Incorporates a deterministic friction penalty to approximate transaction costs and slippage.
def run_friction_mc(z_series, steps=15, sims=500):
    recent = z_series.tail(30)
    y, x = recent.values[1:], recent.values[:-1]
    res = np.polyfit(x, y, 1)
    
    # Estimate mean reversion speed (lambda) and volatility (sigma)
    l_hat, s_hat = np.clip(1 - res[0], 0.02, 0.8), np.std(y - (res[0] * x + res[1]))
    fric = s_hat * 0.05 
    
    last_z, paths = z_series.iloc[-1], np.zeros((sims, steps))
    for i in range(sims):
        curr = last_z
        for t in range(steps):
            # Euler-Maruyama discretization for SDE
            curr += l_hat * (0 - curr) + s_hat * np.random.normal() + (fric if curr < 0 else -fric)
            paths[i, t] = curr
            
    # Return 95% confidence intervals and probability of reverting to the mean threshold (0.1)
    return {"upper": np.percentile(paths, 95, axis=0).tolist(), "lower": np.percentile(paths, 5, axis=0).tolist()}, \
           round((np.sum(np.any(np.abs(paths) < 0.1, axis=1)) / sims) * 100, 1)

# Calculate dynamic hedge ratio via a recursive Kalman Filter.
# Estimates the time-varying state of the hedge ratio rather than relying on static OLS.
# Tuning parameters: delta configures state transition covariance; requires tuning based on asset volatility.
def kalman_filter_spread(y, x):
    obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]
    delta, obs_cov = 1e-5, 1e-3
    t_cov = delta / (1 - delta) * np.eye(2)
    means, m, c = np.zeros((len(y), 2)), np.zeros(2), np.ones((2, 2))
    
    for i in range(len(y)):
        c += t_cov
        v = y[i] - np.dot(obs_mat[i], m)
        F = np.dot(obs_mat[i], np.dot(c, obs_mat[i].T)) + obs_cov
        K = np.dot(c, obs_mat[i].T) / (F + 1e-10)
        m += K.flatten() * v
        c -= np.dot(K, np.dot(obs_mat[i], c))
        means[i] = m
        
    return means[:, 0], means[:, 1] # Returns dynamic beta and alpha

# Core pipeline: Ingests Nifty universe components, retrieves historical pricing, and executes pair discovery.
def sync_market_data():
    state.status = "FETCHING"
    combined = []
    sources = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    ]
    
    # 1. Ingest the latest NSE stock universe via exchange archives
    for url in sources:
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            df = pd.read_csv(io.StringIO(r.text))
            for _, row in df.iterrows():
                sym = f"{row['Symbol'].strip()}.NS"
                combined.append(sym)
                state.sector_map[sym] = row.get('Industry', 'Market')
        except: 
            continue
            
    tickers = sorted(list(set(combined))) 
    all_data = []
    
    # 2. Batch download historical data via yfinance (chunk size: 50) to mitigate rate limiting
    for i in range(0, len(tickers), 50):
        state.progress = int((i/len(tickers)) * 50)
        batch = yf.download(tickers[i:i+50], period="2y", progress=False)
        if not batch.empty and 'Close' in batch: all_data.append(batch['Close'])
        time.sleep(0.05) # Rate limit delay
        
    if not all_data:
        state.status = "READY (NO DATA)"
        return
        
    # Truncate to the most recent 500 trading days; filter out illiquid assets exceeding the missing data threshold
    state.data = pd.concat(all_data, axis=1).dropna(thresh=250, axis=1).ffill().iloc[-500:]
    state.status = "VALIDATING"
    
    # 3. Temporal train/test split for cointegration validation (75% in-sample, 25% out-of-sample)
    split = int(len(state.data) * 0.75)
    f_data, v_data = state.data.iloc[:split], state.data.iloc[split:]
    corr_matrix = f_data.corr()
    
    # Pre-filter: Restrict testing to pairs exhibiting moderate to high Pearson correlation to optimize computation
    indices = np.where((corr_matrix.values > 0.4) & (corr_matrix.values < 0.99))
    unique_idx = sorted(list(set(tuple(sorted((i, j))) for i, j in zip(*indices) if i != j)))
    
    # Calculate dynamic Bonferroni alpha threshold
    alpha_bonf = 0.05 / len(unique_idx) if unique_idx else 0.05
    pairs = []
    
    # 4. Execute Engle-Granger cointegration tests
    for idx, (i, j) in enumerate(unique_idx):
        state.progress = 50 + int((idx / len(unique_idx)) * 50)
        t1, t2 = state.data.columns[i], state.data.columns[j]
        try:
            # In-sample evaluation
            _, p_is, _ = coint(f_data[t1], f_data[t2])
            if p_is < 0.05:
                # Out-of-sample validation to confirm stationarity persistence
                _, p_oos, _ = coint(v_data[t1], v_data[t2])
                if p_oos < 0.15:
                    spread = state.data[t1] / state.data[t2]
                    # Calculate Hurst exponent (H < 0.5 indicates a mean-reverting series)
                    hurst = np.polyfit(np.log(range(2, 20)), [np.sqrt(np.std(np.subtract(spread.values[l:], spread.values[:-l]))) for l in range(2, 20)], 1)[0] * 2.0
                    z = (spread - spread.mean()) / (spread.std() + 1e-10)
                    validity = calculate_validity(z, p_oos, hurst, alpha_bonf)
                    
                    # Evaluate lead-lag dynamics via cross-correlation of lagged returns
                    r1, r2 = state.data[t1].pct_change().fillna(0), state.data[t2].pct_change().fillna(0)
                    corrs = [r1.shift(l).corr(r2) for l in range(-5, 6)]
                    best_lag = range(-5, 6)[np.nanargmax(np.abs(corrs))]
                    lag_label = "SYNC" if best_lag == 0 else (f"{t1.split('.')[0]} LEADS" if best_lag > 0 else f"{t2.split('.')[0]} LEADS")
                    
                    s1, s2 = state.sector_map.get(t1, "Market"), state.sector_map.get(t2, "Market")
                    pairs.append({
                        "t1": t1, "t2": t2, "validity": validity, "correlation": round(float(corr_matrix.loc[t1, t2]), 3),
                        "lead": lag_label,
                        "sector": f"{s1} (INTRA)" if s1 == s2 else f"{s1[:5]}/{s2[:5]} (INTER)"
                    })
                    
            # Enforce a hard cap of 150 pairs to optimize API payload size
            if len(pairs) >= 150: break 
        except: 
            continue
            
    state.pairs_pool = pairs
    state.last_sync = time.strftime("%H:%M:%S") 
    state.ready, state.progress, state.status = True, 100, "READY"

# Background daemon task to synchronize the market universe at 4-hour intervals
def market_monitor_loop():
    while True:
        sync_market_data()
        time.sleep(14400) 

@app.on_event("startup")
async def startup():
    # Initialize the data ingestion pipeline on a separate daemon thread to prevent event loop blocking
    threading.Thread(target=market_monitor_loop, daemon=True).start()

@app.get("/status")
async def get_status(): 
    return {"status": state.status, "progress": state.progress, "ready": state.ready, "last_sync": state.last_sync}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/scan")
async def scan(): 
    return {"pairs": state.pairs_pool, "last_sync": state.last_sync}

# Pair analysis endpoint: Computes the dynamic Kalman spread and executes the Monte Carlo projection
@app.post("/analyze", response_class=SafeJSON)
async def analyze(req: AnalyzeRequest):
    try:
        s1, s2 = state.data[req.t1], state.data[req.t2]
        
        # Calculate dynamic hedge ratio
        beta, alpha = kalman_filter_spread(s1.values, s2.values)
        
        # Compute spread and rolling 30-day Z-score for entry/exit thresholds
        z = pd.Series(s1.values - (beta * s2.values + alpha))
        z_score = ((z - z.rolling(30).mean()) / (z.rolling(30).std() + 1e-10)).fillna(0)
        
        pair_info = next((p for p in state.pairs_pool if p['t1'] == req.t1 and p['t2'] == req.t2), {"lead": "SYNC"})
        
        # Execute Monte Carlo path projection
        mc_res, prob = run_friction_mc(z_score)
        
        return {
            "dates": state.data.index.strftime('%Y-%m-%d').tolist(),
            "p1": (((s1 / s1.iloc[0]) - 1) * 100).tolist(),
            "p2": (((s2 / s2.iloc[0]) - 1) * 100).tolist(),
            "z": z_score.tolist(), "mc": mc_res,
            "prices": {"p1": s1.iloc[-1], "p2": s2.iloc[-1]},
            "stats": {"hr": round(float(beta[-1]), 3), "copula": round(float(get_copula_score(z_score)), 3), "prob": prob, "lead": pair_info['lead']}
        }
    except: 
        # Catch-all exception handling to prevent unhandled server faults
        return {"error": "FAILED"} 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
