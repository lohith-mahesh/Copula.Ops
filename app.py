import uvicorn
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import logging
import os
import requests
import io
import threading
import warnings
import math
import webbrowser
import traceback
import time

## Config
CACHE_FILE = "Cache.csv"
LOOKBACK_YEARS = 2
MIN_CORR = 0.90
MAX_HURST = 0.50
ROLLING_WINDOW = 60
MIN_TURNOVER = 5000000 
HOST = "127.0.0.1"
PORT = 8000
BATCH_SIZE = 50

# Setup
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Global State
SYSTEM_STATUS = "Booting..."
DATA_READY = False
SECTOR_MAP = {}

# Core Logic
def get_liquid_universe():
    """Fetches Nifty 500 and Microcap 250 tickers from NSE."""
    global SECTOR_MAP
    tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Helper to fetch and parse
    def fetch_indices(url, sector_key):
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                for _, row in df.iterrows():
                    sym = f"{row['Symbol'].strip()}.NS"
                    tickers.add(sym)
                    if sym not in SECTOR_MAP:
                        SECTOR_MAP[sym] = row.get('Industry', 'Microcap')
        except Exception as e:
            print(f"Failed to fetch index: {url} | {e}")

    fetch_indices("https://archives.nseindia.com/content/indices/ind_nifty500list.csv", 'Industry')
    fetch_indices("https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv", 'Microcap')
    
    return list(tickers)

def clean_liquidity(prices, volumes):
    """Filters out stocks with low turnover or dead price action."""
    valid_cols = []
    # turnover approx = price * volume
    turnover = prices * volumes
    avg_turnover = turnover.median()
    
    for col in prices.columns:
        try:
            # 1. Turnover Check
            if avg_turnover[col] < MIN_TURNOVER: continue
            
            # 2. Data Length Check
            series = prices[col].dropna()
            if len(series) < 250: continue 
            
            # 3. Dead Stock Check (Zeros or no movement)
            pct_change = series.pct_change().fillna(0)
            if (pct_change == 0).sum() > (len(series) * 0.20): continue # Too many flat days
            if pct_change.abs().median() < 0.0001: continue # Too stable (likely debt/preference share)
            
            valid_cols.append(col)
        except:
            continue
            
    return prices[valid_cols]

def download_batched(tickers):
    all_data = []
    print(f"Starting download for {len(tickers)} stocks...")
    
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        try:
            # Threaded download is faster
            df = yf.download(batch, period=f"{LOOKBACK_YEARS}y", progress=False, threads=True, auto_adjust=True)
            
            # yfinance returns MultiIndex if multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                clean_batch = clean_liquidity(df['Close'], df['Volume'])
                if not clean_batch.empty:
                    all_data.append(clean_batch)
        except Exception as e:
            print(f"Batch {i} error: {e}")
        
        # Polite delay to avoid rate limits
        time.sleep(0.5)
            
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, axis=1)

def update_cache():
    global SYSTEM_STATUS, DATA_READY
    SYSTEM_STATUS = "Syncing Data..."
    
    if os.path.exists(CACHE_FILE):
        print("Loading from cache...")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    else:
        print("Cache not found, downloading fresh data...")
        all_tickers = get_liquid_universe()
        df = download_batched(all_tickers)
        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(CACHE_FILE)
        
    DATA_READY = True
    SYSTEM_STATUS = f"Ready ({len(df.columns)} Stocks)"
    return df

def calculate_rolling_hedge_ratio(y, x, window=60):
    """Calculates dynamic beta (hedge ratio) to avoid look-ahead bias."""
    cov = y.rolling(window=window).cov(x)
    var = x.rolling(window=window).var()
    return (cov / var).fillna(1.0)

def fit_ou_process(spread):
    """Fits Ornstein-Uhlenbeck process to estimate Mean Reversion speed."""
    try:
        spread_np = spread.values
        x = spread_np[:-1]
        y = spread_np[1:]
        
        # Regress S(t) against S(t-1)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        beta = model.params[1]
        
        # Calculate parameters
        theta = -np.log(beta)
        half_life = int(np.log(2) / theta)
        mu = model.params[0] / (1 - beta) # Equilibrium level
        
        return max(1, half_life), mu
    except:
        return 99, spread.mean()

def scan_market(data):
    # 1.Vectorized Correlation
    corr_matrix = data.corr().abs()
    
    # Filter upper triangle only to avoid duplicates
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    candidates = corr_matrix.where(mask).stack()
    candidates = candidates[candidates > MIN_CORR]
    
    results = []
    
    # 2.Detailed Checks (Cointegration & Hurst)
    for t1, t2 in candidates.index.tolist():
        try:
            s1 = data[t1].dropna()
            s2 = data[t2].dropna()
            
            # Sync dates
            common = s1.index.intersection(s2.index)
            if len(common) < 150: continue
            
            # Cointegration Test (Engle-Granger)
            score, p_val, _ = coint(s1[common], s2[common])
            
            if p_val < 0.05:
                # Hurst Exponent (Mean Reversion Speed)
                spread = s1[common] / s2[common]
                lags = range(2, 20)
                # Calculate variance of differences
                tau = [np.sqrt(np.std(np.subtract(spread.values[lag:], spread.values[:-lag]))) for lag in lags]
                hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
                
                if hurst < MAX_HURST:
                    results.append({
                        "t1": t1, "t2": t2, 
                        "correlation": round(candidates[(t1, t2)], 3), 
                        "p_value": round(float(p_val), 5), 
                        "hurst": round(float(hurst), 3), 
                        "sector": SECTOR_MAP.get(t1, "Market")
                    })
        except:
            continue
            
    return sorted(results, key=lambda x: x['hurst'])

# API

app = FastAPI()
market_data = None

class AnalyzeRequest(BaseModel):
    t1: str
    t2: str
    z_threshold: float = 2.0 

@app.on_event("startup")
async def startup():
    # Auto-open browser
    webbrowser.open(f"http://{HOST}:{PORT}")
    # Load data in background
    threading.Thread(target=lambda: globals().update(market_data=update_cache())).start()

@app.get("/status")
async def get_status():
    return {"status": SYSTEM_STATUS, "ready": DATA_READY}

@app.get("/")
async def get_ui():
    return FileResponse('index.html')

@app.post("/scan")
async def scan():
    if not DATA_READY: return {"error": "Processing..."}
    return {"pairs": scan_market(market_data)}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        # Data Prep
        p1 = market_data[req.t1].dropna()
        p2 = market_data[req.t2].dropna()
        common = p1.index.intersection(p2.index)
        p1, p2 = p1[common], p2[common]
        
        # 1.Rolling Stats
        rolling_hr = calculate_rolling_hedge_ratio(p1, p2, window=ROLLING_WINDOW)
        spread = p1 - (rolling_hr * p2)
        
        rolling_mean = spread.rolling(window=ROLLING_WINDOW).mean()
        rolling_std = spread.rolling(window=ROLLING_WINDOW).std()
        z_score = (spread - rolling_mean) / rolling_std
        
        # 2.Clean NaN (Lookback period)
        valid_idx = z_score.dropna().index
        z_score = z_score.loc[valid_idx]
        p1, p2 = p1.loc[valid_idx], p2.loc[valid_idx]
        rolling_hr = rolling_hr.loc[valid_idx]

        # 3.Signals
        thresh = req.z_threshold
        signals = pd.Series(0, index=valid_idx)
        signals[z_score > thresh] = -1 # Short Spread
        signals[z_score < -thresh] = 1 # Long Spread
        signals[abs(z_score) < 0.5] = 0 # Exit
        
        # 4.Backtest (Shift 1 to avoid lookahead)
        pos = signals.replace(0, np.nan).ffill().fillna(0)
        asset_ret = p1.pct_change() - (rolling_hr.shift(1) * p2.pct_change())
        strat_ret = pos.shift(1) * asset_ret
        equity = (1 + strat_ret.fillna(0)).cumprod()
        
        # 5.Metadata
        half_life, ou_mean = fit_ou_process(spread)
        
        # JSON Prep (Handle NaNs for JS)
        def clean_series(s): return s.replace({np.nan: None}).tolist()
        
        return {
            "dates": valid_idx.strftime('%Y-%m-%d').tolist(),
            "norm_price1": clean_series((p1 / p1.iloc[0]) - 1),
            "norm_price2": clean_series((p2 / p2.iloc[0]) - 1),
            "mi": clean_series((z_score - z_score.min()) / (z_score.max() - z_score.min())),
            "equity": clean_series(equity),
            "stats": {
                "hedge_ratio": round(float(rolling_hr.iloc[-1]), 4), 
                "half_life": int(half_life), 
                "ou_target": round(float(ou_mean), 4),
                "current_z": round(float(z_score.iloc[-1]), 2), 
                "sharpe": round(float((strat_ret.mean() / strat_ret.std()) * np.sqrt(252)), 2) if strat_ret.std() != 0 else 0
            }
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": "Analysis Failed"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)