import uvicorn
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from joblib import Parallel, delayed
import logging
import os
import requests
import io
import threading
import warnings
import traceback
import time

## Config & Persistent Pathing
DISK_PATH = "/opt/render/project/src/data"
BASE_DIR = DISK_PATH if os.path.exists(DISK_PATH) else os.getcwd()
CACHE_FILE = os.path.join(BASE_DIR, "Cache.csv")

# Constants
LOOKBACK_YEARS = 2  # Kept at 2 as requested
MIN_CORR = 0.90
MAX_HURST = 0.50
ROLLING_WINDOW = 60
MIN_TURNOVER = 5000000 
BATCH_SIZE = 30 # Balanced for RAM and speed

# Setup
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Global State
SYSTEM_STATUS = "Booting..."
DATA_READY = False
SECTOR_MAP = {}
MARKET_DATA = None
CACHED_SCAN_RESULTS = [] # Stores pre-calculated scan

# --- Core Logic ---

def get_liquid_universe():
    global SECTOR_MAP
    tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    def fetch_indices(url):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                for _, row in df.iterrows():
                    sym = f"{row['Symbol'].strip()}.NS"
                    tickers.add(sym)
                    if sym not in SECTOR_MAP:
                        SECTOR_MAP[sym] = row.get('Industry', 'Microcap')
        except: pass

    fetch_indices("https://archives.nseindia.com/content/indices/ind_nifty500list.csv")
    fetch_indices("https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv")
    return list(tickers)

def download_batched(tickers):
    all_data = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        try:
            df = yf.download(batch, period=f"{LOOKBACK_YEARS}y", progress=False, threads=True, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                # Basic turnover filter
                prices = df['Close']
                volumes = df['Volume']
                turnover = prices * volumes
                avg_turnover = turnover.median()
                valid = [col for col in prices.columns if avg_turnover[col] >= MIN_TURNOVER and len(prices[col].dropna()) > 250]
                if valid: all_data.append(prices[valid])
        except: pass
        time.sleep(1.0) # Rate limit protection
            
    return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()

def check_single_pair(pair_indices, data):
    """Heavy math extracted for parallel execution"""
    t1, t2 = pair_indices
    try:
        s1, s2 = data[t1].dropna(), data[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 200: return None
        
        # Cointegration
        _, p_val, _ = coint(s1[common], s2[common])
        if p_val >= 0.05: return None
        
        # Hurst
        spread = s1[common] / s2[common]
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(spread.values[lag:], spread.values[:-lag]))) for lag in lags]
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
        
        if hurst < MAX_HURST:
            return {
                "t1": t1, "t2": t2, 
                "p_value": round(float(p_val), 5), 
                "hurst": round(float(hurst), 3), 
                "sector": SECTOR_MAP.get(t1, "Market")
            }
    except: return None
    return None

def run_background_pipeline():
    """Sequential engine logic run in a separate thread"""
    global SYSTEM_STATUS, DATA_READY, MARKET_DATA, CACHED_SCAN_RESULTS
    
    # 1. Sync Data
    if os.path.exists(CACHE_FILE):
        SYSTEM_STATUS = "Loading Cache..."
        MARKET_DATA = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    else:
        SYSTEM_STATUS = "Downloading Market..."
        all_tickers = get_liquid_universe()
        MARKET_DATA = download_batched(all_tickers)
        MARKET_DATA = MARKET_DATA.loc[:, ~MARKET_DATA.columns.duplicated()]
        MARKET_DATA.to_csv(CACHE_FILE)

    DATA_READY = True
    
    # 2. Pre-Calculate Scan (The Parallel Boost)
    SYSTEM_STATUS = "Scanning Pairs..."
    corr_matrix = MARKET_DATA.corr().abs()
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    candidates = corr_matrix.where(mask).stack()
    candidate_list = candidates[candidates > MIN_CORR].index.tolist()
    
    # Use all CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(check_single_pair)(pair, MARKET_DATA) for pair in candidate_list
    )
    
    CACHED_SCAN_RESULTS = sorted([r for r in results if r], key=lambda x: x['hurst'])
    SYSTEM_STATUS = f"Ready ({len(MARKET_DATA.columns)} Stocks)"

# --- API ---

app = FastAPI()

class AnalyzeRequest(BaseModel):
    t1: str
    t2: str
    z_threshold: float = 2.0 

@app.on_event("startup")
async def startup():
    threading.Thread(target=run_background_pipeline, daemon=True).start()

@app.get("/status")
async def get_status():
    return {"status": SYSTEM_STATUS, "ready": DATA_READY}

@app.get("/")
async def get_ui():
    return FileResponse('index.html')

@app.post("/scan")
async def scan():
    # Returns the pre-calculated list instantly
    return {"pairs": CACHED_SCAN_RESULTS}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        p1, p2 = MARKET_DATA[req.t1].dropna(), MARKET_DATA[req.t2].dropna()
        common = p1.index.intersection(p2.index)
        p1, p2 = p1[common], p2[common]
        
        # Rolling Stats
        cov = p1.rolling(window=ROLLING_WINDOW).cov(p2)
        var = p2.rolling(window=ROLLING_WINDOW).var()
        rolling_hr = (cov / var).fillna(1.0)
        
        spread = p1 - (rolling_hr * p2)
        z_score = (spread - spread.rolling(ROLLING_WINDOW).mean()) / spread.rolling(ROLLING_WINDOW).std()
        
        valid_idx = z_score.dropna().index
        z_score, p1, p2, rolling_hr = z_score.loc[valid_idx], p1.loc[valid_idx], p2.loc[valid_idx], rolling_hr.loc[valid_idx]

        # Signals & Backtest
        pos = pd.Series(0, index=valid_idx)
        pos[z_score > req.z_threshold] = -1
        pos[z_score < -req.z_threshold] = 1
        pos = pos.replace(0, np.nan).ffill().fillna(0)
        
        returns = (p1.pct_change() - (rolling_hr.shift(1) * p2.pct_change())) * pos.shift(1)
        equity = (1 + returns.fillna(0)).cumprod()
        
        def clean(s): return s.replace({np.nan: None}).tolist()
        return {
            "dates": valid_idx.strftime('%Y-%m-%d').tolist(),
            "norm_price1": clean((p1 / p1.iloc[0]) - 1),
            "norm_price2": clean((p2 / p2.iloc[0]) - 1),
            "mi": clean((z_score - z_score.min()) / (z_score.max() - z_score.min())),
            "equity": clean(equity),
            "stats": {
                "hedge_ratio": round(float(rolling_hr.iloc[-1]), 4),
                "current_z": round(float(z_score.iloc[-1]), 2),
                "sharpe": round(float((returns.mean() / returns.std()) * np.sqrt(252)), 2) if returns.std() != 0 else 0
            }
        }
    except: return {"error": "Failed"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
