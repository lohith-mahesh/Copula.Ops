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
from datetime import datetime, timedelta
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
LOOKBACK_YEARS = 2 
MIN_CORR = 0.90
MAX_HURST = 0.50
ROLLING_WINDOW = 60
MIN_TURNOVER = 5000000 
BATCH_SIZE = 40 

# Setup
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Global State
SYSTEM_STATUS = "Booting..."
DATA_READY = False
SECTOR_MAP = {}
MARKET_DATA = None
CACHED_SCAN_RESULTS = []

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

def get_incremental_data(existing_df):
    """Downloads only the dates missing from the cache"""
    # Get last date from existing data
    last_date = pd.to_datetime(existing_df.index).max()
    today = datetime.now()
    
    # If data is older than 1 day, fetch the delta
    if today - last_date > timedelta(days=1):
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Fetching incremental data starting from {start_date}...")
        
        tickers = existing_df.columns.tolist()
        new_data_list = []
        
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i : i + BATCH_SIZE]
            try:
                # Fetch only the missing period
                df = yf.download(batch, start=start_date, progress=False, threads=True, auto_adjust=True)
                if not df.empty and 'Close' in df:
                    new_data_list.append(df['Close'])
            except: pass
            time.sleep(0.5)
            
        if new_data_list:
            new_df = pd.concat(new_data_list, axis=1)
            # Combine and remove duplicates
            combined = pd.concat([existing_df, new_df]).sort_index()
            return combined[~combined.index.duplicated(keep='last')]
            
    return existing_df

def check_single_pair(pair_indices, data):
    t1, t2 = pair_indices
    try:
        s1, s2 = data[t1].dropna(), data[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 250: return None
        
        _, p_val, _ = coint(s1[common], s2[common])
        if p_val >= 0.05: return None
        
        spread = s1[common] / s2[common]
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(spread.values[lag:], spread.values[:-lag]))) for lag in lags]
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
        
        if hurst < MAX_HURST:
            return {"t1": t1, "t2": t2, "p_value": round(float(p_val), 5), "hurst": round(float(hurst), 3), "sector": SECTOR_MAP.get(t1, "Market")}
    except: return None

def run_background_pipeline():
    global SYSTEM_STATUS, DATA_READY, MARKET_DATA, CACHED_SCAN_RESULTS
    
    # 1. Sync Data (Smart Logic)
    if os.path.exists(CACHE_FILE):
        SYSTEM_STATUS = "Updating Cache..."
        MARKET_DATA = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        MARKET_DATA = get_incremental_data(MARKET_DATA)
        MARKET_DATA.to_csv(CACHE_FILE)
    else:
        SYSTEM_STATUS = "Full Download..."
        all_tickers = get_liquid_universe()
        # Initial full download logic from previous version
        full_data = []
        for i in range(0, len(all_tickers), BATCH_SIZE):
            batch = all_tickers[i : i + BATCH_SIZE]
            df = yf.download(batch, period=f"{LOOKBACK_YEARS}y", progress=False, threads=True, auto_adjust=True)
            if not df.empty: full_data.append(df['Close'])
            time.sleep(0.5)
        MARKET_DATA = pd.concat(full_data, axis=1)
        MARKET_DATA.to_csv(CACHE_FILE)

    DATA_READY = True
    
    # 2. Parallel Background Scan
    SYSTEM_STATUS = "Scanning Pairs..."
    corr_matrix = MARKET_DATA.corr().abs()
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    candidates = corr_matrix.where(mask).stack()
    candidate_list = candidates[candidates > MIN_CORR].index.tolist()
    
    results = Parallel(n_jobs=-1)(delayed(check_single_pair)(p, MARKET_DATA) for p in candidate_list)
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
    return {"pairs": CACHED_SCAN_RESULTS}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        p1, p2 = MARKET_DATA[req.t1].dropna(), MARKET_DATA[req.t2].dropna()
        common = p1.index.intersection(p2.index)
        p1, p2 = p1[common], p2[common]
        
        cov = p1.rolling(window=ROLLING_WINDOW).cov(p2)
        var = p2.rolling(window=ROLLING_WINDOW).var()
        rolling_hr = (cov / var).fillna(1.0)
        
        spread = p1 - (rolling_hr * p2)
        z_score = (spread - spread.rolling(ROLLING_WINDOW).mean()) / spread.rolling(ROLLING_WINDOW).std()
        
        valid_idx = z_score.dropna().index
        z_score, p1, p2, rolling_hr = z_score.loc[valid_idx], p1.loc[valid_idx], p2.loc[valid_idx], rolling_hr.loc[valid_idx]

        returns = (p1.pct_change() - (rolling_hr.shift(1) * p2.pct_change())) * pd.Series(0, index=valid_idx).mask(z_score > req.z_threshold, -1).mask(z_score < -req.z_threshold, 1).ffill().shift(1)
        equity = (1 + returns.fillna(0)).cumprod()
        
        def clean(s): return s.replace({np.nan: None}).tolist()
        return {
            "dates": valid_idx.strftime('%Y-%m-%d').tolist(),
            "norm_price1": clean((p1 / p1.iloc[0]) - 1),
            "norm_price2": clean((p2 / p2.iloc[0]) - 1),
            "mi": clean((z_score - z_score.min()) / (z_score.max() - z_score.min())),
            "equity": clean(equity),
            "stats": {"hedge_ratio": round(float(rolling_hr.iloc[-1]), 4), "current_z": round(float(z_score.iloc[-1]), 2), "sharpe": round(float((returns.mean()/returns.std())*np.sqrt(252)), 2) if returns.std() != 0 else 0}
        }
    except: return {"error": "Failed"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
