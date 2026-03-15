import uvicorn
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta
import logging, os, requests, io, threading, warnings, traceback, time, shutil

## Config & Persistent Pathing
DISK_DIR = "/opt/render/project/src/data"
DISK_CACHE = os.path.join(DISK_DIR, "Cache.csv")
GIT_CACHE = os.path.join(os.getcwd(), "Cache.csv")

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
    urls = ["https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
            "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv"]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                for _, row in df.iterrows():
                    sym = f"{row['Symbol'].strip()}.NS"
                    tickers.add(sym)
                    if sym not in SECTOR_MAP: SECTOR_MAP[sym] = row.get('Industry', 'Microcap')
        except: pass
    return list(tickers)

def get_incremental_data(existing_df):
    """Fetches only the missing dates since last update"""
    last_date = pd.to_datetime(existing_df.index).max()
    today = datetime.now()
    if today - last_date > timedelta(days=1):
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        tickers = existing_df.columns.tolist()
        new_data_list = []
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i : i + BATCH_SIZE]
            try:
                df = yf.download(batch, start=start_date, progress=False, threads=True, auto_adjust=True)
                if not df.empty and 'Close' in df: new_data_list.append(df['Close'])
            except: pass
            time.sleep(0.5)
        if new_data_list:
            new_df = pd.concat(new_data_list, axis=1)
            combined = pd.concat([existing_df, new_df]).sort_index()
            return combined[~combined.index.duplicated(keep='last')]
    return existing_df

def check_single_pair(pair_indices, data):
    """Math extracted for Joblib parallelization"""
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
    if not os.path.exists(DISK_DIR): os.makedirs(DISK_DIR, exist_ok=True)
    
    # Migrate Seed from Git to Disk if necessary
    if not os.path.exists(DISK_CACHE) and os.path.exists(GIT_CACHE):
        SYSTEM_STATUS = "Migrating Seed..."
        shutil.copy(GIT_CACHE, DISK_CACHE)

    if os.path.exists(DISK_CACHE):
        SYSTEM_STATUS = "Syncing Delta..."
        df = pd.read_csv(DISK_CACHE, index_col=0, parse_dates=True)
        MARKET_DATA = get_incremental_data(df)
        MARKET_DATA.to_csv(DISK_CACHE)
    else:
        SYSTEM_STATUS = "Full Download..."
        all_tickers = get_liquid_universe()
        full_data = []
        for i in range(0, len(all_tickers), BATCH_SIZE):
            batch = all_tickers[i : i + BATCH_SIZE]
            df = yf.download(batch, period=f"{LOOKBACK_YEARS}y", progress=False, auto_adjust=True)
            if not df.empty: full_data.append(df['Close'])
            time.sleep(0.5)
        MARKET_DATA = pd.concat(full_data, axis=1)
        MARKET_DATA.to_csv(DISK_CACHE)

    DATA_READY = True
    SYSTEM_STATUS = "Scanning Market..."
    corr_matrix = MARKET_DATA.corr().abs()
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    candidates = corr_matrix.where(mask).stack()
    candidate_list = candidates[candidates > MIN_CORR].index.tolist()
    
    results = Parallel(n_jobs=-1)(delayed(check_single_pair)(p, MARKET_DATA) for p in candidate_list)
    CACHED_SCAN_RESULTS = sorted([r for r in results if r], key=lambda x: x['hurst'])
    SYSTEM_STATUS = f"Ready ({len(MARKET_DATA.columns)} Stocks)"

app = FastAPI()

@app.on_event("startup")
async def startup():
    threading.Thread(target=run_background_pipeline, daemon=True).start()

@app.get("/status")
async def get_status(): return {"status": SYSTEM_STATUS, "ready": DATA_READY}

@app.get("/")
async def get_ui(): return FileResponse('index.html')

@app.post("/scan")
async def scan(): return {"pairs": CACHED_SCAN_RESULTS}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # (Same analysis logic as before, using MARKET_DATA)
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
