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
import traceback
import time

CACHE_FILE = "Cache.csv"
LOOKBACK_YEARS = 2
MIN_CORR = 0.80
MAX_HURST = 0.55
ROLLING_WINDOW = 60
MIN_TURNOVER = 5000000 
HOST = "0.0.0.0"

warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

SYSTEM_STATUS = "Booting..."
DATA_READY = False
SECTOR_MAP = {}

def get_liquid_universe():
    global SECTOR_MAP
    tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0'}
    
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
    valid_cols = []
    turnover = prices * volumes
    avg_turnover = turnover.median()
    
    for col in prices.columns:
        try:
            if avg_turnover[col] < MIN_TURNOVER: continue
            series = prices[col].dropna()
            if len(series) < 250: continue 
            pct_change = series.pct_change().fillna(0)
            if (pct_change == 0).sum() > (len(series) * 0.20): continue
            if pct_change.abs().median() < 0.0001: continue
            valid_cols.append(col)
        except:
            continue
    return prices[valid_cols]

def download_batched(tickers):
    BATCH_SIZE = 50
    all_data = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        try:
            df = yf.download(batch, period=f"{LOOKBACK_YEARS}y", progress=False, threads=True, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                clean_batch = clean_liquidity(df['Close'], df['Volume'])
                if not clean_batch.empty:
                    all_data.append(clean_batch)
        except Exception as e:
            pass
        time.sleep(0.5)
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, axis=1)

def update_cache():
    global SYSTEM_STATUS, DATA_READY
    SYSTEM_STATUS = "Syncing Data..."
    
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    else:
        all_tickers = get_liquid_universe()
        df = download_batched(all_tickers)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(CACHE_FILE)
        
    DATA_READY = True
    SYSTEM_STATUS = f"Ready ({len(df.columns)} Symbols)"
    return df

def calculate_rolling_hedge_ratio(y, x, window=60):
    cov = y.rolling(window=window).cov(x)
    var = x.rolling(window=window).var()
    return (cov / var).fillna(1.0)

def fit_ou_process(spread):
    try:
        spread_np = spread.values
        x = spread_np[:-1]
        y = spread_np[1:]
        model = sm.OLS(y, sm.add_constant(x)).fit()
        beta = model.params[1]
        theta = -np.log(beta)
        half_life = int(np.log(2) / theta)
        mu = model.params[0] / (1 - beta)
        return max(1, half_life), mu
    except:
        return 99, spread.mean()

def scan_market(data):
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    candidates = corr_matrix.where(mask).stack()
    candidates = candidates[candidates > MIN_CORR]
    results = []
    for t1, t2 in candidates.index.tolist():
        try:
            s1 = data[t1].dropna()
            s2 = data[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 150: 
                print(f"Skipping {t1}-{t2}: Only {len(common)} days of data.")
                continue
            score, p_val, _ = coint(s1[common], s2[common])
            if p_val < 0.05:
                spread = s1[common] / s2[common]
                lags = range(2, 20)
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
        except Exception as e:
            print(f"Failed on {t1} and {t2}: {str(e)}")
            continue
    return sorted(results, key=lambda x: x['hurst'])

app = FastAPI()
market_data = None

class AnalyzeRequest(BaseModel):
    t1: str
    t2: str
    z_threshold: float = 2.0 

@app.on_event("startup")
async def startup():
    threading.Thread(target=lambda: globals().update(market_data=update_cache())).start()

@app.get("/status")
async def get_status():
    return {"status": SYSTEM_STATUS, "ready": DATA_READY}

@app.get("/")
async def get_ui():
    return FileResponse('index.html')

@app.post("/scan")
def scan():
    if not DATA_READY: return {"error": "Processing..."}
    return {"pairs": scan_market(market_data)}
    
@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        p1 = market_data[req.t1].dropna()
        p2 = market_data[req.t2].dropna()
        common = p1.index.intersection(p2.index)
        p1, p2 = p1[common], p2[common]
        
        rolling_hr = calculate_rolling_hedge_ratio(p1, p2, window=ROLLING_WINDOW)
        spread = p1 - (rolling_hr * p2)
        
        rolling_mean = spread.rolling(window=ROLLING_WINDOW).mean()
        rolling_std = spread.rolling(window=ROLLING_WINDOW).std()
        z_score = (spread - rolling_mean) / rolling_std
        
        valid_idx = z_score.dropna().index
        z_score = z_score.loc[valid_idx]
        p1, p2 = p1.loc[valid_idx], p2.loc[valid_idx]
        rolling_hr = rolling_hr.loc[valid_idx]

        thresh = req.z_threshold
        signals = pd.Series(0, index=valid_idx)
        signals[z_score > thresh] = -1
        signals[z_score < -thresh] = 1
        signals[abs(z_score) < 0.5] = 0
        
        pos = signals.replace(0, np.nan).ffill().fillna(0)
        asset_ret = p1.pct_change() - (rolling_hr.shift(1) * p2.pct_change())
        strat_ret = pos.shift(1) * asset_ret
        equity = (1 + strat_ret.fillna(0)).cumprod()
        
        half_life, ou_mean = fit_ou_process(spread)
        
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
