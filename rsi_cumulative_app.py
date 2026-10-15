import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import re

st.set_page_config(page_title="Cumulative RSI(2) Edge Platform", layout="wide")

st.title("Cumulative RSI(2) Quantitative Edge Platform")
st.caption(
    "Parametric Multi-Day Smoothed RSI(2) Engine | Real-Time Ticker Watchlist Analyzer | "
    "Dynamic Snapback Exits (SMA5, RSI70) | 200-Day Trend Filtering | MAE Drawdown Tracking."
)

# ==========================================
# 1. CORE QUANTITATIVE FUNCTIONS
# ==========================================

def calculate_wilders_rsi(prices, period=2):
    """Calculates standard Wilder's Smoothed RSI(2)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)

def prepare_stock_data(df):
    """Computes indicators, preceding momentum, fixed/dynamic exits, and MAE drawdowns."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    close_s = df[price_col]
    low_s = df['Low']
    
    # 1. Core Indicators & Lookback Horizons
    df['RSI2'] = calculate_wilders_rsi(close_s, period=2)
    df['RSI2_2d_avg'] = df['RSI2'].rolling(window=2).mean()
    df['RSI2_3d_avg'] = df['RSI2'].rolling(window=3).mean()
    df['RSI2_5d_avg'] = df['RSI2'].rolling(window=5).mean()
    df['RSI2_10d_avg'] = df['RSI2'].rolling(window=10).mean()
    
    # 2. Trend Overlays
    df['SMA_5'] = close_s.rolling(window=5).mean()
    df['SMA_200'] = close_s.rolling(window=200).mean()
    df['Above_200SMA'] = close_s > df['SMA_200']
    
    # 3. Preceding Returns (Trailing Momentum)
    df['Prior_5d'] = (close_s - close_s.shift(5)) / close_s.shift(5) * 100
    df['Prior_10d'] = (close_s - close_s.shift(10)) / close_s.shift(10) * 100
    df['Prior_15d'] = (close_s - close_s.shift(15)) / close_s.shift(15) * 100
    
    # 4. Fixed Forward Returns
    df['Fwd_5d'] = (close_s.shift(-5) - close_s) / close_s * 100
    df['Fwd_10d'] = (close_s.shift(-10) - close_s) / close_s * 100
    df['Fwd_15d'] = (close_s.shift(-15) - close_s) / close_s * 100
    
    # 5. Maximum Adverse Excursion (MAE)
    min_low_5d = low_s.iloc[::-1].rolling(5).min().shift(1).iloc[::-1]
    min_low_10d = low_s.iloc[::-1].rolling(10).min().shift(1).iloc[::-1]
    df['MAE_5d'] = (min_low_5d - close_s) / close_s * 100
    df['MAE_10d'] = (min_low_10d - close_s) / close_s * 100
    
    # 6. Dynamic Snapback Exits
    n = len(df)
    c_vals = close_s.values
    sma5_vals = df['SMA_5'].values
    rsi_vals = df['RSI2'].values
    
    fwd_sma5_ret = np.full(n, np.nan)
    fwd_sma5_days = np.full(n, np.nan)
    fwd_rsi70_ret = np.full(n, np.nan)
    fwd_rsi70_days = np.full(n, np.nan)
    
    for i in range(n - 15):
        c0 = c_vals[i]
        
        # Exit on Close > 5-Day SMA
        for k in range(1, 16):
            if c_vals[i + k] > sma5_vals[i + k]:
                fwd_sma5_ret[i] = (c_vals[i + k] - c0) / c0 * 100
                fwd_sma5_days[i] = k
                break
        else:
            fwd_sma5_ret[i] = (c_vals[i + 15] - c0) / c0 * 100
            fwd_sma5_days[i] = 15
            
        # Exit on RSI(2) > 70
        for k in range(1, 16):
            if rsi_vals[i + k] > 70:
                fwd_rsi70_ret[i] = (c_vals[i + k] - c0) / c0 * 100
                fwd_rsi70_days[i] = k
                break
        else:
            fwd_rsi70_ret[i] = (c_vals[i + 15] - c0) / c0 * 100
            fwd_rsi70_days[i] = 15
            
    df['Fwd_SMA5_Ret'] = fwd_sma5_ret
    df['Fwd_SMA5_Days'] = fwd_sma5_days
    df['Fwd_RSI70_Ret'] = fwd_rsi70_ret
    df['Fwd_RSI70_Days'] = fwd_rsi70_days
    
    # 7. Discrete 10-Point Bucketing
    bins = [-0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.001]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    
    df['Bucket_2d'] = pd.cut(df['RSI2_2d_avg'], bins=bins, labels=labels)
    df['Bucket_3d'] = pd.cut(df['RSI2_3d_avg'], bins=bins, labels=labels)
    df['Bucket_5d'] = pd.cut(df['RSI2_5d_avg'], bins=bins, labels=labels)
    df['Bucket_10d'] = pd.cut(df['RSI2_10d_avg'], bins=bins, labels=labels)
    
    return df

def aggregate_bucket_performance(df, bucket_col, fresh_entry_only=False, require_uptrend=False):
    """Aggregates win rate, return distributions, MAE, and dynamic snapback exits by bucket."""
    data = df.dropna(subset=[bucket_col, 'Fwd_5d', 'Fwd_10d', 'Fwd_15d']).copy()
    
    if require_uptrend:
        data = data[data['Above_200SMA'] == True]
        
    if fresh_entry_only:
        data = data[data[bucket_col] != data[bucket_col].shift(1)]
        
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    records = []
    
    for b in labels:
        sub = data[data[bucket_col] == b]
        n = len(sub)
        if n == 0:
            records.append({
                "RSI Bucket": b, "Signals": 0,
                "5D Win Rate (%)": 0.0, "5D Avg Ret (%)": 0.0, "5D Med Drawdown (%)": 0.0,
                "10D Win Rate (%)": 0.0, "10D Avg Ret (%)": 0.0, "10D Med Drawdown (%)": 0.0,
                "SMA5 Exit Win Rate (%)": 0.0, "SMA5 Exit Avg Ret (%)": 0.0, "SMA5 Avg Hold (Days)": 0.0,
                "RSI70 Exit Win Rate (%)": 0.0, "RSI70 Exit Avg Ret (%)": 0.0, "RSI70 Avg Hold (Days)": 0.0
            })
            continue
            
        def stats_fixed(ret_series, mae_series):
            wr = (ret_series > 0).mean() * 100
            avg_r = ret_series.mean()
            med_mae = mae_series.median()
            return round(wr, 1), round(avg_r, 2), round(med_mae, 2)
            
        def stats_dynamic(ret_series, days_series):
            wr = (ret_series > 0).mean() * 100
            avg_r = ret_series.mean()
            avg_d = days_series.mean()
            return round(wr, 1), round(avg_r, 2), round(avg_d, 1)
            
        wr5, avg5, mae5 = stats_fixed(sub['Fwd_5d'], sub['MAE_5d'])
        wr10, avg10, mae10 = stats_fixed(sub['Fwd_10d'], sub['MAE_10d'])
        wr_sma, avg_sma, days_sma = stats_dynamic(sub['Fwd_SMA5_Ret'], sub['Fwd_SMA5_Days'])
        wr_rsi, avg_rsi, days_rsi = stats_dynamic(sub['Fwd_RSI70_Ret'], sub['Fwd_RSI70_Days'])
        
        records.append({
            "RSI Bucket": b,
            "Signals": int(n),
            "5D Win Rate (%)": wr5,
            "5D Avg Ret (%)": avg5,
            "5D Med Drawdown (%)": mae5,
            "10D Win Rate (%)": wr10,
            "10D Avg Ret (%)": avg10,
            "10D Med Drawdown (%)": mae10,
            "SMA5 Exit Win Rate (%)": wr_sma,
            "SMA5 Exit Avg Ret (%)": avg_sma,
            "SMA5 Avg Hold (Days)": days_sma,
            "RSI70 Exit Win Rate (%)": wr_rsi,
            "RSI70 Exit Avg Ret (%)": avg_rsi,
            "RSI70 Avg Hold (Days)": days_rsi
        })
        
    return pd.DataFrame(records).set_index("RSI Bucket")

# ==========================================
# 2. UNIVERSE FETCHERS & PARSERS
# ==========================================

DJIA_STATIC_FALLBACK = [
    "AAPL", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
]

@st.cache_data(show_spinner=False, ttl=86400)
def get_universe_tickers(universe_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36"}
    
    def extract_wiki(url):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                for df in pd.read_html(resp.text):
                    for col in ['Symbol', 'Ticker']:
                        if col in df.columns:
                            return [s.replace('.', '-') for s in df[col].dropna().astype(str).tolist()]
        except Exception:
            pass
        return []

    if universe_name == "S&P 500":
        res = extract_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if not res:
            try:
                res = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")['Symbol'].tolist()
            except Exception: pass
        return res
    elif universe_name == "Nasdaq 100":
        res = extract_wiki("https://en.wikipedia.org/wiki/Nasdaq-100")
        return res if res else ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "AMD"]
    elif universe_name == "Dow Jones 30":
        res = extract_wiki("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        return res if res else DJIA_STATIC_FALLBACK
    elif universe_name == "Major ETFs":
        return ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "SMH", "TLT", "GLD"]
    return []

@st.cache_data(show_spinner=False)
def fetch_daily_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def extract_symbol_daily(bulk_data, sym, total_tickers):
    """Robustly extracts single-symbol daily OHLCV from bulk yfinance downloads."""
    try:
        if total_tickers == 1:
            df = bulk_data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            return df.dropna(how='all')
        if not isinstance(bulk_data.columns, pd.MultiIndex):
            return pd.DataFrame()
        if sym in bulk_data.columns.levels[0]:
            return bulk_data[sym].copy().dropna(how='all')
        elif sym in bulk_data.columns.levels[1]:
            return bulk_data.xs(sym, axis=1, level=1).copy().dropna(how='all')
    except Exception:
        pass
    return pd.DataFrame()

# ==========================================
# 3. INTERFACE LAYOUT & TABS
# ==========================================

tab_study, tab_watchlist, tab_screener = st.tabs([
    "Single Ticker Study & Current Status",
    "Custom Watchlist Analyzer",
    "Live Market Screener"
])

# ----------------------------------------------------
# TAB 1: SINGLE TICKER STUDY & CURRENT STATUS
# ----------------------------------------------------
with tab_study:
    st.header("Single Ticker Expectancy & Live Status")
    st.write(
        "Evaluate full historical bucket return distributions while inspecting where the asset currently sits today."
    )
    
    col_t1, col_t2, col_t3, col_t4 = st.columns([1.1, 1.2, 1.2, 0.8])
    with col_t1:
        study_ticker = st.text_input("Ticker Symbol", value="SPY", key="study_sym").upper().strip()
    with col_t2:
        study_indicator = st.selectbox(
            "Smoothed RSI(2) Lookback", 
            ["2-Day Average RSI(2)", "3-Day Average RSI(2)", "5-Day Average RSI(2)", "10-Day Average RSI(2)"],
            index=2
        )
    with col_t3:
        signal_mode = st.selectbox("Sample Mode", ["All Days in Bucket", "Fresh Entry Day Only (No Overlap)"])
    with col_t4:
        years_back = st.number_input("Lookback (Years)", min_value=1, max_value=30, value=20)
        
    require_uptrend_study = st.checkbox(
        "Uptrend Filter: Only include setups trading ABOVE 200-Day SMA",
        value=True,
        help="Filters out secular bear markets and value traps, isolating high-probability pullback dips."
    )
        
    if st.button("Run Forward Edge & Dynamic Exit Study", type="primary", use_container_width=True):
        start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        with st.spinner(f"Analyzing daily tick data for {study_ticker}..."):
            raw_df = fetch_daily_data(study_ticker, start_date, end_date)
            
        if raw_df is None or len
