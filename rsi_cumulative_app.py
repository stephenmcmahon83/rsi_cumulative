import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import re

st.set_page_config(page_title="Cumulative RSI(2) Edge Platform", layout="wide")

st.title("⚡ Cumulative RSI(2) Forward Edge & Live Screener")
st.caption(
    "Quantitative study and live screening of 5-Day and 10-Day smoothed RSI(2) regimes across 10-point buckets "
    "evaluating 5, 10, and 15-day forward return expectations."
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
    """Computes RSI(2), 5D/10D rolling averages, preceding returns, and forward returns."""
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    # 1. Indicator Calculations
    df['RSI2'] = calculate_wilders_rsi(df[price_col], period=2)
    df['RSI2_5d_avg'] = df['RSI2'].rolling(window=5).mean()
    df['RSI2_10d_avg'] = df['RSI2'].rolling(window=10).mean()
    
    # 2. Preceding Returns
    df['Prior_5d'] = (df[price_col] - df[price_col].shift(5)) / df[price_col].shift(5) * 100
    df['Prior_10d'] = (df[price_col] - df[price_col].shift(10)) / df[price_col].shift(10) * 100
    df['Prior_15d'] = (df[price_col] - df[price_col].shift(15)) / df[price_col].shift(15) * 100
    
    # 3. Forward Returns
    df['Fwd_5d'] = (df[price_col].shift(-5) - df[price_col]) / df[price_col] * 100
    df['Fwd_10d'] = (df[price_col].shift(-10) - df[price_col]) / df[price_col] * 100
    df['Fwd_15d'] = (df[price_col].shift(-15) - df[price_col]) / df[price_col] * 100
    
    # 4. Discrete 10-Point Bucketing (0-10, 11-20, ..., 91-100)
    bins = [-0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.001]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    
    df['Bucket_5d'] = pd.cut(df['RSI2_5d_avg'], bins=bins, labels=labels)
    df['Bucket_10d'] = pd.cut(df['RSI2_10d_avg'], bins=bins, labels=labels)
    
    return df

def aggregate_bucket_performance(df, bucket_col, fresh_entry_only=False):
    """Aggregates win rate, mean/median return, and profit factor for 5D, 10D, and 15D exits."""
    data = df.dropna(subset=[bucket_col, 'Fwd_5d', 'Fwd_10d', 'Fwd_15d']).copy()
    
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
                "5D Win Rate (%)": 0.0, "5D Avg Ret (%)": 0.0, "5D Med Ret (%)": 0.0, "5D Profit Factor": 0.0,
                "10D Win Rate (%)": 0.0, "10D Avg Ret (%)": 0.0, "10D Med Ret (%)": 0.0, "10D Profit Factor": 0.0,
                "15D Win Rate (%)": 0.0, "15D Avg Ret (%)": 0.0, "15D Med Ret (%)": 0.0, "15D Profit Factor": 0.0
            })
            continue
            
        def calc_stats(series):
            wr = (series > 0).mean() * 100
            avg_r = series.mean()
            med_r = series.median()
            gains = series[series > 0].sum()
            losses = abs(series[series < 0].sum())
            pf = round(gains / losses, 2) if losses > 0 else np.nan
            return wr, avg_r, med_r, pf
            
        wr5, avg5, med5, pf5 = calc_stats(sub['Fwd_5d'])
        wr10, avg10, med10, pf10 = calc_stats(sub['Fwd_10d'])
        wr15, avg15, med15, pf15 = calc_stats(sub['Fwd_15d'])
        
        records.append({
            "RSI Bucket": b,
            "Signals": int(n),
            "5D Win Rate (%)": wr5,
            "5D Avg Ret (%)": avg5,
            "5D Med Ret (%)": med5,
            "5D Profit Factor": pf5,
            "10D Win Rate (%)": wr10,
            "10D Avg Ret (%)": avg10,
            "10D Med Ret (%)": med10,
            "10D Profit Factor": pf10,
            "15D Win Rate (%)": wr15,
            "15D Avg Ret (%)": avg15,
            "15D Med Ret (%)": med15,
            "15D Profit Factor": pf15
        })
        
    return pd.DataFrame(records).set_index("RSI Bucket")

# ==========================================
# 2. UNIVERSE FETCHERS (CLOUD RESILIENT)
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

def style_apply_elementwise(styler, func, subset=None):
    if hasattr(styler, 'map'):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)

# ==========================================
# 3. INTERFACE LAYOUT & TABS
# ==========================================

tab_study, tab_screener = st.tabs(["📊 Forward Return Matrix (Study)", "📡 Live RSI(2) Screener"])

# ----------------------------------------------------
# TAB 1: FORWARD RETURN DISTRIBUTION ANALYZER
# ----------------------------------------------------
with tab_study:
    st.header("Historical Forward Edge by RSI(2) Bucket")
    st.write("Examine how an asset performs 5, 10, and 15 days forward based on its 5-day or 10-day smoothed RSI(2) entry levels.")
    
    col_t1, col_t2, col_t3, col_t4 = st.columns([1.2, 1.2, 1.2, 1])
    with col_t1:
        study_ticker = st.text_input("Ticker Symbol", value="SPY", key="study_sym").upper().strip()
    with col_t2:
        study_indicator = st.selectbox("Indicator Model", ["5-Day Average RSI(2)", "10-Day Average RSI(2)"])
    with col_t3:
        signal_mode = st.selectbox("Sample Mode", ["All Days in Bucket", "Fresh Entry Day Only (No Overlap)"])
    with col_t4:
        years_back = st.number_input("Lookback (Years)", min_value=1, max_value=30, value=20)
        
    if st.button("Run Forward Edge Study", type="primary", use_container_width=True):
        start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        with st.spinner(f"Loading {years_back} years of daily data for {study_ticker}..."):
            raw_df = fetch_daily_data(study_ticker, start_date, end_date)
            
        if raw_df is None or len(raw_df) < 50:
            st.error(f"Could not load sufficient daily history for {study_ticker}.")
        else:
            processed_df = prepare_stock_data(raw_df)
            active_bucket_col = 'Bucket_5d' if "5-Day" in study_indicator else 'Bucket_10d'
            fresh_only = (signal_mode == "Fresh Entry Day Only (No Overlap)")
            
            perf_table = aggregate_bucket_performance(processed_df, active_bucket_col, fresh_entry_only=fresh_only)
            
            st.subheader(f"📋 Forward Expectancy Matrix: {study_ticker} ({study_indicator})")
            
            format_dict = {
                "Signals": "{:d}",
                "5D Win Rate (%)": "{:.1f}%",
                "5D Avg Ret (%)": "{:+.2f}%",
                "5D Med Ret (%)": "{:+.2f}%",
                "5D Profit Factor": "{:.2f}",
                "10D Win Rate (%)": "{:.1f}%",
                "10D Avg Ret (%)": "{:+.2f}%",
                "10D Med Ret (%)": "{:+.2f}%",
                "10D Profit Factor": "{:.2f}",
                "15D Win Rate (%)": "{:.1f}%",
                "15D Avg Ret (%)": "{:+.2f}%",
                "15D Med Ret (%)": "{:+.2f}%",
                "15D Profit Factor": "{:.2f}"
            }
            
            def color_returns(val):
                if isinstance(val, (int, float)):
                    if val > 0: return 'color: #00FF66;'
                    elif val < 0: return 'color: #FF4B4B;'
                return ''

            return_cols = [
                "5D Avg Ret (%)",
                "5D Med Ret (%)",
                "10D Avg Ret (%)",
                "10D Med Ret (%)",
                "15D Avg Ret (%)",
                "15D Med Ret (%)"
            ]

            styled_perf = perf_table.style.format(format_dict)
            styled_perf = style_apply_elementwise(styled_perf, color_returns, subset=return_cols)

            st.dataframe(styled_perf, use_container_width=True)
            
            st.subheader("📈 Forward Win Rate Comparison Across Buckets")
            fig_wr = go.Figure()
            fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['5D Win Rate (%)'], name='5-Day Horizon', marker_color='#29B6F6'))
            fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['10D Win Rate (%)'], name='10-Day Horizon', marker_color='#66BB6A'))
            fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['15D Win Rate (%)'], name='15-Day Horizon', marker_color='#FFA726'))
            fig_wr.add_hline(y=50, line_dash="dash", line_color="white", annotation_text="50% Parity")
            fig_wr.update_layout(
                barmode='group', template="plotly_dark", yaxis_title="Win Rate (%)",
                xaxis_title="RSI(2) Smoothed Bucket", height=420, margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_wr, use_container_width=True)

# ----------------------------------------------------
# TAB 2: LIVE MARKET SCREENER
# ----------------------------------------------------
with tab_screener:
    st.header("Live Cumulative RSI(2) Screener")
    st.write(
        "Screens constituents for current RSI(2), 5-Day and 10-Day smoothed readings, and shows "
        "preceding 5, 10, and 15-day returns alongside historical forward win rates."
    )
    
    col_s1, col_s2, col_s3 = st.columns([1.2, 1.5, 1])
    with col_s1:
        universe_choice = st.selectbox("Screener Universe", ["S&P 500", "Nasdaq 100", "Dow Jones 30", "Major ETFs"])
    with col_s2:
        bucket_filter = st.multiselect(
            "Filter Target Buckets (5-Day Avg RSI(2))",
            options=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'],
            default=['0-10', '11-20'],
            help="Defaults to oversold capitulation regimes (0-20)."
        )
    with col_s3:
        st.write("")
        st.write("")
        run_screener = st.button("🚀 Run Live Screen", type="primary", use_container_width=True)
        
    if run_screener:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Fetching constituents for {universe_choice}...")
        tickers = get_universe_tickers(universe_choice)
        progress_bar.progress(15)
        
        status_text.text(f"Downloading market history for {len(tickers)} stocks...")
        bulk_data = yf.download(tickers, period="2y", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        progress_bar.progress(60)
        
        status_text.text("Analyzing cumulative indicator regimes & preceding returns...")
        screener_rows = []
        
        for sym in tickers:
            try:
                if len(tickers) == 1:
                    df_sym = bulk_data.copy()
                else:
                    if sym in bulk_data.columns.levels[0]:
                        df_sym = bulk_data[sym].copy().dropna(how='all')
                    elif sym in bulk_data.columns.levels[1]:
                        df_sym = bulk_data.xs(sym, axis=1, level=1).copy().dropna(how='all')
                    else:
                        continue
                        
                if len(df_sym) < 40:
                    continue
                    
                processed = prepare_stock_data(df_sym)
                latest = processed.iloc[-1]
                
                b5 = str(latest['Bucket_5d'])
                b10 = str(latest['Bucket_10d'])
                
                if bucket_filter and b5 not in bucket_filter:
                    continue
                    
                sub_hist = processed[processed['Bucket_5d'] == b5].dropna(subset=['Fwd_5d', 'Fwd_10d'])
                if len(sub_hist) > 0:
                    hist_5d_wr = (sub_hist['Fwd_5d'] > 0).mean() * 100
                    hist_10d_wr = (sub_hist['Fwd_10d'] > 0).mean() * 100
                    hist_5d_avg = sub_hist['Fwd_5d'].mean()
                else:
                    hist_5d_wr, hist_10d_wr, hist_5d_avg = np.nan, np.nan, np.nan
                    
                screener_rows.append({
                    "Ticker": sym,
                    "Current Price": round(latest['Adj Close'] if 'Adj Close' in latest else latest['Close'], 2),
                    "RSI(2)": round(latest['RSI2'], 1),
                    "5D Avg RSI(2)": round(latest['RSI2_5d_avg'], 1),
                    "10D Avg RSI(2)": round(latest['RSI2_10d_avg'], 1),
                    "5D Bucket": b5,
                    "Prior 5D Ret (%)": round(latest['Prior_5d'], 2),
                    "Prior 10D Ret (%)": round(latest['Prior_10d'], 2),
                    "Prior 15D Ret (%)": round(latest['Prior_15d'], 2),
                    "Hist 5D Win Rate": round(hist_5d_wr, 1) if pd.notna(hist_5d_wr) else np.nan,
                    "Hist 10D Win Rate": round(hist_10d_wr, 1) if pd.notna(hist_10d_wr) else np.nan,
                    "Hist 5D Avg Ret (%)": round(hist_5d_avg, 2) if pd.notna(hist_5d_avg) else np.nan
                })
            except Exception:
                pass
                
        progress_bar.progress(100)
        status_text.empty()
        
        screen_df = pd.DataFrame(screener_rows)
        
        if not screen_df.empty:
            st.subheader(f"🟢 Filtered Signals: {len(screen_df)} Setups Found")
            
            screen_df = screen_df.sort_values(by="5D Avg RSI(2)", ascending=True)
            
            screener_col_cfg = {
                "
