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
    "Parametric Multi-Day Smoothed RSI(2) Engine | Dynamic Snapback Exits (SMA5, RSI70) | "
    "200-Day Trend Filtering | Maximum Adverse Excursion (MAE) Pain Tracking | Macro S&P 500 Breadth Context."
)

# ==========================================
# 1. CORE QUANTITATIVE ENGINE
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
    
    # 2. Trend & Benchmark Overlays
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
    
    # 5. Maximum Adverse Excursion (MAE) Calculation (Worst Drawdown in Holding Window)
    min_low_5d = low_s.iloc[::-1].rolling(5).min().shift(1).iloc[::-1]
    min_low_10d = low_s.iloc[::-1].rolling(10).min().shift(1).iloc[::-1]
    df['MAE_5d'] = (min_low_5d - close_s) / close_s * 100
    df['MAE_10d'] = (min_low_10d - close_s) / close_s * 100
    
    # 6. Dynamic Snapback Exits (Vector-Accelerated Simulation)
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
        
        # Exit on Close > 5-Day SMA (Capped at 15 bars)
        for k in range(1, 16):
            if c_vals[i + k] > sma5_vals[i + k]:
                fwd_sma5_ret[i] = (c_vals[i + k] - c0) / c0 * 100
                fwd_sma5_days[i] = k
                break
        else:
            fwd_sma5_ret[i] = (c_vals[i + 15] - c0) / c0 * 100
            fwd_sma5_days[i] = 15
            
        # Exit on RSI(2) > 70 Overbought Spike (Capped at 15 bars)
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
    
    # 7. Discrete 10-Point Bucketing (0-10, 11-20, ..., 91-100)
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
# 2. UNIVERSE FETCHERS
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

# ==========================================
# 3. INTERFACE LAYOUT & TABS
# ==========================================

tab_study, tab_screener = st.tabs(["Forward Return Matrix & Dynamic Exits (Study)", "Live RSI(2) Screener"])

# ----------------------------------------------------
# TAB 1: FORWARD RETURN DISTRIBUTION ANALYZER
# ----------------------------------------------------
with tab_study:
    st.header("Historical Expectancy Matrix & Risk Analysis")
    st.write(
        "Evaluate fixed-time returns, dynamic snapback exits (Close > SMA5, RSI2 > 70), "
        "and Maximum Adverse Excursion (drawdown pain) across 10-point RSI(2) regimes."
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
            
        if raw_df is None or len(raw_df) < 220:
            st.error(f"Insufficient history found for {study_ticker} (minimum 220 trading sessions required).")
        else:
            processed_df = prepare_stock_data(raw_df)
            
            bucket_map = {
                "2-Day Average RSI(2)": "Bucket_2d",
                "3-Day Average RSI(2)": "Bucket_3d",
                "5-Day Average RSI(2)": "Bucket_5d",
                "10-Day Average RSI(2)": "Bucket_10d"
            }
            active_bucket_col = bucket_map[study_indicator]
            fresh_only = (signal_mode == "Fresh Entry Day Only (No Overlap)")
            
            perf_table = aggregate_bucket_performance(
                processed_df, 
                active_bucket_col, 
                fresh_entry_only=fresh_only,
                require_uptrend=require_uptrend_study
            )
            
            st.subheader(f"Expectancy & Risk Matrix: {study_ticker} ({study_indicator})")
            
            study_col_cfg = {
                "Signals": st.column_config.NumberColumn(format="%d"),
                "5D Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "5D Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "5D Med Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%", help="Median worst intraday drawdown experienced over 5 days."),
                "10D Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "10D Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "10D Med Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%", help="Median worst intraday drawdown experienced over 10 days."),
                "SMA5 Exit Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%", help="Win rate when exiting on the first close above 5-Day SMA."),
                "SMA5 Exit Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "SMA5 Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d", help="Average holding period to achieve Close > SMA5."),
                "RSI70 Exit Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%", help="Win rate when exiting on the first RSI(2) > 70 thrust."),
                "RSI70 Exit Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "RSI70 Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d", help="Average holding period to achieve RSI(2) > 70.")
            }

            st.dataframe(perf_table, column_config=study_col_cfg, use_container_width=True)
            
            # --- Visual Expectancy Comparison ---
            st.subheader("Fixed Horizon vs. Dynamic Snapback Win Rates")
            fig_exits = go.Figure()
            fig_exits.add_trace(go.Bar(x=perf_table.index, y=perf_table['5D Win Rate (%)'], name='Fixed 5-Day Exit', marker_color='#42A5F5'))
            fig_exits.add_trace(go.Bar(x=perf_table.index, y=perf_table['10D Win Rate (%)'], name='Fixed 10-Day Exit', marker_color='#7E57C2'))
            fig_exits.add_trace(go.Bar(x=perf_table.index, y=perf_table['SMA5 Exit Win Rate (%)'], name='Dynamic Exit (Close > SMA5)', marker_color='#26A69A'))
            fig_exits.add_trace(go.Bar(x=perf_table.index, y=perf_table['RSI70 Exit Win Rate (%)'], name='Dynamic Exit (RSI2 > 70)', marker_color='#FFA726'))
            fig_exits.add_hline(y=50, line_dash="dash", line_color="white", annotation_text="50% Benchmark")
            fig_exits.update_layout(
                barmode='group', 
                template="plotly_dark", 
                yaxis_title="Win Rate (%)",
                xaxis_title="RSI(2) Smoothed Bucket", 
                height=440, 
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_exits, use_container_width=True)

# ----------------------------------------------------
# TAB 2: LIVE MARKET SCREENER
# ----------------------------------------------------
with tab_screener:
    st.header("Live Cumulative RSI(2) Screener")
    st.write(
        "Screens constituents for current RSI(2) regimes, checks 200-day trend qualification, "
        "and synchronizes signals with broad S&P 500 macro breadth."
    )
    
    col_s1, col_s2, col_s3, col_s4 = st.columns([1.1, 1.2, 1.3, 0.8])
    with col_s1:
        universe_choice = st.selectbox("Screener Universe", ["S&P 500", "Nasdaq 100", "Dow Jones 30", "Major ETFs"])
    with col_s2:
        screener_lookback = st.selectbox(
            "Filter By Indicator",
            ["5D Avg RSI(2)", "3D Avg RSI(2)", "2D Avg RSI(2)", "10D Avg RSI(2)"],
            index=0
        )
    with col_s3:
        bucket_filter = st.multiselect(
            "Filter Target Buckets",
            options=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'],
            default=['0-10', '11-20'],
            help="Defaults to high-edge oversold washouts (0-20)."
        )
    with col_s4:
        st.write("")
        st.write("")
        run_screener = st.button("Run Live Screen", type="primary", use_container_width=True)
        
    require_uptrend_screener = st.checkbox(
        "Enforce 200-Day SMA Trend Gate (Only show stocks trading above 200-Day SMA)",
        value=True,
        help="Restricts screener to healthy structural uptrends, filtering out secular losers."
    )
        
    if run_screener:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Macro Breadth Analysis: Fetch SPY Benchmark Context
        status_text.text("Checking broad market S&P 500 (SPY) breadth...")
        spy_raw = yf.download("SPY", period="6mo", interval="1d", auto_adjust=False, progress=False)
        spy_prep = prepare_stock_data(spy_raw)
        latest_spy_rsi5 = spy_prep['RSI2_5d_avg'].iloc[-1]
        
        if latest_spy_rsi5 <= 25:
            macro_tag = "Systemic Flush (Broad Market Oversold - High Edge)"
            macro_desc = "Broad market capitulation detected (SPY 5D Avg RSI <= 25). Historical recovery odds are heavily elevated."
            macro_color = "success"
        else:
            macro_tag = "Idiosyncratic Dip (Selective / Stock-Specific)"
            macro_desc = f"Broad market is neutral/bullish (SPY 5D Avg RSI = {latest_spy_rsi5:.1f}). Dips are stock-specific."
            macro_color = "info"
            
        progress_bar.progress(15)
        
        # 2. Universe Roster & Bulk Download
        status_text.text(f"Fetching constituents for {universe_choice}...")
        tickers = get_universe_tickers(universe_choice)
        progress_bar.progress(30)
        
        status_text.text(f"Downloading daily market data for {len(tickers)} stocks...")
        bulk_data = yf.download(tickers, period="2y", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        progress_bar.progress(70)
        
        status_text.text("Evaluating indicators, MAE drawdowns, and dynamic snapbacks...")
        screener_rows = []
        
        lookback_key_map = {
            "2D Avg RSI(2)": "Bucket_2d",
            "3D Avg RSI(2)": "Bucket_3d",
            "5D Avg RSI(2)": "Bucket_5d",
            "10D Avg RSI(2)": "Bucket_10d"
        }
        active_filter_bucket = lookback_key_map[screener_lookback]
        
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
                        
                if len(df_sym) < 220:
                    continue
                    
                processed = prepare_stock_data(df_sym)
                latest = processed.iloc[-1]
                
                # Check 200 SMA Gate
                is_above_200 = bool(latest['Above_200SMA'])
                if require_uptrend_screener and not is_above_200:
                    continue
                    
                target_b = str(latest[active_filter_bucket])
                if bucket_filter and target_b not in bucket_filter:
                    continue
                    
                # Calculate quick 2-year forward historical win rate & median MAE for this stock in this bucket
                sub_hist = processed[processed[active_filter_bucket] == target_b].dropna(subset=['Fwd_5d', 'MAE_5d', 'Fwd_SMA5_Ret'])
                if len(sub_hist) > 0:
                    h_5d_wr = (sub_hist['Fwd_5d'] > 0).mean() * 100
                    h_5d_avg = sub_hist['Fwd_5d'].mean()
                    h_mae = sub_hist['MAE_5d'].median()
                    h_sma5_wr = (sub_hist['Fwd_SMA5_Ret'] > 0).mean() * 100
                else:
                    h_5d_wr, h_5d_avg, h_mae, h_sma5_wr = np.nan, np.nan, np.nan, np.nan
                    
                screener_rows.append({
                    "Ticker": sym,
                    "Current Price": round(latest['Adj Close'] if 'Adj Close' in latest else latest['Close'], 2),
                    "RSI(2)": round(latest['RSI2'], 1),
                    "2D Avg": round(latest['RSI2_2d_avg'], 1),
                    "3D Avg": round(latest['RSI2_3d_avg'], 1),
                    "5D Avg": round(latest['RSI2_5d_avg'], 1),
                    "10D Avg": round(latest['RSI2_10d_avg'], 1),
                    "Active Bucket": target_b,
                    "Above 200 SMA": "Yes" if is_above_200 else "No",
                    "Market Context": "Systemic" if latest_spy_rsi5 <= 25 else "Idiosyncratic",
                    "Prior 5D Ret (%)": round(latest['Prior_5d'], 2),
                    "Prior 10D Ret (%)": round(latest['Prior_10d'], 2),
                    "Hist 5D Win Rate": round(h_5d_wr, 1) if pd.notna(h_5d_wr) else np.nan,
                    "Hist 5D Avg Ret (%)": round(h_5d_avg, 2) if pd.notna(h_5d_avg) else np.nan,
                    "Hist Med Drawdown (%)": round(h_mae, 2) if pd.notna(h_mae) else np.nan,
                    "Hist SMA5 Exit Win Rate": round(h_sma5_wr, 1) if pd.notna(h_sma5_wr) else np.nan
                })
            except Exception:
                pass
                
        progress_bar.progress(100)
        status_text.empty()
        
        # Display S&P 500 Macro Context Banner
        if macro_color == "success":
            st.success(f"Macro Breadth Regime: {macro_tag} | {macro_desc}")
        else:
            st.info(f"Macro Breadth Regime: {macro_tag} | {macro_desc}")
        
        screen_df = pd.DataFrame(screener_rows)
        
        if not screen_df.empty:
            st.subheader(f"Filtered Setups: {len(screen_df)} Opportunities Found")
            screen_df = screen_df.sort_values(by="5D Avg", ascending=True)
            
            screener_col_cfg = {
                "Ticker": st.column_config.TextColumn(help="Asset Ticker Symbol."),
                "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                "RSI(2)": st.column_config.NumberColumn(help="Single-day RSI(2) close.", format="%.1f"),
                "2D Avg": st.column_config.NumberColumn(help="Connors 2-Day Smoothed RSI(2).", format="%.1f"),
                "3D Avg": st.column_config.NumberColumn(help="Connors 3-Day Smoothed RSI(2).", format="%.1f"),
                "5D Avg": st.column_config.NumberColumn(help="5-Day Rolling Average RSI(2).", format="%.1f"),
                "10D Avg": st.column_config.NumberColumn(help="10-Day Rolling Average RSI(2).", format="%.1f"),
                "Active Bucket": st.column_config.TextColumn(help=f"Bucket level under {screener_lookback}."),
                "Above 200 SMA": st.column_config.TextColumn(help="Trend verification filter."),
                "Market Context": st.column_config.TextColumn(help="Systemic (SPY Oversold) vs Idiosyncratic (Stock-Specific)."),
                "Prior 5D Ret (%)": st.column_config.NumberColumn(help="Trailing 5-day return momentum.", format="%+.2f%%"),
                "Prior 10D Ret (%)": st.column_config.NumberColumn(help="Trailing 10-day return momentum.", format="%+.2f%%"),
                "Hist 5D Win Rate": st.column_config.ProgressColumn(help="2-year historical 5-day forward win rate.", format="%.1f%%", min_value=0, max_value=100),
                "Hist 5D Avg Ret (%)": st.column_config.NumberColumn(help="Average 5-day forward return.", format="%+.2f%%"),
                "Hist Med Drawdown (%)": st.column_config.NumberColumn(help="Median 5-day Maximum Adverse Excursion (worst drawdown experienced).", format="%.2f%%"),
                "Hist SMA5 Exit Win Rate": st.column_config.ProgressColumn(help="Win rate exiting dynamically when Close > 5-Day SMA.", format="%.1f%%", min_value=0, max_value=100)
            }
            
            st.dataframe(screen_df, column_config=screener_col_cfg, use_container_width=True)
            
            st.download_button(
                label="Download Screener Results as CSV",
                data=screen_df.to_csv(index=False).encode('utf-8'),
                file_name=f"rsi2_advanced_scan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        else:
            bucket_str = ", ".join(bucket_filter)
            st.info(f"No stocks in {universe_choice} currently match the selected criteria ({bucket_str}) with trend gates active.")
