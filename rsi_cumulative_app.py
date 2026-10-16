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
    "Parametric Multi-Day Smoothed RSI(2) Engine | Multi-Stock Portfolio Backtester (Up to 10 Tickers) | "
    "Year-by-Year Performance Breakdown | Dynamic Snapback Exits (SMA5, RSI70) | MAE Drawdown Tracking."
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
    
    # 5. True Maximum Adverse Excursion (MAE)
    min_low_5d = low_s.iloc[::-1].rolling(5).min().shift(1).iloc[::-1]
    min_low_10d = low_s.iloc[::-1].rolling(10).min().shift(1).iloc[::-1]
    
    raw_mae_5d = (min_low_5d - close_s) / close_s * 100
    raw_mae_10d = (min_low_10d - close_s) / close_s * 100
    
    df['MAE_5d'] = raw_mae_5d.clip(upper=0.0)
    df['MAE_10d'] = raw_mae_10d.clip(upper=0.0)
    
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
    """Aggregates win rate, return distributions, average MAE, and profit factor by bucket."""
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
                "RSI Bucket": b,
                "Signals": 0,
                "5D Win Rate (%)": 0.0,
                "5D Avg Ret (%)": 0.0,
                "5D Avg Drawdown (%)": 0.0,
                "5D Profit Factor": 0.0,
                "10D Win Rate (%)": 0.0,
                "10D Avg Ret (%)": 0.0,
                "10D Avg Drawdown (%)": 0.0,
                "10D Profit Factor": 0.0,
                "SMA5 Exit Win Rate (%)": 0.0,
                "SMA5 Exit Avg Ret (%)": 0.0,
                "SMA5 Avg Hold (Days)": 0.0,
                "SMA5 Profit Factor": 0.0,
                "RSI70 Exit Win Rate (%)": 0.0,
                "RSI70 Exit Avg Ret (%)": 0.0,
                "RSI70 Avg Hold (Days)": 0.0,
                "RSI70 Profit Factor": 0.0
            })
            continue
            
        def stats_fixed(ret_series, mae_series):
            wr = (ret_series > 0).mean() * 100
            avg_r = ret_series.mean()
            avg_mae = mae_series.mean()
            gains = ret_series[ret_series > 0].sum()
            losses = abs(ret_series[ret_series < 0].sum())
            pf = round(gains / losses, 2) if losses > 0 else np.nan
            return round(wr, 1), round(avg_r, 2), round(avg_mae, 2), pf
            
        def stats_dynamic(ret_series, days_series):
            wr = (ret_series > 0).mean() * 100
            avg_r = ret_series.mean()
            avg_d = days_series.mean()
            gains = ret_series[ret_series > 0].sum()
            losses = abs(ret_series[ret_series < 0].sum())
            pf = round(gains / losses, 2) if losses > 0 else np.nan
            return round(wr, 1), round(avg_r, 2), round(avg_d, 1), pf
            
        wr5, avg5, mae5, pf5 = stats_fixed(sub['Fwd_5d'], sub['MAE_5d'])
        wr10, avg10, mae10, pf10 = stats_fixed(sub['Fwd_10d'], sub['MAE_10d'])
        wr_sma, avg_sma, days_sma, pf_sma = stats_dynamic(sub['Fwd_SMA5_Ret'], sub['Fwd_SMA5_Days'])
        wr_rsi, avg_rsi, days_rsi, pf_rsi = stats_dynamic(sub['Fwd_RSI70_Ret'], sub['Fwd_RSI70_Days'])
        
        records.append({
            "RSI Bucket": b,
            "Signals": int(n),
            "5D Win Rate (%)": wr5,
            "5D Avg Ret (%)": avg5,
            "5D Avg Drawdown (%)": mae5,
            "5D Profit Factor": pf5,
            "10D Win Rate (%)": wr10,
            "10D Avg Ret (%)": avg10,
            "10D Avg Drawdown (%)": mae10,
            "10D Profit Factor": pf10,
            "SMA5 Exit Win Rate (%)": wr_sma,
            "SMA5 Exit Avg Ret (%)": avg_sma,
            "SMA5 Avg Hold (Days)": days_sma,
            "SMA5 Profit Factor": pf_sma,
            "RSI70 Exit Win Rate (%)": wr_rsi,
            "RSI70 Exit Avg Ret (%)": avg_rsi,
            "RSI70 Avg Hold (Days)": days_rsi,
            "RSI70 Profit Factor": pf_rsi
        })
        
    return pd.DataFrame(records).set_index("RSI Bucket")

def aggregate_yearly_performance(df, bucket_col, selected_buckets, exit_col_ret, exit_col_days=None, require_uptrend=False, fresh_entry_only=False):
    """Calculates yearly breakdown metrics for an asset under chosen bucket and exit criteria."""
    data = df.dropna(subset=[bucket_col, exit_col_ret]).copy()
    
    if require_uptrend:
        data = data[data['Above_200SMA'] == True]
        
    if fresh_entry_only:
        data = data[data[bucket_col] != data[bucket_col].shift(1)]
        
    data = data[data[bucket_col].astype(str).isin(selected_buckets)]
    
    if data.empty:
        return pd.DataFrame()
        
    data['Year'] = data.index.year
    records = []
    
    for yr, group in data.groupby('Year'):
        n_sigs = len(group)
        ret_s = group[exit_col_ret]
        
        wr = (ret_s > 0).mean() * 100
        avg_ret = ret_s.mean()
        
        gains = ret_s[ret_s > 0].sum()
        losses = abs(ret_s[ret_s < 0].sum())
        pf = round(gains / losses, 2) if losses > 0 else (np.nan if gains == 0 else 99.0)
        
        comp_ret = ((1 + ret_s / 100).prod() - 1) * 100
        avg_days = group[exit_col_days].mean() if exit_col_days and exit_col_days in group else np.nan
        
        records.append({
            "Year": str(yr),
            "Signals": int(n_sigs),
            "Win Rate (%)": round(wr, 1),
            "Avg Return (%)": round(avg_ret, 2),
            "Profit Factor": pf,
            "Annual Compounded Ret (%)": round(comp_ret, 2),
            "Avg Hold (Days)": round(avg_days, 1) if pd.notna(avg_days) else np.nan
        })
        
    all_ret = data[exit_col_ret]
    all_gains = all_ret[all_ret > 0].sum()
    all_losses = abs(all_ret[all_ret < 0].sum())
    all_pf = round(all_gains / all_losses, 2) if all_losses > 0 else np.nan
    all_comp = ((1 + all_ret / 100).prod() - 1) * 100
    all_days = data[exit_col_days].mean() if exit_col_days and exit_col_days in data else np.nan

    records.append({
        "Year": "All Years",
        "Signals": int(len(data)),
        "Win Rate (%)": round((all_ret > 0).mean() * 100, 1),
        "Avg Return (%)": round(all_ret.mean(), 2),
        "Profit Factor": all_pf,
        "Annual Compounded Ret (%)": round(all_comp, 2),
        "Avg Hold (Days)": round(all_days, 1) if pd.notna(all_days) else np.nan
    })
    
    return pd.DataFrame(records).set_index("Year")

# ==========================================
# 2. MULTI-STOCK PORTFOLIO ENGINE
# ==========================================

def run_portfolio_backtest(prepared_dfs, selected_buckets, bucket_col, indicator_col,
                           exit_mode="Close > SMA5", max_positions=5, initial_capital=100000.0,
                           require_uptrend=True, fresh_entry_only=True):
    """
    Day-by-day multi-stock portfolio simulation.
    Handles position sizing, cash management, signal ranking, exits, and annual accounting.
    """
    all_dates = sorted(list(set.union(*[set(df.index) for df in prepared_dfs.values()])))
    
    cash = float(initial_capital)
    open_positions = {}
    trade_log = []
    daily_equity_records = []
    
    for d in all_dates:
        # Step A: Process Exits
        closed_tickers = []
        for ticker, pos in open_positions.items():
            df = prepared_dfs[ticker]
            if d not in df.index:
                pos['days_held'] += 1
                continue
                
            row = df.loc[d]
            c_price = row['Adj Close'] if 'Adj Close' in row else row['Close']
            pos['days_held'] += 1
            
            exit_hit = False
            exit_reason = ""
            
            if exit_mode == "Close > SMA5" and c_price > row['SMA_5']:
                exit_hit = True
                exit_reason = "Close > SMA5"
            elif exit_mode == "RSI(2) > 70" and row['RSI2'] >= 70:
                exit_hit = True
                exit_reason = "RSI(2) >= 70"
            elif exit_mode == "Fixed 5-Day Hold" and pos['days_held'] >= 5:
                exit_hit = True
                exit_reason = "5-Day Limit"
            elif exit_mode == "Fixed 10-Day Hold" and pos['days_held'] >= 10:
                exit_hit = True
                exit_reason = "10-Day Limit"
                
            if exit_hit:
                proceeds = pos['shares'] * c_price
                cash += proceeds
                trade_ret = (c_price - pos['entry_price']) / pos['entry_price'] * 100
                
                trade_log.append({
                    "Ticker": ticker,
                    "Entry Date": pos['entry_date'].strftime('%Y-%m-%d'),
                    "Entry Price": round(pos['entry_price'], 2),
                    "Exit Date": d.strftime('%Y-%m-%d'),
                    "Exit Price": round(c_price, 2),
                    "Exit Reason": exit_reason,
                    "Return (%)": round(trade_ret, 2),
                    "Days Held": pos['days_held'],
                    "Year": d.year,
                    "Result": "Win" if trade_ret > 0 else "Loss"
                })
                closed_tickers.append(ticker)
                
        for t in closed_tickers:
            del open_positions[t]
            
        # Step B: Identify Candidate Entries
        candidates = []
        available_slots = max_positions - len(open_positions)
        
        for ticker, df in prepared_dfs.items():
            if ticker in open_positions or d not in df.index:
                continue
                
            row = df.loc[d]
            current_bucket = str(row[bucket_col])
            
            if current_bucket not in selected_buckets:
                continue
                
            if require_uptrend and not bool(row['Above_200SMA']):
                continue
                
            if fresh_entry_only:
                prev_loc = df.index.get_loc(d) - 1
                if prev_loc >= 0:
                    prev_bucket = str(df.iloc[prev_loc][bucket_col])
                    if current_bucket == prev_bucket:
                        continue
                        
            c_price = row['Adj Close'] if 'Adj Close' in row else row['Close']
            ind_val = row[indicator_col]
            candidates.append({
                "ticker": ticker,
                "price": c_price,
                "ind_val": ind_val
            })
            
        candidates.sort(key=lambda x: x['ind_val'])
        
        # Step C: Allocate Capital to Free Slots
        if available_slots > 0 and len(candidates) > 0:
            current_mtm = cash
            for t_open, p_open in open_positions.items():
                if d in prepared_dfs[t_open].index:
                    cur_px = prepared_dfs[t_open].loc[d]['Adj Close'] if 'Adj Close' in prepared_dfs[t_open].loc[d] else prepared_dfs[t_open].loc[d]['Close']
                    current_mtm += p_open['shares'] * cur_px
                else:
                    current_mtm += p_open['shares'] * p_open['entry_price']
                    
            target_slot_size = current_mtm / max_positions
            
            for cand in candidates[:available_slots]:
                invest_amt = min(cash, target_slot_size)
                if invest_amt >= 250.0:
                    shares = invest_amt / cand['price']
                    cash -= invest_amt
                    open_positions[cand['ticker']] = {
                        'entry_date': d,
                        'entry_price': cand['price'],
                        'shares': shares,
                        'cost_basis': invest_amt,
                        'days_held': 0
                    }
                    
        # Step D: Mark-to-Market Portfolio Equity
        total_equity = cash
        for t_open, p_open in open_positions.items():
            if d in prepared_dfs[t_open].index:
                cur_px = prepared_dfs[t_open].loc[d]['Adj Close'] if 'Adj Close' in prepared_dfs[t_open].loc[d] else prepared_dfs[t_open].loc[d]['Close']
                total_equity += p_open['shares'] * cur_px
            else:
                total_equity += p_open['shares'] * p_open['entry_price']
                
        daily_equity_records.append({"Date": d, "Equity": total_equity})
        
    equity_df = pd.DataFrame(daily_equity_records).set_index("Date")
    trades_df = pd.DataFrame(trade_log)
    
    # Step E: Aggregate Year-by-Year Results
    equity_df['Year'] = equity_df.index.year
    yearly_records = []
    
    for yr, grp in equity_df.groupby('Year'):
        start_eq = grp['Equity'].iloc[0]
        end_eq = grp['Equity'].iloc[-1]
        yr_return = ((end_eq / start_eq) - 1) * 100
        
        cummax = grp['Equity'].cummax()
        yr_dd = ((grp['Equity'] - cummax) / cummax).min() * 100
        
        yr_trades = trades_df[trades_df['Year'] == yr] if not trades_df.empty else pd.DataFrame()
        n_trades = len(yr_trades)
        
        if n_trades > 0:
            win_rate = (yr_trades['Result'] == 'Win').mean() * 100
            avg_trade_ret = yr_trades['Return (%)'].mean()
            gains = yr_trades[yr_trades['Return (%)'] > 0]['Return (%)'].sum()
            losses = abs(yr_trades[yr_trades['Return (%)'] < 0]['Return (%)'].sum())
            pf = round(gains / losses, 2) if losses > 0 else (np.nan if gains == 0 else 99.0)
            avg_days = yr_trades['Days Held'].mean()
        else:
            win_rate, avg_trade_ret, pf, avg_days = 0.0, 0.0, np.nan, 0.0
            
        yearly_records.append({
            "Year": str(yr),
            "Trades": int(n_trades),
            "Win Rate (%)": round(win_rate, 1),
            "Avg Trade Ret (%)": round(avg_trade_ret, 2),
            "Profit Factor": pf,
            "Annual Portfolio Ret (%)": round(yr_return, 2),
            "Max Drawdown (%)": round(yr_dd, 1),
            "Avg Hold (Days)": round(avg_days, 1)
        })
        
    tot_ret = ((equity_df['Equity'].iloc[-1] / initial_capital) - 1) * 100
    overall_cummax = equity_df['Equity'].cummax()
    overall_dd = ((equity_df['Equity'] - overall_cummax) / overall_cummax).min() * 100
    
    if not trades_df.empty:
        all_wr = (trades_df['Result'] == 'Win').mean() * 100
        all_avg_ret = trades_df['Return (%)'].mean()
        all_gains = trades_df[trades_df['Return (%)'] > 0]['Return (%)'].sum()
        all_losses = abs(trades_df[trades_df['Return (%)'] < 0]['Return (%)'].sum())
        all_pf = round(all_gains / all_losses, 2) if all_losses > 0 else np.nan
        all_hold = trades_df['Days Held'].mean()
    else:
        all_wr, all_avg_ret, all_pf, all_hold = 0.0, 0.0, np.nan, 0.0
        
    yearly_records.append({
        "Year": "All Years",
        "Trades": int(len(trades_df)),
        "Win Rate (%)": round(all_wr, 1),
        "Avg Trade Ret (%)": round(all_avg_ret, 2),
        "Profit Factor": all_pf,
        "Annual Portfolio Ret (%)": round(tot_ret, 2),
        "Max Drawdown (%)": round(overall_dd, 1),
        "Avg Hold (Days)": round(all_hold, 1)
    })
    
    yearly_summary_df = pd.DataFrame(yearly_records).set_index("Year")
    return yearly_summary_df, equity_df, trades_df

# ==========================================
# 3. UNIVERSE FETCHERS & PARSERS
# ==========================================

DJIA_STATIC_FALLBACK = [
    "AAPL", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
]

@st.cache_data(show_spinner=False, ttl=86400)
def get_universe_tickers(universe_name):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36"
    }
    
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
            except Exception:
                pass
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
    """Downloads daily data with safe multi-line arguments preventing line truncation."""
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )
    if df.empty:
        return None
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
# 4. INTERFACE LAYOUT & TABS
# ==========================================

tab_study, tab_watchlist, tab_portfolio, tab_screener = st.tabs([
    "Single Ticker Study & Current Status",
    "Custom Watchlist Analyzer",
    "📁 Multi-Stock Portfolio Backtester",
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
        study_ticker = st.text_input("Ticker Symbol", value="NVDA", key="study_sym").upper().strip()
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
            indicator_col_map = {
                "2-Day Average RSI(2)": "RSI2_2d_avg",
                "3-Day Average RSI(2)": "RSI2_3d_avg",
                "5-Day Average RSI(2)": "RSI2_5d_avg",
                "10-Day Average RSI(2)": "RSI2_10d_avg"
            }
            
            active_bucket_col = bucket_map[study_indicator]
            active_ind_col = indicator_col_map[study_indicator]
            fresh_only = (signal_mode == "Fresh Entry Day Only (No Overlap)")
            
            latest_row = processed_df.iloc[-1]
            curr_price = latest_row['Adj Close'] if 'Adj Close' in latest_row else latest_row['Close']
            curr_rsi2 = latest_row['RSI2']
            curr_ind_val = latest_row[active_ind_col]
            curr_bucket = str(latest_row[active_bucket_col])
            is_above_200 = bool(latest_row['Above_200SMA'])
            
            st.subheader(f"Current Market Status: {study_ticker}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Current Price", f"${curr_price:.2f}")
            m2.metric("RSI(2) (Single Bar)", f"{curr_rsi2:.1f}")
            m3.metric(f"Current {study_indicator}", f"{curr_ind_val:.1f}")
            m4.metric("Active Regime Bucket", curr_bucket)
            m5.metric("Trend (> 200 SMA)", "Uptrend (Bullish)" if is_above_200 else "Downtrend (Bearish)")
            
            perf_table = aggregate_bucket_performance(
                processed_df, 
                active_bucket_col, 
                fresh_entry_only=fresh_only,
                require_uptrend=require_uptrend_study
            )
            
            if curr_bucket in perf_table.index and perf_table.loc[curr_bucket, "Signals"] > 0:
                c_wr = perf_table.loc[curr_bucket, "5D Win Rate (%)"]
                c_ret = perf_table.loc[curr_bucket, "5D Avg Ret (%)"]
                c_mae = perf_table.loc[curr_bucket, "5D Avg Drawdown (%)"]
                c_sma_wr = perf_table.loc[curr_bucket, "SMA5 Exit Win Rate (%)"]
                c_sma_pf = perf_table.loc[curr_bucket, "SMA5 Profit Factor"]
                c_sma_days = perf_table.loc[curr_bucket, "SMA5 Avg Hold (Days)"]
                
                st.info(
                    f"**Empirical Baseline for Current Regime ({curr_bucket})**: "
                    f"When **{study_ticker}** trades in the **{curr_bucket}** bucket, its historical **5-day forward win rate is {c_wr}%** "
                    f"(Avg Return: **{c_ret:+.2f}%**, Avg Drawdown: **{c_mae:.2f}%**). "
                    f"Exiting dynamically on the first close above the 5-day SMA has yielded a **{c_sma_wr}% win rate** and a **{c_sma_pf} Profit Factor** over an average hold of **{c_sma_days:.1f} days**."
                )
            
            st.subheader(f"Expectancy & Risk Matrix across All Buckets: {study_ticker}")
            
            study_col_cfg = {
                "Signals": st.column_config.NumberColumn(format="%d"),
                "5D Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "5D Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "5D Avg Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%", help="Average worst adverse price drop experienced during the 5 days."),
                "5D Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                "10D Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "10D Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "10D Avg Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%", help="Average worst adverse price drop experienced during the 10 days."),
                "10D Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                "SMA5 Exit Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%", help="Win rate when exiting on the first close above 5-Day SMA."),
                "SMA5 Exit Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "SMA5 Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d"),
                "SMA5 Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                "RSI70 Exit Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%", help="Win rate when exiting on the first RSI(2) > 70 thrust."),
                "RSI70 Exit Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "RSI70 Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d"),
                "RSI70 Profit Factor": st.column_config.NumberColumn(format="%.2f")
            }

            st.dataframe(perf_table, column_config=study_col_cfg, use_container_width=True)
            
            st.subheader("Quantitative Edge Visualizer")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig_wr = go.Figure()
                fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['5D Win Rate (%)'], name='Fixed 5-Day', marker_color='#42A5F5'))
                fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['10D Win Rate (%)'], name='Fixed 10-Day', marker_color='#7E57C2'))
                fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['SMA5 Exit Win Rate (%)'], name='Dynamic (Close > SMA5)', marker_color='#26A69A'))
                fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['RSI70 Exit Win Rate (%)'], name='Dynamic (RSI2 > 70)', marker_color='#FFA726'))
                fig_wr.add_hline(y=50, line_dash="dash", line_color="white", annotation_text="50% Parity")
                fig_wr.update_layout(title="Win Rate (%) Across Exit Strategies", barmode='group', template="plotly_dark", yaxis_title="Win Rate (%)", xaxis_title="RSI(2) Bucket", height=400, margin=
