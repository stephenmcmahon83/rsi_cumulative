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
    "Parametric Multi-Day Smoothed RSI(2) Engine | Next-Day Open Limit Order Execution (ATR Scaled) | "
    "Fill Rate Analytics | Multi-Stock & Universe Portfolio Backtester | Russell 2000 & S&P 500."
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

def calculate_atr(df, period=10):
    """Calculates Wilder's Average True Range (ATR)."""
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    prev_close = df[price_col].shift(1)
    high = df['High']
    low = df['Low']
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def prepare_stock_data(df):
    """Computes technical indicators, moving averages, preceding momentum, and ATR(10)."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    close_s = df[price_col]
    
    # 1. Core Indicators & Lookback Horizons
    df['RSI2'] = calculate_wilders_rsi(close_s, period=2)
    df['RSI2_2d_avg'] = df['RSI2'].rolling(window=2).mean()
    df['RSI2_3d_avg'] = df['RSI2'].rolling(window=3).mean()
    df['RSI2_5d_avg'] = df['RSI2'].rolling(window=5).mean()
    df['RSI2_10d_avg'] = df['RSI2'].rolling(window=10).mean()
    df['ATR10'] = calculate_atr(df, period=10)
    
    # 2. Trend Overlays
    df['SMA_5'] = close_s.rolling(window=5).mean()
    df['SMA_200'] = close_s.rolling(window=200).mean()
    df['Above_200SMA'] = close_s > df['SMA_200']
    
    # 3. Preceding Returns (Trailing Momentum)
    df['Prior_5d'] = (close_s - close_s.shift(5)) / close_s.shift(5) * 100
    df['Prior_10d'] = (close_s - close_s.shift(10)) / close_s.shift(10) * 100
    df['Prior_15d'] = (close_s - close_s.shift(15)) / close_s.shift(15) * 100
    
    # 4. Discrete 10-Point Bucketing
    bins = [-0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.001]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    
    df['Bucket_2d'] = pd.cut(df['RSI2_2d_avg'], bins=bins, labels=labels)
    df['Bucket_3d'] = pd.cut(df['RSI2_3d_avg'], bins=bins, labels=labels)
    df['Bucket_5d'] = pd.cut(df['RSI2_5d_avg'], bins=bins, labels=labels)
    df['Bucket_10d'] = pd.cut(df['RSI2_10d_avg'], bins=bins, labels=labels)
    
    return df

def simulate_single_ticker_execution(df, bucket_col, atr_pct=0.0, fresh_entry_only=False, require_uptrend=False):
    """
    Simulates Next-Day Limit Order Execution: Open - (atr_pct * ATR10).
    Tracks fill status, dynamic exits, and MAE from entry price.
    """
    n = len(df)
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    c_vals = df[price_col].values
    o_vals = df['Open'].values
    l_vals = df['Low'].values
    sma5_vals = df['SMA_5'].values
    rsi_vals = df['RSI2'].values
    atr_vals = df['ATR10'].values
    b_vals = df[bucket_col].astype(str).values
    uptrend_vals = df['Above_200SMA'].values
    
    records = []
    atr_mult = atr_pct / 100.0
    
    for i in range(15, n - 16):
        # Signal Evaluated on Day T
        b_curr = b_vals[i]
        if b_curr == 'nan' or b_curr == 'None':
            continue
            
        if require_uptrend and not uptrend_vals[i]:
            continue
            
        if fresh_entry_only and b_vals[i] == b_vals[i - 1]:
            continue
            
        signal_date = df.index[i]
        atr_t = atr_vals[i]
        
        # Day T+1 Execution Check
        open_next = o_vals[i + 1]
        low_next = l_vals[i + 1]
        limit_price = open_next - (atr_mult * atr_t)
        
        # Check Fill Condition on Day T+1
        if low_next <= limit_price:
            filled = True
            entry_price = min(open_next, limit_price)  # Price improvement on gaps
        else:
            filled = False
            entry_price = np.nan
            
        trade_data = {
            "Signal_Date": signal_date,
            "Bucket": b_curr,
            "Year": signal_date.year,
            "Filled": filled,
            "Entry_Price": entry_price
        }
        
        if filled:
            # Fixed 5-Day & 10-Day Exits (evaluated from bar i+1)
            c_fwd5 = c_vals[i + 1 + 5]
            c_fwd10 = c_vals[i + 1 + 10]
            trade_data['Fwd_5d'] = (c_fwd5 - entry_price) / entry_price * 100
            trade_data['Fwd_10d'] = (c_fwd10 - entry_price) / entry_price * 100
            
            # MAE Drawdown Tracking (Worst drop over holding period)
            min_low_5d = np.min(l_vals[i + 1 : i + 1 + 6])
            min_low_10d = np.min(l_vals[i + 1 : i + 1 + 11])
            trade_data['MAE_5d'] = min(0.0, (min_low_5d - entry_price) / entry_price * 100)
            trade_data['MAE_10d'] = min(0.0, (min_low_10d - entry_price) / entry_price * 100)
            
            # Dynamic Snapback: Close > 5-Day SMA
            for k in range(1, 16):
                bar_idx = i + 1 + (k - 1)
                if c_vals[bar_idx] > sma5_vals[bar_idx]:
                    trade_data['SMA5_Ret'] = (c_vals[bar_idx] - entry_price) / entry_price * 100
                    trade_data['SMA5_Days'] = k
                    break
            else:
                trade_data['SMA5_Ret'] = (c_vals[i + 1 + 15] - entry_price) / entry_price * 100
                trade_data['SMA5_Days'] = 15
                
            # Dynamic Snapback: RSI(2) > 70
            for k in range(1, 16):
                bar_idx = i + 1 + (k - 1)
                if rsi_vals[bar_idx] >= 70:
                    trade_data['RSI70_Ret'] = (c_vals[bar_idx] - entry_price) / entry_price * 100
                    trade_data['RSI70_Days'] = k
                    break
            else:
                trade_data['RSI70_Ret'] = (c_vals[i + 1 + 15] - entry_price) / entry_price * 100
                trade_data['RSI70_Days'] = 15
        else:
            trade_data['Fwd_5d'] = np.nan
            trade_data['Fwd_10d'] = np.nan
            trade_data['MAE_5d'] = np.nan
            trade_data['MAE_10d'] = np.nan
            trade_data['SMA5_Ret'] = np.nan
            trade_data['SMA5_Days'] = np.nan
            trade_data['RSI70_Ret'] = np.nan
            trade_data['RSI70_Days'] = np.nan
            
        records.append(trade_data)
        
    return pd.DataFrame(records)

def aggregate_bucket_performance(sim_df):
    """Compiles statistics by 10-point bucket reflecting limit fills."""
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    records = []
    
    for b in labels:
        sub = sim_df[sim_df['Bucket'] == b]
        sigs = len(sub)
        filled_sub = sub[sub['Filled'] == True]
        n_filled = len(filled_sub)
        
        fill_rate = (n_filled / sigs * 100) if sigs > 0 else 0.0
        
        if n_filled == 0:
            records.append({
                "RSI Bucket": b, "Signals": sigs, "Filled": 0, "Fill Rate (%)": fill_rate,
                "5D Win Rate (%)": 0.0, "5D Avg Ret (%)": 0.0, "5D Avg Drawdown (%)": 0.0, "5D Profit Factor": 0.0,
                "10D Win Rate (%)": 0.0, "10D Avg Ret (%)": 0.0, "10D Avg Drawdown (%)": 0.0, "10D Profit Factor": 0.0,
                "SMA5 Exit Win Rate (%)": 0.0, "SMA5 Exit Avg Ret (%)": 0.0, "SMA5 Avg Hold (Days)": 0.0, "SMA5 Profit Factor": 0.0,
                "RSI70 Exit Win Rate (%)": 0.0, "RSI70 Exit Avg Ret (%)": 0.0, "RSI70 Avg Hold (Days)": 0.0, "RSI70 Profit Factor": 0.0
            })
            continue
            
        def stats_fixed(ret_series, mae_series):
            wr = (ret_series > 0).mean() * 100
            avg_r = ret_series.mean()
            avg_mae = mae_series.mean()
            gains = ret_series[ret_series > 0].sum()
            losses = abs(ret_series[ret_series < 0].sum())
            pf = round(gains / losses, 2) if losses > 0 else (np.nan if gains == 0 else 99.0)
            return round(wr, 1), round(avg_r, 2), round(avg_mae, 2), pf
            
        def stats_dynamic(ret_series, days_series):
            wr = (ret_series > 0).mean() * 100
            avg_r = ret_series.mean()
            avg_d = days_series.mean()
            gains = ret_series[ret_series > 0].sum()
            losses = abs(ret_series[ret_series < 0].sum())
            pf = round(gains / losses, 2) if losses > 0 else (np.nan if gains == 0 else 99.0)
            return round(wr, 1), round(avg_r, 2), round(avg_d, 1), pf
            
        wr5, avg5, mae5, pf5 = stats_fixed(filled_sub['Fwd_5d'], filled_sub['MAE_5d'])
        wr10, avg10, mae10, pf10 = stats_fixed(filled_sub['Fwd_10d'], filled_sub['MAE_10d'])
        wr_sma, avg_sma, days_sma, pf_sma = stats_dynamic(filled_sub['SMA5_Ret'], filled_sub['SMA5_Days'])
        wr_rsi, avg_rsi, days_rsi, pf_rsi = stats_dynamic(filled_sub['RSI70_Ret'], filled_sub['RSI70_Days'])
        
        records.append({
            "RSI Bucket": b,
            "Signals": int(sigs),
            "Filled": int(n_filled),
            "Fill Rate (%)": round(fill_rate, 1),
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

def aggregate_yearly_performance(sim_df, selected_buckets, exit_col_ret, exit_col_days=None):
    """Calculates yearly breakdown metrics reflecting limit fills."""
    sub = sim_df[sim_df['Bucket'].isin(selected_buckets)].copy()
    if sub.empty:
        return pd.DataFrame()
        
    records = []
    for yr, group in sub.groupby('Year'):
        sigs = len(group)
        filled_grp = group[group['Filled'] == True]
        n_filled = len(filled_grp)
        fill_rate = (n_filled / sigs * 100) if sigs > 0 else 0.0
        
        if n_filled > 0:
            ret_s = filled_grp[exit_col_ret]
            wr = (ret_s > 0).mean() * 100
            avg_ret = ret_s.mean()
            gains = ret_s[ret_s > 0].sum()
            losses = abs(ret_s[ret_s < 0].sum())
            pf = round(gains / losses, 2) if losses > 0 else (np.nan if gains == 0 else 99.0)
            comp_ret = ((1 + ret_s / 100).prod() - 1) * 100
            avg_days = filled_grp[exit_col_days].mean() if exit_col_days and exit_col_days in filled_grp else np.nan
        else:
            wr, avg_ret, pf, comp_ret, avg_days = 0.0, 0.0, np.nan, 0.0, np.nan
            
        records.append({
            "Year": str(yr),
            "Signals": int(sigs),
            "Filled": int(n_filled),
            "Fill Rate (%)": round(fill_rate, 1),
            "Win Rate (%)": round(wr, 1),
            "Avg Return (%)": round(avg_ret, 2),
            "Profit Factor": pf,
            "Annual Compounded Ret (%)": round(comp_ret, 2),
            "Avg Hold (Days)": round(avg_days, 1) if pd.notna(avg_days) else np.nan
        })
        
    # All Years Summary
    tot_sigs = len(sub)
    tot_filled = sub[sub['Filled'] == True]
    all_n_filled = len(tot_filled)
    all_fr = (all_n_filled / tot_sigs * 100) if tot_sigs > 0 else 0.0
    
    if all_n_filled > 0:
        all_ret = tot_filled[exit_col_ret]
        all_wr = (all_ret > 0).mean() * 100
        all_avg_ret = all_ret.mean()
        all_gains = all_ret[all_ret > 0].sum()
        all_losses = abs(all_ret[all_ret < 0].sum())
        all_pf = round(all_gains / all_losses, 2) if all_losses > 0 else np.nan
        all_comp = ((1 + all_ret / 100).prod() - 1) * 100
        all_days = tot_filled[exit_col_days].mean() if exit_col_days and exit_col_days in tot_filled else np.nan
    else:
        all_wr, all_avg_ret, all_pf, all_comp, all_days = 0.0, 0.0, np.nan, 0.0, np.nan
        
    records.append({
        "Year": "All Years",
        "Signals": int(tot_sigs),
        "Filled": int(all_n_filled),
        "Fill Rate (%)": round(all_fr, 1),
        "Win Rate (%)": round(all_wr, 1),
        "Avg Return (%)": round(all_avg_ret, 2),
        "Profit Factor": all_pf,
        "Annual Compounded Ret (%)": round(all_comp, 2),
        "Avg Hold (Days)": round(all_days, 1) if pd.notna(all_days) else np.nan
    })
    
    return pd.DataFrame(records).set_index("Year")

# ==========================================
# 2. MULTI-STOCK & UNIVERSE PORTFOLIO ENGINE
# ==========================================

def run_portfolio_backtest(prepared_dfs, selected_buckets, bucket_col, indicator_col,
                           atr_pct=0.0, exit_mode="Close > SMA5", max_positions=5, initial_capital=100000.0,
                           require_uptrend=True, fresh_entry_only=True):
    """
    Day-by-day portfolio simulator supporting Next-Day Limit Orders (Open - X% ATR10).
    Manages slot allocations, signal prioritization, order fills, dynamic exits, and annual accounting.
    """
    all_dates = sorted(list(set.union(*[set(df.index) for df in prepared_dfs.values()])))
    
    cash = float(initial_capital)
    open_positions = {}
    pending_orders = []  # [{ticker, limit_price, signal_date, ind_val}]
    trade_log = []
    daily_equity_records = []
    
    atr_mult = atr_pct / 100.0
    
    for d in all_dates:
        # Step A: Attempt to Fill Pending Limit Orders from Prior Close (d)
        available_slots = max_positions - len(open_positions)
        if available_slots > 0 and len(pending_orders) > 0:
            # Sort by lowest RSI reading (deepest oversold first)
            pending_orders.sort(key=lambda x: x['ind_val'])
            
            # Current Portfolio Value for slot sizing
            current_mtm = cash
            for t_open, p_open in open_positions.items():
                if d in prepared_dfs[t_open].index:
                    px_now = prepared_dfs[t_open].loc[d]['Adj Close'] if 'Adj Close' in prepared_dfs[t_open].loc[d] else prepared_dfs[t_open].loc[d]['Close']
                    current_mtm += p_open['shares'] * px_now
                else:
                    current_mtm += p_open['shares'] * p_open['entry_price']
                    
            target_slot_size = current_mtm / max_positions
            
            filled_count = 0
            for ord_info in pending_orders:
                if filled_count >= available_slots:
                    break
                sym = ord_info['ticker']
                df_sym = prepared_dfs[sym]
                
                if d not in df_sym.index:
                    continue
                    
                row_today = df_sym.loc[d]
                o_today = row_today['Open']
                l_today = row_today['Low']
                
                # Compute limit price based on today's open
                limit_px = o_today - (atr_mult * ord_info['atr_prior'])
                
                # Check Fill Condition
                if l_today <= limit_px:
                    exec_px = min(o_today, limit_px)
                    invest_amt = min(cash, target_slot_size)
                    
                    if invest_amt >= 250.0:
                        shares = invest_amt / exec_px
                        cash -= invest_amt
                        open_positions[sym] = {
                            'entry_date': d,
                            'entry_price': exec_px,
                            'shares': shares,
                            'cost_basis': invest_amt,
                            'days_held': 0
                        }
                        filled_count += 1
                        
            # Day orders expire at the close if unfilled
            pending_orders = []
            
        # Step B: Process Exits on Open Positions (Evaluated at close of day d)
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
            
        # Step C: Generate New Pending Limit Orders for Tomorrow's Open (d+1)
        for ticker, df in prepared_dfs.items():
            if ticker in open_positions or d not in df.index:
                continue
                
            row = df.loc[d]
            curr_b = str(row[bucket_col])
            
            if curr_b not in selected_buckets:
                continue
                
            if require_uptrend and not bool(row['Above_200SMA']):
                continue
                
            if fresh_entry_only:
                prev_loc = df.index.get_loc(d) - 1
                if prev_loc >= 0:
                    prev_b = str(df.iloc[prev_loc][bucket_col])
                    if curr_b == prev_b:
                        continue
                        
            pending_orders.append({
                "ticker": ticker,
                "atr_prior": row['ATR10'],
                "ind_val": row[indicator_col]
            })
            
        # Step D: Mark-to-Market Portfolio Equity at Close
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
# 3. UNIVERSE FETCHERS & CONSTITUENTS
# ==========================================

DJIA_STATIC_FALLBACK = [
    "AAPL", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
]

RUSSELL2000_STATIC_FALLBACK = [
    "BTSG", "MOG-A", "FROG", "TWST", "GKOS", "UMBF", "SM", "PCVX", "FCFS", "EAT",
    "PBF", "CTRE", "ONB", "KRYS", "TXG", "JXN", "LTH", "KYMR", "PTGX", "KNTK",
    "CYTK", "TGTX", "IDCC", "VSAT", "ZETA", "HNGE", "RHP", "ZWS", "HQY", "PACS",
    "URBN", "SLAB", "TTAN", "CMC", "ACA", "VSXY", "ALKT", "ACLS", "ADMA", "AEO",
    "AMN", "ARWR", "CALX", "CARG", "CEIX", "CNX", "CRUS", "CVCO", "EPRT", "EXPO",
    "FABR", "FLR", "FN", "GTLS", "HAE", "HALO", "ITGR", "KNSL", "LSTR", "MEDP",
    "MRC", "NEOG", "NSIT", "NVST", "OII", "ORA", "OSCR", "POWI", "PRVA", "QTWO",
    "RBA", "SAIA", "SPSC", "STRL", "TALO", "TKR", "VVI", "WING", "WNS", "ABG"
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
        return res if res else []
    elif universe_name == "Nasdaq 100":
        res = extract_wiki("https://en.wikipedia.org/wiki/Nasdaq-100")
        return res if res else ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "AMD"]
    elif universe_name == "Dow Jones 30":
        res = extract_wiki("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        return res if res else DJIA_STATIC_FALLBACK
    elif "Russell 2000" in universe_name:
        res = extract_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
        if not res:
            res = RUSSELL2000_STATIC_FALLBACK
        else:
            res = list(dict.fromkeys(res + RUSSELL2000_STATIC_FALLBACK))
        return res
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
    "📁 Universe & Portfolio Backtester",
    "Live Market Screener"
])

# ----------------------------------------------------
# TAB 1: SINGLE TICKER STUDY & CURRENT STATUS
# ----------------------------------------------------
with tab_study:
    st.header("Single Ticker Expectancy & Limit Fill Analysis")
    st.write(
        "Evaluate full historical bucket distributions with orders placed on the **Next-Day Open** "
        "at **Open - X% ATR(10)** with fill rate tracking."
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
        
    col_t5, col_t6 = st.columns([1.5, 1.5])
    with col_t5:
        atr_limit_discount = st.number_input(
            "Next-Day Limit Discount (% of 10-Day ATR below Open)",
            min_value=0.0, max_value=100.0, value=25.0, step=5.0,
            help="0% = Market on Open. 25% = Limit order at Open - 0.25 * ATR(10). Orders only fill if Low <= Limit Price."
        )
    with col_t6:
        st.write("")
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
            curr_atr = latest_row['ATR10']
            is_above_200 = bool(latest_row['Above_200SMA'])
            
            # Metric Header
            st.subheader(f"Current Market Status: {study_ticker}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Current Price", f"${curr_price:.2f}")
            m2.metric("RSI(2) (Single Bar)", f"{curr_rsi2:.1f}")
            m3.metric(f"Current {study_indicator}", f"{curr_ind_val:.1f}")
            m4.metric("Active Bucket", curr_bucket)
            m5.metric("Current ATR(10)", f"${curr_atr:.2f}")
            
            # Simulate Limit Executions
            sim_df = simulate_single_ticker_execution(
                processed_df,
                bucket_col=active_bucket_col,
                atr_pct=atr_limit_discount,
                fresh_entry_only=fresh_only,
                require_uptrend=require_uptrend_study
            )
            
            perf_table = aggregate_bucket_performance(sim_df)
            
            # Baseline Callout
            if curr_bucket in perf_table.index and perf_table.loc[curr_bucket, "Signals"] > 0:
                c_sigs = perf_table.loc[curr_bucket, "Signals"]
                c_filled = perf_table.loc[curr_bucket, "Filled"]
                c_fr = perf_table.loc[curr_bucket, "Fill Rate (%)"]
                c_wr = perf_table.loc[curr_bucket, "5D Win Rate (%)"]
                c_ret = perf_table.loc[curr_bucket, "5D Avg Ret (%)"]
                c_sma_wr = perf_table.loc[curr_bucket, "SMA5 Exit Win Rate (%)"]
                c_sma_pf = perf_table.loc[curr_bucket, "SMA5 Profit Factor"]
                
                st.info(
                    f"**Empirical Baseline for Current Regime ({curr_bucket})**: "
                    f"Out of **{c_sigs} historical signals**, **{c_filled} orders filled** (**{c_fr}% Fill Rate**) "
                    f"using a limit at Open - {atr_limit_discount:.1f}% ATR. "
                    f"Filled trades achieved a **5-day win rate of {c_wr}%** (Avg Return: **{c_ret:+.2f}%**). "
                    f"Exiting dynamically at Close > 5-Day SMA yielded a **{c_sma_wr}% win rate** and a **{c_sma_pf} Profit Factor**."
                )
            
            st.subheader(f"Expectancy & Risk Matrix across All Buckets: {study_ticker}")
            study_col_cfg = {
                "Signals": st.column_config.NumberColumn(format="%d"),
                "Filled": st.column_config.NumberColumn(format="%d"),
                "Fill Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "5D Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "5D Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "5D Avg Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "5D Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                "10D Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "10D Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "10D Avg Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "10D Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                "SMA5 Exit Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "SMA5 Exit Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "SMA5 Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d"),
                "SMA5 Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                "RSI70 Exit Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "RSI70 Exit Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "RSI70 Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d"),
                "RSI70 Profit Factor": st.column_config.NumberColumn(format="%.2f")
            }
            st.dataframe(perf_table, column_config=study_col_cfg, use_container_width=True)
            
            # Analytics Dashboard
            st.subheader("Quantitative Edge Visualizer")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig_wr = go.Figure()
                fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['5D Win Rate (%)'], name='Fixed 5-Day', marker_color='#42A5F5'))
                fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['SMA5 Exit Win Rate (%)'], name='Dynamic (Close > SMA5)', marker_color='#26A69A'))
                fig_wr.add_trace(go.Bar(x=perf_table.index, y=perf_table['RSI70 Exit Win Rate (%)'], name='Dynamic (RSI2 > 70)', marker_color='#FFA726'))
                fig_wr.add_hline(y=50, line_dash="dash", line_color="white", annotation_text="50% Parity")
                fig_wr.update_layout(title="Win Rate (%) Across Exit Strategies", barmode='group', template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_wr, use_container_width=True)
                
            with chart_col2:
                fig_pf = go.Figure()
                fig_pf.add_trace(go.Bar(x=perf_table.index, y=perf_table['5D Profit Factor'], name='Fixed 5-Day PF', marker_color='#42A5F5'))
                fig_pf.add_trace(go.Bar(x=perf_table.index, y=perf_table['SMA5 Profit Factor'], name='SMA5 Exit PF', marker_color='#26A69A'))
                fig_pf.add_trace(go.Bar(x=perf_table.index, y=perf_table['RSI70 Profit Factor'], name='RSI70 Exit PF', marker_color='#FFA726'))
                fig_pf.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Breakeven (1.0)")
                fig_pf.update_layout(title="Profit Factor Across Buckets", barmode='group', template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_pf, use_container_width=True)

            # Year-by-Year Breakdown
            st.divider()
            st.subheader(f"📅 Year-by-Year Performance Breakdown: {study_ticker}")
            col_y1, col_y2 = st.columns([1.5, 2.5])
            with col_y1:
                year_exit_strat = st.selectbox(
                    "Evaluate Exit Strategy",
                    ["Dynamic Snapback: Close > 5-Day SMA", "Dynamic Snapback: RSI(2) > 70", "Fixed 5-Day Holding Period", "Fixed 10-Day Holding Period"],
                    index=0, key="study_year_exit"
                )
            with col_y2:
                bucket_scope_options = [f"Current Bucket Only ({curr_bucket})", "Oversold Washouts (0-20)", "Severe Capitulation (0-10)", "All Buckets (0-100)"]
                year_bucket_scope = st.selectbox("Bucket Selection Scope", bucket_scope_options, index=1, key="study_year_scope")
                
            if year_exit_strat == "Dynamic Snapback: Close > 5-Day SMA":
                target_ret_col = "SMA5_Ret"
                target_days_col = "SMA5_Days"
            elif year_exit_strat == "Dynamic Snapback: RSI(2) > 70":
                target_ret_col = "RSI70_Ret"
                target_days_col = "RSI70_Days"
            elif year_exit_strat == "Fixed 5-Day Holding Period":
                target_ret_col = "Fwd_5d"
                target_days_col = None
            else:
                target_ret_col = "Fwd_10d"
                target_days_col = None
                
            if "Current Bucket Only" in year_bucket_scope:
                chosen_buckets = [curr_bucket]
            elif "Oversold Washouts" in year_bucket_scope:
                chosen_buckets = ['0-10', '11-20']
            elif "Severe Capitulation" in year_bucket_scope:
                chosen_buckets = ['0-10']
            else:
                chosen_buckets = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
                
            yearly_table = aggregate_yearly_performance(sim_df, chosen_buckets, target_ret_col, target_days_col)
            
            if not yearly_table.empty:
                year_col_cfg = {
                    "Signals": st.column_config.NumberColumn(format="%d"),
                    "Filled": st.column_config.NumberColumn(format="%d"),
                    "Fill Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "Win Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "Avg Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                    "Annual Compounded Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d")
                }
                st.dataframe(yearly_table, column_config=year_col_cfg, use_container_width=True)
                
                chart_year_df = yearly_table[yearly_table.index != "All Years"].copy()
                if not chart_year_df.empty:
                    bar_colors = ['#00FF66' if r >= 0 else '#FF3366' for r in chart_year_df['Annual Compounded Ret (%)']]
                    fig_annual = go.Figure()
                    fig_annual.add_trace(go.Bar(
                        x=chart_year_df.index, y=chart_year_df['Annual Compounded Ret (%)'], marker_color=bar_colors,
                        hovertext=[f"Year: {y}<br>Return: {r:+.2f}%<br>Trades: {s}<br>Win Rate: {w:.1f}%" 
                                   for y, r, s, w in zip(chart_year_df.index, chart_year_df['Annual Compounded Ret (%)'], chart_year_df['Signals'], chart_year_df['Win Rate (%)'])],
                        hoverinfo='text'
                    ))
                    fig_annual.add_hline(y=0, line_dash="dash", line_color="white")
                    fig_annual.update_layout(title=f"{study_ticker} Annual Compounded Strategy Return (%)", template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_annual, use_container_width=True)

# ----------------------------------------------------
# TAB 2: CUSTOM WATCHLIST ANALYZER
# ----------------------------------------------------
with tab_watchlist:
    st.header("Custom Watchlist Analyzer & Pending Orders")
    st.write("Scan personal watchlist tickers for current RSI readings, ATR(10) levels, and calculate tomorrow's limit order formula.")
    
    col_w1, col_w2, col_w3 = st.columns([2.5, 1.2, 1])
    with col_w1:
        watchlist_input = st.text_input("Enter Ticker Symbols (comma or space separated)", value="SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMZN, IWM", key="watch_input")
    with col_w2:
        watch_lookback_choice = st.selectbox("Primary Regime Indicator", ["5D Avg RSI(2)", "3D Avg RSI(2)", "2D Avg RSI(2)", "10D Avg RSI(2)"], index=0, key="watch_lookback")
    with col_w3:
        watch_hist_years = st.number_input("Historical Depth (Years)", min_value=2, max_value=20, value=5, key="watch_hist_y")
        
    col_w4, col_w5 = st.columns([1.5, 1.5])
    with col_w4:
        watch_atr_discount = st.number_input("Watchlist ATR Discount (%)", min_value=0.0, max_value=100.0, value=25.0, step=5.0, key="watch_atr_disc")
    with col_w5:
        st.write("")
        watch_uptrend_only = st.checkbox("Require Historical Signals to be Above 200 SMA", value=True, key="watch_trend")
        
    run_watch = st.button("Analyze Custom Watchlist", type="primary", use_container_width=True)
    
    if run_watch:
        watch_tickers = [s.strip().upper().replace('.', '-') for s in re.split(r'[, \n]+', watchlist_input) if s.strip()]
        
        if not watch_tickers:
            st.warning("Please enter at least one valid ticker symbol.")
        else:
            with st.spinner(f"Fetching daily data for {len(watch_tickers)} stocks..."):
                start_w = (datetime.now() - timedelta(days=watch_hist_years * 365)).strftime('%Y-%m-%d')
                end_w = datetime.now().strftime('%Y-%m-%d')
                bulk_w = yf.download(watch_tickers, start=start_w, end=end_w, group_by="ticker", auto_adjust=False, progress=False, threads=True)
                
            w_bucket_map = {"2D Avg RSI(2)": "Bucket_2d", "3D Avg RSI(2)": "Bucket_3d", "5D Avg RSI(2)": "Bucket_5d", "10D Avg RSI(2)": "Bucket_10d"}
            active_w_bucket_col = w_bucket_map[watch_lookback_choice]
            
            watchlist_results = []
            for sym in watch_tickers:
                try:
                    df_sym = extract_symbol_daily(bulk_w, sym, len(watch_tickers))
                    if len(df_sym) < 220:
                        continue
                    prep_df = prepare_stock_data(df_sym)
                    latest = prep_df.iloc[-1]
                    
                    curr_p = latest['Adj Close'] if 'Adj Close' in latest else latest['Close']
                    curr_b = str(latest[active_w_bucket_col])
                    curr_atr = latest['ATR10']
                    is_bull = bool(latest['Above_200SMA'])
                    
                    limit_dollar_discount = (watch_atr_discount / 100.0) * curr_atr
                    
                    sim_sub = simulate_single_ticker_execution(
                        prep_df, bucket_col=active_w_bucket_col, atr_pct=watch_atr_discount,
                        fresh_entry_only=True, require_uptrend=watch_uptrend_only
                    )
                    
                    b_sub = sim_sub[sim_sub['Bucket'] == curr_b]
                    sigs = len(b_sub)
                    filled_b = b_sub[b_sub['Filled'] == True]
                    n_filled = len(filled_b)
                    
                    if n_filled > 0:
                        fr = n_filled / sigs * 100
                        wr5 = (filled_b['Fwd_5d'] > 0).mean() * 100
                        ret5 = filled_b['Fwd_5d'].mean()
                        gains = filled_b[filled_b['Fwd_5d'] > 0]['Fwd_5d'].sum()
                        losses = abs(filled_b[filled_b['Fwd_5d'] < 0]['Fwd_5d'].sum())
                        pf = round(gains / losses, 2) if losses > 0 else (np.nan if gains == 0 else 99.0)
                        sma_wr = (filled_b['SMA5_Ret'] > 0).mean() * 100
                    else:
                        fr, wr5, ret5, pf, sma_wr = 0.0, np.nan, np.nan, np.nan, np.nan
                        
                    watchlist_results.append({
                        "Ticker": sym, "Price": round(curr_p, 2), "RSI(2)": round(latest['RSI2'], 1),
                        "5D Avg": round(latest['RSI2_5d_avg'], 1), "Active Bucket": curr_b,
                        "ATR(10)": round(curr_atr, 2),
                        "Tomorrow Limit Offset": f"-${limit_dollar_discount:.2f} off Open",
                        "> 200 SMA": "Yes" if is_bull else "No",
                        "Hist Signals": sigs,
                        "Hist Fill Rate": round(fr, 1) if pd.notna(fr) else 0.0,
                        "Hist 5D Win Rate": round(wr5, 1) if pd.notna(wr5) else np.nan,
                        "Hist 5D Avg Ret (%)": round(ret5, 2) if pd.notna(ret5) else np.nan,
                        "Hist Profit Factor": pf,
                        "Hist SMA5 Exit Win Rate": round(sma_wr, 1) if pd.notna(sma_wr) else np.nan
                    })
                except Exception:
                    pass
                    
            w_df = pd.DataFrame(watchlist_results)
            if not w_df.empty:
                st.subheader(f"Watchlist Diagnostics ({len(w_df)} Tickers Analyzed)")
                w_df = w_df.sort_values(by="5D Avg", ascending=True)
                w_cfg = {
                    "Ticker": st.column_config.TextColumn(), "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "RSI(2)": st.column_config.NumberColumn(format="%.1f"), "5D Avg": st.column_config.NumberColumn(format="%.1f"),
                    "Active Bucket": st.column_config.TextColumn(), "ATR(10)": st.column_config.NumberColumn(format="$%.2f"),
                    "Tomorrow Limit Offset": st.column_config.TextColumn(help="Exact discount subtracted from tomorrow's market open price."),
                    "> 200 SMA": st.column_config.TextColumn(), "Hist Signals": st.column_config.NumberColumn(format="%d"),
                    "Hist Fill Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "Hist 5D Win Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "Hist 5D Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Hist Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                    "Hist SMA5 Exit Win Rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)
                }
                st.dataframe(w_df, column_config=w_cfg, use_container_width=True)
                st.download_button(label="Download Watchlist CSV", data=w_df.to_csv(index=False).encode('utf-8'), file_name="watchlist_analysis.csv", mime='text/csv')

# ----------------------------------------------------
# TAB 3: UNIVERSE & PORTFOLIO BACKTESTER
# ----------------------------------------------------
with tab_portfolio:
    st.header("📁 Universe & Multi-Stock Portfolio Backtester")
    st.write(
        "Simulate an active portfolio backtest across an entire universe or custom basket. "
        "Orders are submitted for the **Next-Day Open** with a limit discount of **Open - X% ATR(10)**."
    )
    
    col_mode1, col_mode2 = st.columns([1.5, 2.5])
    with col_mode1:
        portfolio_mode = st.radio(
            "Portfolio Source",
            ["Universe Benchmark (Entire Roster)", "Custom Ticker Basket (Manual)"],
            index=0, key="port_source_mode"
        )
        
    if portfolio_mode == "Universe Benchmark (Entire Roster)":
        col_u1, col_u2 = st.columns([1.5, 1.5])
        with col_u1:
            port_universe_choice = st.selectbox(
                "Benchmark Universe",
                ["Russell 2000 (Small Caps)", "S&P 500", "Nasdaq 100", "Dow Jones 30", "Major ETFs"],
                index=0, key="port_univ_sel"
            )
        with col_u2:
            port_pool_size = st.selectbox(
                "Universe Candidate Pool Size",
                ["Top 25 Liquid (Fastest & Stable)", "Top 50 Liquid (Balanced)", "Top 100 Liquid (Deep)", "Full Universe Roster"],
                index=0
            )
    else:
        port_tickers_input = st.text_input(
            "Portfolio Tickers (comma or space separated)",
            value="SPY, QQQ, AAPL, MSFT, NVDA, AMZN, TSLA, GOOGL, META, IWM",
            key="port_tickers_input"
        )
        
    col_p2, col_p3, col_p4 = st.columns([1.2, 1.2, 1.2])
    with col_p2:
        port_indicator_choice = st.selectbox(
            "Regime Indicator",
            ["5D Avg RSI(2)", "3D Avg RSI(2)", "2D Avg RSI(2)", "10D Avg RSI(2)"],
            index=0, key="port_ind_choice"
        )
    with col_p3:
        port_years = st.number_input("Lookback Scope (Years)", min_value=2, max_value=25, value=10, key="port_years")
    with col_p4:
        port_exit_mode = st.selectbox(
            "Exit Strategy",
            ["Close > SMA5", "RSI(2) > 70", "Fixed 5-Day Hold", "Fixed 10-Day Hold"],
            index=0, key="port_exit_mode"
        )
        
    col_p5, col_p6, col_p7 = st.columns([1.2, 1.2, 1.2])
    with col_p5:
        port_atr_discount = st.number_input("Next-Day ATR Limit Discount (%)", min_value=0.0, max_value=100.0, value=25.0, step=5.0, key="port_atr_disc")
    with col_p6:
        port_bucket_scope = st.selectbox("Target Entry Buckets", ["Oversold Washouts (0-20)", "Severe Capitulation (0-10)", "All Buckets (0-100)"], index=0, key="port_bucket_scope")
    with col_p7:
        port_max_pos = st.number_input("Max Concurrent Position Slots", min_value=1, max_value=20, value=5, key="port_max_pos")
        
    col_p8, col_p9, col_p10 = st.columns([1.2, 1.2, 1.2])
    with col_p8:
        port_init_cash = st.number_input("Starting Capital ($)", min_value=5000, value=100000, step=10000, key="port_init_cash")
    with col_p9:
        st.write("")
        port_require_uptrend = st.checkbox("Require Trend Filter (Close > 200 SMA)", value=True, key="port_trend")
    with col_p10:
        st.write("")
        port_fresh_only = st.checkbox("Fresh Entry Only (Do not re-enter if stock stayed in bucket)", value=True, key="port_fresh")
        
    run_port_backtest = st.button("🚀 Run Portfolio Backtest", type="primary", use_container_width=True)
    
    if run_port_backtest:
        if portfolio_mode == "Universe Benchmark (Entire Roster)":
            with st.spinner(f"Loading roster for {port_universe_choice}..."):
                full_roster = get_universe_tickers(port_universe_choice)
                
            if "Top 25" in port_pool_size:
                port_tickers = full_roster[:25]
            elif "Top 50" in port_pool_size:
                port_tickers = full_roster[:50]
            elif "Top 100" in port_pool_size:
                port_tickers = full_roster[:100]
            else:
                port_tickers = full_roster
        else:
            port_tickers = [s.strip().upper().replace('.', '-') for s in re.split(r'[, \n]+', port_tickers_input) if s.strip()]
            
        if not port_tickers:
            st.warning("Please provide at least one ticker symbol or select a valid universe.")
        else:
            with st.spinner(f"Downloading historical daily data for {len(port_tickers)} tickers over {port_years} years..."):
                start_p = (datetime.now() - timedelta(days=port_years * 365)).strftime('%Y-%m-%d')
                end_p = datetime.now().strftime('%Y-%m-%d')
                bulk_port = yf.download(
                    port_tickers,
                    start=start_p,
                    end=end_p,
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True
                )
                
            prepared_port_dict = {}
            for sym in port_tickers:
                df_sym = extract_symbol_daily(bulk_port, sym, len(port_tickers))
                if len(df_sym) >= 220:
                    prep = prepare_stock_data(df_sym)
                    prepared_port_dict[sym] = prep
                    
            if not prepared_port_dict:
                st.error("Could not obtain sufficient trading history for the requested tickers.")
            else:
                p_bucket_map = {"2D Avg RSI(2)": "Bucket_2d", "3D Avg RSI(2)": "Bucket_3d", "5D Avg RSI(2)": "Bucket_5d", "10D Avg RSI(2)": "Bucket_10d"}
                p_ind_map = {"2D Avg RSI(2)": "RSI2_2d_avg", "3D Avg RSI(2)": "RSI2_3d_avg", "5D Avg RSI(2)": "RSI2_5d_avg", "10D Avg RSI(2)": "RSI2_10d_avg"}
                
                active_p_bucket_col = p_bucket_map[port_indicator_choice]
                active_p_ind_col = p_ind_map[port_indicator_choice]
                
                if "Oversold Washouts" in port_bucket_scope:
                    selected_p_buckets = ['0-10', '11-20']
                elif "Severe Capitulation" in port_bucket_scope:
                    selected_p_buckets = ['0-10']
                else:
                    selected_p_buckets = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
                    
                port_year_df, port_equity_df, port_trades_df = run_portfolio_backtest(
                    prepared_dfs=prepared_port_dict,
                    selected_buckets=selected_p_buckets,
                    bucket_col=active_p_bucket_col,
                    indicator_col=active_p_ind_col,
                    atr_pct=port_atr_discount,
                    exit_mode=port_exit_mode,
                    max_positions=int(port_max_pos),
                    initial_capital=float(port_init_cash),
                    require_uptrend=port_require_uptrend,
                    fresh_entry_only=port_fresh_only
                )
                
                st.subheader("Portfolio Performance Summary")
                overall_row = port_year_df.loc["All Years"]
                
                c_m1, c_m2, c_m3, c_m4, c_m5, c_m6 = st.columns(6)
                c_m1.metric("Compounded Return", f"{overall_row['Annual Portfolio Ret (%)']:+,.2f}%")
                c_m2.metric("Total Trades Filled", f"{int(overall_row['Trades'])}")
                c_m3.metric("Win Rate", f"{overall_row['Win Rate (%)']:.1f}%")
                c_m4.metric("Avg Trade Return", f"{overall_row['Avg Trade Ret (%)']:+.2f}%")
                c_m5.metric("Profit Factor", f"{overall_row['Profit Factor']}")
                c_m6.metric("Max Portfolio DD", f"{overall_row['Max Drawdown (%)']:.1f}%")
                
                st.subheader("Audited Year-by-Year Results")
                port_year_cfg = {
                    "Trades": st.column_config.NumberColumn(format="%d"),
                    "Win Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "Avg Trade Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                    "Annual Portfolio Ret (%)": st.column_config.NumberColumn(format="%+.2f%%", help="True portfolio equity return from Jan 1 to Dec 31."),
                    "Max Drawdown (%)": st.column_config.NumberColumn(format="%.1f%%", help="Worst peak-to-trough drawdown during that calendar year."),
                    "Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d")
                }
                st.dataframe(port_year_df, column_config=port_year_cfg, use_container_width=True)
                
                col_chart_p1, col_chart_p2 = st.columns(2)
                with col_chart_p1:
                    chart_yr_df = port_year_df[port_year_df.index != "All Years"].copy()
                    if not chart_yr_df.empty:
                        bar_colors = ['#00FF66' if r >= 0 else '#FF3366' for r in chart_yr_df['Annual Portfolio Ret (%)']]
                        fig_port_annual = go.Figure()
                        fig_port_annual.add_trace(go.Bar(
                            x=chart_yr_df.index,
                            y=chart_yr_df['Annual Portfolio Ret (%)'],
                            marker_color=bar_colors,
                            hovertext=[f"Year: {y}<br>Return: {r:+.2f}%<br>Trades: {t}<br>Win Rate: {w:.1f}%" 
                                       for y, r, t, w in zip(chart_yr_df.index, chart_yr_df['Annual Portfolio Ret (%)'], chart_yr_df['Trades'], chart_yr_df['Win Rate (%)'])],
                            hoverinfo='text'
                        ))
                        fig_port_annual.add_hline(y=0, line_dash="dash", line_color="white")
                        fig_port_annual.update_layout(title="Annual Portfolio Return (%) by Calendar Year", template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_port_annual, use_container_width=True)
                        
                with col_chart_p2:
                    fig_port_equity = go.Figure()
                    fig_port_equity.add_trace(go.Scatter(
                        x=port_equity_df.index, y=port_equity_df['Equity'], mode='lines', name='Portfolio Value', line=dict(color='#00FF66', width=2)
                    ))
                    fig_port_equity.update_layout(title="Portfolio Equity Compounding ($)", template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_port_equity, use_container_width=True)
                    
                st.subheader(f"Trade Execution Audit Log ({len(port_trades_df)} Closed Trades)")
                if not port_trades_df.empty:
                    st.download_button(
                        label="Download Portfolio Trade Log as CSV",
                        data=port_trades_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"portfolio_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv'
                    )
                    port_trade_cfg = {
                        "Ticker": st.column_config.TextColumn(),
                        "Entry Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Exit Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                        "Days Held": st.column_config.NumberColumn(format="%d d"),
                        "Result": st.column_config.TextColumn()
                    }
                    st.dataframe(port_trades_df.sort_values(by="Exit Date", ascending=False), column_config=port_trade_cfg, use_container_width=True)

# ----------------------------------------------------
# TAB 4: LIVE MARKET SCREENER
# ----------------------------------------------------
with tab_screener:
    st.header("Live Cumulative RSI(2) Screener")
    st.write("Screens constituents for current RSI(2) regimes, checks 200-day trend qualification, and synchronizes signals with broad S&P 500 macro breadth.")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns([1.1, 1.2, 1.3, 0.8])
    with col_s1:
        universe_choice = st.selectbox(
            "Screener Universe",
            ["Russell 2000 (Small Caps)", "S&P 500", "Nasdaq 100", "Dow Jones 30", "Major ETFs"],
            index=0, key="screen_univ"
        )
    with col_s2:
        screener_lookback = st.selectbox("Filter By Indicator", ["5D Avg RSI(2)", "3D Avg RSI(2)", "2D Avg RSI(2)", "10D Avg RSI(2)"], index=0, key="screen_ind")
    with col_s3:
        bucket_filter = st.multiselect("Filter Target Buckets", options=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'], default=['0-10', '11-20'], key="screen_buckets")
    with col_s4:
        st.write("")
        st.write("")
        run_screener = st.button("Run Live Screen", type="primary", use_container_width=True)
        
    require_uptrend_screener = st.checkbox("Enforce 200-Day SMA Trend Gate (Only show stocks trading above 200-Day SMA)", value=True, key="screen_trend")
        
    if run_screener:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Checking broad market S&P 500 (SPY) breadth...")
        spy_raw = yf.download("SPY", period="6mo", interval="1d", auto_adjust=False, progress=False)
        spy_prep = prepare_stock_data(spy_raw)
        latest_spy_rsi5 = spy_prep['RSI2_5d_avg'].iloc[-1]
        
        if latest_spy_rsi5 <= 25:
            macro_tag = "Systemic Flush (Broad Market Oversold - High Edge)"
            macro_desc = "Broad market capitulation detected (SPY 5D Avg RSI <= 25). Historical recovery odds are elevated."
            macro_color = "success"
        else:
            macro_tag = "Idiosyncratic Dip (Selective / Stock-Specific)"
            macro_desc = f"Broad market is neutral/bullish (SPY 5D Avg RSI = {latest_spy_rsi5:.1f}). Dips are stock-specific."
            macro_color = "info"
            
        progress_bar.progress(15)
        status_text.text(f"Fetching constituents for {universe_choice}...")
        tickers = get_universe_tickers(universe_choice)
        progress_bar.progress(30)
        
        status_text.text(f"Downloading daily market data for {len(tickers)} stocks...")
        bulk_data = yf.download(tickers, period="2y", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        progress_bar.progress(70)
        
        status_text.text("Evaluating indicators, MAE drawdowns, and dynamic snapbacks...")
        screener_rows = []
        lookback_key_map = {"2D Avg RSI(2)": "Bucket_2d", "3D Avg RSI(2)": "Bucket_3d", "5D Avg RSI(2)": "Bucket_5d", "10D Avg RSI(2)": "Bucket_10d"}
        active_filter_bucket = lookback_key_map[screener_lookback]
        
        for sym in tickers:
            try:
                df_sym = extract_symbol_daily(bulk_data, sym, len(tickers))
                if len(df_sym) < 220:
                    continue
                processed = prepare_stock_data(df_sym)
                latest = processed.iloc[-1]
                
                is_above_200 = bool(latest['Above_200SMA'])
                if require_uptrend_screener and not is_above_200:
                    continue
                    
                target_b = str(latest[active_filter_bucket])
                if bucket_filter and target_b not in bucket_filter:
                    continue
                    
                c_price = latest['Adj Close'] if 'Adj Close' in latest else latest['Close']
                atr10 = latest['ATR10']
                
                screener_rows.append({
                    "Ticker": sym, "Current Price": round(c_price, 2), "RSI(2)": round(latest['RSI2'], 1),
                    "2D Avg": round(latest['RSI2_2d_avg'], 1), "3D Avg": round(latest['RSI2_3d_avg'], 1),
                    "5D Avg": round(latest['RSI2_5d_avg'], 1), "10D Avg": round(latest['RSI2_10d_avg'], 1),
                    "Active Bucket": target_b, "ATR(10)": round(atr10, 2),
                    "Tomorrow Limit Target (25% ATR)": f"Open - ${0.25 * atr10:.2f}",
                    "Above 200 SMA": "Yes" if is_above_200 else "No",
                    "Market Context": "Systemic" if latest_spy_rsi5 <= 25 else "Idiosyncratic",
                    "Prior 5D Ret (%)": round(latest['Prior_5d'], 2), "Prior 10D Ret (%)": round(latest['Prior_10d'], 2)
                })
            except Exception:
                pass
                
        progress_bar.progress(100)
        status_text.empty()
        
        if macro_color == "success":
            st.success(f"Macro Breadth Regime: {macro_tag} | {macro_desc}")
        else:
            st.info(f"Macro Breadth Regime: {macro_tag} | {macro_desc}")
        
        screen_df = pd.DataFrame(screener_rows)
        if not screen_df.empty:
            st.subheader(f"Filtered Setups: {len(screen_df)} Opportunities Found")
            screen_df = screen_df.sort_values(by="5D Avg", ascending=True)
            screener_col_cfg = {
                "Ticker": st.column_config.TextColumn(), "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                "RSI(2)": st.column_config.NumberColumn(format="%.1f"), "2D Avg": st.column_config.NumberColumn(format="%.1f"),
                "3D Avg": st.column_config.NumberColumn(format="%.1f"), "5D Avg": st.column_config.NumberColumn(format="%.1f"),
                "10D Avg": st.column_config.NumberColumn(format="%.1f"), "Active Bucket": st.column_config.TextColumn(),
                "ATR(10)": st.column_config.NumberColumn(format="$%.2f"),
                "Tomorrow Limit Target (25% ATR)": st.column_config.TextColumn(help="Suggested limit entry offset from tomorrow's market open."),
                "Above 200 SMA": st.column_config.TextColumn(), "Market Context": st.column_config.TextColumn(),
                "Prior 5D Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"), "Prior 10D Ret (%)": st.column_config.NumberColumn(format="%+.2f%%")
            }
            st.dataframe(screen_df, column_config=screener_col_cfg, use_container_width=True)
            st.download_button(label="Download Screener CSV", data=screen_df.to_csv(index=False).encode('utf-8'), file_name="rsi2_screener.csv", mime='text/csv')
        else:
            st.info(f"No stocks in {universe_choice} currently match the selected criteria with trend gates active.")
