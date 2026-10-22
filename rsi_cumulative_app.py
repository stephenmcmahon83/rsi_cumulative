import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import re

st.set_page_config(page_title="RSI(2) Next-Open Backtester", layout="wide")

st.title("RSI(2) Quantitative Backtester (Next-Open Execution)")
st.caption(
    "Execution Model: Entry Signal on Close → Buy Next Day Open | "
    "Exit Signal on Close → Sell Next Day Open"
)

# ==========================================
# CONSTANTS & STRATEGY DEFINITIONS
# ==========================================

BUCKET_LABELS = [
    '<10', '11-20', '21-30', '31-40', '41-50',
    '51-60', '61-70', '71-80', '81-90', '91-100'
]
RSI_THRESHOLDS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
HIGH_CLOSE_WINDOWS = [3, 5, 7, 10]

EXIT_KEYS = (
    ["SMA5"]
    + [f"HC{x}" for x in HIGH_CLOSE_WINDOWS]
    + [f"RSI_{t}" for t in RSI_THRESHOLDS]
    + ["FIX5", "FIX10"]
)

RSI_DAYS_MAP = {"2-Day RSI(2)": 2, "3-Day RSI(2)": 3, "5-Day RSI(2)": 5}

def get_exit_label_map(rsi_name="5-Day RSI(2)"):
    """Returns human-readable exit labels dynamically matched to the chosen RSI lookback."""
    mapping = {
        "SMA5": "Close > 5-Day SMA",
        "HC3": "Close at 3-Day High Close",
        "HC5": "Close at 5-Day High Close",
        "HC7": "Close at 7-Day High Close",
        "HC10": "Close at 10-Day High Close",
    }
    for t in RSI_THRESHOLDS:
        mapping[f"RSI_{t}"] = f"{rsi_name} > {t}"
    mapping["FIX5"] = "Fixed 5-Day Hold"
    mapping["FIX10"] = "Fixed 10-Day Hold"
    return mapping

# ==========================================
# 1. OHLCV EXTRACTION & CORE QUANT ENGINE
# ==========================================

OHLCV_NAMES = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}

def normalize_ohlcv_df(df):
    """
    Safely flattens any yfinance DataFrame (single or multi-ticker, group_by='ticker' or 'column')
    so that columns are guaranteed to be standard ['Open', 'High', 'Low', 'Close', ...].
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        if len(lvl0.intersection(OHLCV_NAMES)) >= 3:
            df.columns = df.columns.get_level_values(0)
        elif len(lvl1.intersection(OHLCV_NAMES)) >= 3:
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = [c[0] for c in df.columns]

    # Remove any duplicate columns and drop empty rows
    df = df.loc[:, ~df.columns.duplicated()].dropna(how='all')
    required = {'Open', 'High', 'Low', 'Close'}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    return df

def extract_symbol_daily(bulk_data, sym, total_tickers=1):
    """Robustly extracts single-symbol daily OHLCV from bulk or single yfinance downloads."""
    if bulk_data is None or bulk_data.empty:
        return pd.DataFrame()
    try:
        if isinstance(bulk_data.columns, pd.MultiIndex):
            lvl0 = bulk_data.columns.get_level_values(0)
            lvl1 = bulk_data.columns.get_level_values(1)
            if sym in lvl0:
                sub = bulk_data[sym].copy()
                return normalize_ohlcv_df(sub)
            elif sym in lvl1:
                sub = bulk_data.xs(sym, axis=1, level=1).copy()
                return normalize_ohlcv_df(sub)
            elif total_tickers == 1:
                return normalize_ohlcv_df(bulk_data)
        else:
            if total_tickers == 1:
                return normalize_ohlcv_df(bulk_data)
    except Exception:
        pass
    return pd.DataFrame()

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

def prepare_stock_data(raw_df, rsi_days=5, max_hold=20):
    """
    Computes adjusted OHLC, N-day RSI(2) buckets, and Next-Open forward trade outcomes
    for all exit strategies.
    """
    df = normalize_ohlcv_df(raw_df)
    if df.empty:
        return pd.DataFrame()

    # Normalize OHLC for dividends/splits so Open and Close are on the exact same scale
    if 'Adj Close' in df.columns and 'Close' in df.columns:
        adj_ratio = (df['Adj Close'] / df['Close']).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        df['Open_Adj'] = df['Open'] * adj_ratio
        df['High_Adj'] = df['High'] * adj_ratio
        df['Low_Adj'] = df['Low'] * adj_ratio
        df['Close_Adj'] = df['Adj Close']
    else:
        df['Open_Adj'] = df['Open']
        df['High_Adj'] = df['High']
        df['Low_Adj'] = df['Low']
        df['Close_Adj'] = df['Close']

    open_s = df['Open_Adj']
    low_s = df['Low_Adj']
    close_s = df['Close_Adj']

    # 1. Single-Bar and Multi-Day RSI(2)
    df['RSI2_1d'] = calculate_wilders_rsi(close_s, period=2)
    df['RSI2_2d'] = df['RSI2_1d'].rolling(window=2).mean()
    df['RSI2_3d'] = df['RSI2_1d'].rolling(window=3).mean()
    df['RSI2_5d'] = df['RSI2_1d'].rolling(window=5).mean()

    active_rsi_col = f'RSI2_{rsi_days}d'
    df['Active_RSI'] = df[active_rsi_col]

    # 2. Moving Averages & X-Day High Closes
    df['SMA_5'] = close_s.rolling(window=5).mean()
    df['SMA_200'] = close_s.rolling(window=200).mean()
    df['Above_200SMA'] = close_s > df['SMA_200']

    for x in HIGH_CLOSE_WINDOWS:
        df[f'HighClose_{x}d'] = close_s.rolling(window=x).max()

    # 3. Trailing Returns for Context
    df['Prior_5d'] = (close_s - close_s.shift(5)) / close_s.shift(5) * 100
    df['Prior_10d'] = (close_s - close_s.shift(10)) / close_s.shift(10) * 100

    # 4. Discrete 10-Point Buckets (<10, 11-20, ..., 91-100)
    bins = [-0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.001]
    df['Bucket'] = pd.cut(df['Active_RSI'], bins=bins, labels=BUCKET_LABELS)

    # 5. Next-Open Entry & Next-Open Exit Simulation
    # Signal at Close[i] -> Enter at Open[i+1]
    # Exit Signal at Close[i+k] (k >= 1) -> Exit at Open[i+k+1]
    n = len(df)
    o_vals = open_s.values
    l_vals = low_s.values
    c_vals = close_s.values
    sma5_vals = df['SMA_5'].values
    rsi_vals = df['Active_RSI'].values
    hc_vals = {x: df[f'HighClose_{x}d'].values for x in HIGH_CLOSE_WINDOWS}

    ret_arrays = {k: np.full(n, np.nan) for k in EXIT_KEYS}
    day_arrays = {k: np.full(n, np.nan) for k in EXIT_KEYS}
    mae_arrays = {k: np.full(n, np.nan) for k in EXIT_KEYS}
    exit_idx_arrays = {k: np.full(n, -1, dtype=int) for k in EXIT_KEYS}

    for i in range(n - 2):
        ent_idx = i + 1
        entry_px = o_vals[ent_idx]
        if np.isnan(entry_px) or entry_px <= 0:
            continue

        max_k = min(max_hold, n - 2 - i)
        if max_k < 1:
            continue

        def record_exit(key, k_steps):
            ex_idx = i + k_steps + 1
            exit_px = o_vals[ex_idx]
            ret_arrays[key][i] = (exit_px - entry_px) / entry_px * 100.0
            day_arrays[key][i] = k_steps
            exit_idx_arrays[key][i] = ex_idx
            worst_low = min(np.min(l_vals[ent_idx:ex_idx]), exit_px)
            mae_arrays[key][i] = min(0.0, (worst_low - entry_px) / entry_px * 100.0)

        # A. Close > 5-Day SMA
        for k in range(1, max_k + 1):
            if c_vals[i + k] > sma5_vals[i + k]:
                record_exit("SMA5", k)
                break
        else:
            if i + max_hold + 1 < n:
                record_exit("SMA5", max_hold)

        # B. Close at X-Day High Close
        for x in HIGH_CLOSE_WINDOWS:
            key = f"HC{x}"
            hc_arr = hc_vals[x]
            for k in range(1, max_k + 1):
                if c_vals[i + k] >= hc_arr[i + k]:
                    record_exit(key, k)
                    break
            else:
                if i + max_hold + 1 < n:
                    record_exit(key, max_hold)

        # C. Selected N-Day RSI(2) > Threshold (10, 20, ..., 90)
        untriggered = set(RSI_THRESHOLDS)
        for k in range(1, max_k + 1):
            r_val = rsi_vals[i + k]
            for t in list(untriggered):
                if r_val > t:
                    record_exit(f"RSI_{t}", k)
                    untriggered.remove(t)
            if not untriggered:
                break
        if i + max_hold + 1 < n:
            for t in untriggered:
                record_exit(f"RSI_{t}", max_hold)

        # D. Fixed 5-Day & 10-Day Holds (Open[i+1] to Open[i+1+M])
        if i + 6 < n:
            record_exit("FIX5", 5)
        if i + 11 < n:
            record_exit("FIX10", 10)

    new_cols = {}
    for k in EXIT_KEYS:
        new_cols[f'Ret_{k}'] = ret_arrays[k]
        new_cols[f'Days_{k}'] = day_arrays[k]
        new_cols[f'MAE_{k}'] = mae_arrays[k]
        new_cols[f'ExitIdx_{k}'] = exit_idx_arrays[k]

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df

def filter_signal_rows(df, require_uptrend=True, fresh_only=False):
    """Applies trend and fresh-entry filters to valid signal rows."""
    data = df.dropna(subset=['Bucket']).copy()
    if require_uptrend:
        data = data[data['Above_200SMA'] == True]
    if fresh_only:
        data = data[data['Bucket'] != data['Bucket'].shift(1)]
    return data

def summarize_series(ret_s, days_s, mae_s):
    """Computes clean summary statistics for a return series."""
    valid = ret_s.dropna()
    n = len(valid)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    wr = (valid > 0).mean() * 100.0
    avg_ret = valid.mean()
    gains = valid[valid > 0].sum()
    losses = abs(valid[valid < 0].sum())
    pf = round(gains / losses, 2) if losses > 0 else (99.0 if gains > 0 else 0.0)
    avg_days = days_s.loc[valid.index].mean()
    avg_mae = mae_s.loc[valid.index].mean()
    return int(n), round(wr, 1), round(avg_ret, 2), pf, round(avg_days, 1), round(avg_mae, 2)

def build_bucket_table(filtered_df, exit_key):
    """Table 1: Performance across all 10 Entry Buckets for a single selected Exit Rule."""
    rows = []
    for b in BUCKET_LABELS:
        sub = filtered_df[filtered_df['Bucket'] == b]
        n, wr, avg_r, pf, days, mae = summarize_series(
            sub[f'Ret_{exit_key}'], sub[f'Days_{exit_key}'], sub[f'MAE_{exit_key}']
        )
        rows.append({
            "Entry Bucket": b,
            "Trades": n,
            "Win Rate (%)": wr,
            "Avg Return (%)": avg_r,
            "Profit Factor": pf,
            "Avg Hold (Days)": days,
            "Avg Drawdown (%)": mae
        })
    return pd.DataFrame(rows).set_index("Entry Bucket")

def build_exit_comparison_table(filtered_df, selected_buckets, exit_label_map):
    """Table 2: Compares all Exit Rules as rows for the selected Entry Bucket(s)."""
    sub = filtered_df[filtered_df['Bucket'].astype(str).isin(selected_buckets)]
    rows = []
    for k in EXIT_KEYS:
        n, wr, avg_r, pf, days, mae = summarize_series(
            sub[f'Ret_{k}'], sub[f'Days_{k}'], sub[f'MAE_{k}']
        )
        rows.append({
            "Exit Strategy (Next Open)": exit_label_map[k],
            "Trades": n,
            "Win Rate (%)": wr,
            "Avg Return (%)": avg_r,
            "Profit Factor": pf,
            "Avg Hold (Days)": days,
            "Avg Drawdown (%)": mae
        })
    return pd.DataFrame(rows).set_index("Exit Strategy (Next Open)")

def build_entry_exit_matrix(filtered_df, metric_choice, exit_label_map):
    """Builds a clean 2D matrix (Rows = Entry Buckets, Columns = Exit Rules) for a single chosen metric."""
    rows = []
    for b in BUCKET_LABELS:
        sub = filtered_df[filtered_df['Bucket'] == b]
        row_dict = {"Entry Bucket": b, "Trades": int(sub['Ret_SMA5'].dropna().shape[0])}
        for k in EXIT_KEYS:
            n, wr, avg_r, pf, days, _ = summarize_series(
                sub[f'Ret_{k}'], sub[f'Days_{k}'], sub[f'MAE_{k}']
            )
            col_name = exit_label_map[k]
            if n == 0:
                row_dict[col_name] = np.nan
            elif metric_choice == "Avg Return (%)":
                row_dict[col_name] = avg_r
            elif metric_choice == "Win Rate (%)":
                row_dict[col_name] = wr
            elif metric_choice == "Profit Factor":
                row_dict[col_name] = pf
            else:
                row_dict[col_name] = days
        rows.append(row_dict)
    return pd.DataFrame(rows).set_index("Entry Bucket")

def build_yearly_and_trades(full_df, filtered_df, selected_buckets, exit_key):
    """Builds Year-by-Year table and individual trade audit log with exact Next-Open dates/prices."""
    sub = filtered_df[filtered_df['Bucket'].astype(str).isin(selected_buckets)].dropna(subset=[f'Ret_{exit_key}']).copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = full_df.index
    opens = full_df['Open_Adj'].values

    trade_rows = []
    for sig_date, row in sub.iterrows():
        sig_loc = full_df.index.get_loc(sig_date)
        ent_loc = sig_loc + 1
        ex_loc = int(row[f'ExitIdx_{exit_key}'])
        if ex_loc <= ent_loc or ex_loc >= len(full_df):
            continue

        trade_rows.append({
            "Signal Date (Close)": sig_date.strftime('%Y-%m-%d'),
            "Entry Bucket": str(row['Bucket']),
            "Signal RSI": round(row['Active_RSI'], 1),
            "Entry Date (Open)": dates[ent_loc].strftime('%Y-%m-%d'),
            "Entry Open ($)": round(opens[ent_loc], 2),
            "Exit Signal (Close)": dates[ex_loc - 1].strftime('%Y-%m-%d'),
            "Exit Date (Open)": dates[ex_loc].strftime('%Y-%m-%d'),
            "Exit Open ($)": round(opens[ex_loc], 2),
            "Return (%)": round(row[f'Ret_{exit_key}'], 2),
            "Hold (Days)": int(row[f'Days_{exit_key}']),
            "Drawdown (%)": round(row[f'MAE_{exit_key}'], 2),
            "Year": sig_date.year
        })

    trades_df = pd.DataFrame(trade_rows)
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    yearly_rows = []
    for yr, grp in trades_df.groupby('Year'):
        ret_s = grp['Return (%)']
        gains = ret_s[ret_s > 0].sum()
        losses = abs(ret_s[ret_s < 0].sum())
        pf = round(gains / losses, 2) if losses > 0 else (99.0 if gains > 0 else np.nan)
        comp_ret = ((1 + ret_s / 100.0).prod() - 1) * 100.0
        yearly_rows.append({
            "Year": str(yr),
            "Trades": int(len(grp)),
            "Win Rate (%)": round((ret_s > 0).mean() * 100.0, 1),
            "Avg Return (%)": round(ret_s.mean(), 2),
            "Profit Factor": pf,
            "Compounded Return (%)": round(comp_ret, 2),
            "Avg Hold (Days)": round(grp['Hold (Days)'].mean(), 1)
        })

    all_ret = trades_df['Return (%)']
    all_gains = all_ret[all_ret > 0].sum()
    all_losses = abs(all_ret[all_ret < 0].sum())
    all_pf = round(all_gains / all_losses, 2) if all_losses > 0 else np.nan
    all_comp = ((1 + all_ret / 100.0).prod() - 1) * 100.0

    yearly_rows.append({
        "Year": "All Years",
        "Trades": int(len(trades_df)),
        "Win Rate (%)": round((all_ret > 0).mean() * 100.0, 1),
        "Avg Return (%)": round(all_ret.mean(), 2),
        "Profit Factor": all_pf,
        "Compounded Return (%)": round(all_comp, 2),
        "Avg Hold (Days)": round(trades_df['Hold (Days)'].mean(), 1)
    })

    return pd.DataFrame(yearly_rows).set_index("Year"), trades_df

# ==========================================
# 2. MULTI-STOCK PORTFOLIO ENGINE (NEXT OPEN)
# ==========================================

def run_portfolio_backtest_next_open(
    prepared_dfs, selected_buckets, exit_key, exit_label,
    max_positions=5, initial_capital=100000.0,
    require_uptrend=True, fresh_entry_only=True
):
    """
    Day-by-day portfolio simulation using strict Next-Open execution:
    - Exits execute at Today's Open if Yesterday's Close triggered the exit rule.
    - Entries execute at Today's Open if Yesterday's Close triggered the entry bucket.
    """
    all_dates = sorted(list(set.union(*[set(df.index) for df in prepared_dfs.values()])))
    if len(all_dates) < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cash = float(initial_capital)
    open_positions = {}
    trade_log = []
    daily_equity_records = []

    for idx in range(1, len(all_dates)):
        d = all_dates[idx]
        prev_d = all_dates[idx - 1]

        # Step A: Process Exits at Today's Open (d) based on Yesterday's Close (prev_d)
        closed_tickers = []
        for ticker, pos in open_positions.items():
            df = prepared_dfs[ticker]
            if d not in df.index or prev_d not in df.index:
                continue

            if pos['entry_date'] > prev_d:
                continue

            pos['days_held'] += 1
            prev_row = df.loc[prev_d]
            today_open = df.loc[d, 'Open_Adj']

            exit_hit = False
            if exit_key == "SMA5" and prev_row['Close_Adj'] > prev_row['SMA_5']:
                exit_hit = True
            elif exit_key.startswith("HC"):
                x_win = int(exit_key.replace("HC", ""))
                if prev_row['Close_Adj'] >= prev_row[f'HighClose_{x_win}d']:
                    exit_hit = True
            elif exit_key.startswith("RSI_"):
                thresh = float(exit_key.replace("RSI_", ""))
                if prev_row['Active_RSI'] > thresh:
                    exit_hit = True
            elif exit_key == "FIX5" and pos['days_held'] >= 5:
                exit_hit = True
            elif exit_key == "FIX10" and pos['days_held'] >= 10:
                exit_hit = True

            if not exit_hit and pos['days_held'] >= 20:
                exit_hit = True

            if exit_hit:
                proceeds = pos['shares'] * today_open
                cash += proceeds
                trade_ret = (today_open - pos['entry_price']) / pos['entry_price'] * 100.0

                trade_log.append({
                    "Ticker": ticker,
                    "Signal Date (Close)": pos['signal_date'].strftime('%Y-%m-%d'),
                    "Entry Date (Open)": pos['entry_date'].strftime('%Y-%m-%d'),
                    "Entry Open ($)": round(pos['entry_price'], 2),
                    "Exit Date (Open)": d.strftime('%Y-%m-%d'),
                    "Exit Open ($)": round(today_open, 2),
                    "Exit Rule": exit_label,
                    "Return (%)": round(trade_ret, 2),
                    "Hold (Days)": pos['days_held'],
                    "Year": d.year,
                    "Result": "Win" if trade_ret > 0 else "Loss"
                })
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del open_positions[t]

        # Step B: Identify Entry Signals from prev_d Close -> Enter at Today's Open (d)
        available_slots = max_positions - len(open_positions)
        candidates = []

        if available_slots > 0:
            for ticker, df in prepared_dfs.items():
                if ticker in open_positions or d not in df.index or prev_d not in df.index:
                    continue

                prev_row = df.loc[prev_d]
                prev_bucket = str(prev_row['Bucket'])
                if prev_bucket not in selected_buckets:
                    continue

                if require_uptrend and not bool(prev_row['Above_200SMA']):
                    continue

                if fresh_entry_only:
                    prev_loc = df.index.get_loc(prev_d)
                    if prev_loc > 0 and str(df.iloc[prev_loc - 1]['Bucket']) == prev_bucket:
                        continue

                today_open = df.loc[d, 'Open_Adj']
                if pd.isna(today_open) or today_open <= 0:
                    continue

                candidates.append({
                    "ticker": ticker,
                    "entry_open": today_open,
                    "rsi_val": prev_row['Active_RSI']
                })

            candidates.sort(key=lambda x: x['rsi_val'])

            current_equity = cash
            for t_open, p_open in open_positions.items():
                df_o = prepared_dfs[t_open]
                px = df_o.loc[d, 'Open_Adj'] if d in df_o.index else p_open['entry_price']
                current_equity += p_open['shares'] * px

            target_slot_size = current_equity / max_positions

            for cand in candidates[:available_slots]:
                invest_amt = min(cash, target_slot_size)
                if invest_amt >= 250.0:
                    shares = invest_amt / cand['entry_open']
                    cash -= invest_amt
                    open_positions[cand['ticker']] = {
                        'signal_date': prev_d,
                        'entry_date': d,
                        'entry_price': cand['entry_open'],
                        'shares': shares,
                        'days_held': 0
                    }

        # Step C: Mark-to-Market Portfolio Equity at Close of Day d
        eod_equity = cash
        for t_open, p_open in open_positions.items():
            df_o = prepared_dfs[t_open]
            px = df_o.loc[d, 'Close_Adj'] if d in df_o.index else p_open['entry_price']
            eod_equity += p_open['shares'] * px

        daily_equity_records.append({"Date": d, "Equity": eod_equity})

    equity_df = pd.DataFrame(daily_equity_records).set_index("Date")
    trades_df = pd.DataFrame(trade_log)

    equity_df['Year'] = equity_df.index.year
    yearly_records = []

    for yr, grp in equity_df.groupby('Year'):
        start_eq = grp['Equity'].iloc[0]
        end_eq = grp['Equity'].iloc[-1]
        yr_return = ((end_eq / start_eq) - 1) * 100.0
        cummax = grp['Equity'].cummax()
        yr_dd = ((grp['Equity'] - cummax) / cummax).min() * 100.0

        yr_trades = trades_df[trades_df['Year'] == yr] if not trades_df.empty else pd.DataFrame()
        n_tr = len(yr_trades)
        if n_tr > 0:
            wr = (yr_trades['Result'] == 'Win').mean() * 100.0
            avg_r = yr_trades['Return (%)'].mean()
            gains = yr_trades[yr_trades['Return (%)'] > 0]['Return (%)'].sum()
            losses = abs(yr_trades[yr_trades['Return (%)'] < 0]['Return (%)'].sum())
            pf = round(gains / losses, 2) if losses > 0 else (99.0 if gains > 0 else np.nan)
            avg_d = yr_trades['Hold (Days)'].mean()
        else:
            wr, avg_r, pf, avg_d = 0.0, 0.0, np.nan, 0.0

        yearly_records.append({
            "Year": str(yr),
            "Trades": int(n_tr),
            "Win Rate (%)": round(wr, 1),
            "Avg Trade Ret (%)": round(avg_r, 2),
            "Profit Factor": pf,
            "Annual Portfolio Ret (%)": round(yr_return, 2),
            "Max Drawdown (%)": round(yr_dd, 1),
            "Avg Hold (Days)": round(avg_d, 1)
        })

    tot_ret = ((equity_df['Equity'].iloc[-1] / initial_capital) - 1) * 100.0
    overall_cummax = equity_df['Equity'].cummax()
    overall_dd = ((equity_df['Equity'] - overall_cummax) / overall_cummax).min() * 100.0

    if not trades_df.empty:
        all_wr = (trades_df['Result'] == 'Win').mean() * 100.0
        all_avg_r = trades_df['Return (%)'].mean()
        all_g = trades_df[trades_df['Return (%)'] > 0]['Return (%)'].sum()
        all_l = abs(trades_df[trades_df['Return (%)'] < 0]['Return (%)'].sum())
        all_pf = round(all_g / all_l, 2) if all_l > 0 else np.nan
        all_h = trades_df['Hold (Days)'].mean()
    else:
        all_wr, all_avg_r, all_pf, all_h = 0.0, 0.0, np.nan, 0.0

    yearly_records.append({
        "Year": "All Years",
        "Trades": int(len(trades_df)),
        "Win Rate (%)": round(all_wr, 1),
        "Avg Trade Ret (%)": round(all_avg_r, 2),
        "Profit Factor": all_pf,
        "Annual Portfolio Ret (%)": round(tot_ret, 2),
        "Max Drawdown (%)": round(overall_dd, 1),
        "Avg Hold (Days)": round(all_h, 1)
    })

    return pd.DataFrame(yearly_records).set_index("Year"), equity_df, trades_df

# ==========================================
# 3. DATA FETCHING & CACHING
# ==========================================

DJIA_FALLBACK = [
    "AAPL", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
]

RUSSELL_FALLBACK = [
    "BTSG", "FROG", "TWST", "GKOS", "UMBF", "SM", "PCVX", "FCFS", "EAT",
    "PBF", "CTRE", "ONB", "KRYS", "TXG", "JXN", "LTH", "KYMR", "PTGX", "KNTK",
    "CYTK", "TGTX", "IDCC", "VSAT", "ZETA", "RHP", "ZWS", "HQY", "URBN", "SLAB"
]

@st.cache_data(show_spinner=False, ttl=86400)
def get_universe_tickers(universe_name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
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
        return res if res else DJIA_FALLBACK
    elif universe_name == "Nasdaq 100":
        res = extract_wiki("https://en.wikipedia.org/wiki/Nasdaq-100")
        return res if res else ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "AMD"]
    elif universe_name == "Dow Jones 30":
        res = extract_wiki("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        return res if res else DJIA_FALLBACK
    elif "Russell 2000" in universe_name:
        res = extract_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
        return list(dict.fromkeys(res + RUSSELL_FALLBACK)) if res else RUSSELL_FALLBACK
    elif universe_name == "Major ETFs":
        return ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "SMH", "TLT", "GLD"]
    return []

@st.cache_data(show_spinner=False)
def fetch_daily_data(symbol, start, end):
    raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    df = normalize_ohlcv_df(raw)
    return df if not df.empty else None

# ==========================================
# 4. INTERFACE TABS (4 SEPARATE TABS)
# ==========================================

tab_study, tab_watchlist, tab_screener, tab_portfolio = st.tabs([
    "📊 Single Ticker Study (Next-Open)",
    "📋 Custom Watchlist Analyzer",
    "🔎 Live Universe Screener",
    "📁 Multi-Stock Portfolio Backtester"
])

# ----------------------------------------------------
# TAB 1: SINGLE TICKER STUDY (SIMPLIFIED & PERSISTENT)
# ----------------------------------------------------
with tab_study:
    st.subheader("Strategy Configuration")

    c1, c2, c3, c4 = st.columns([1.0, 1.3, 1.5, 0.9])
    with c1:
        study_ticker = st.text_input("Ticker Symbol", value="META", key="study_sym").upper().strip()
    with c2:
        rsi_choice = st.selectbox(
            "RSI(2) Lookback",
            ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"],
            index=0,
            help="Selects whether entry buckets and RSI exit thresholds use the 2-day, 3-day, or 5-day average of RSI(2)."
        )
    with c3:
        signal_mode = st.selectbox(
            "Entry Signal Counting",
            ["All Days in Bucket", "First Day Entering Bucket Only (No Overlap)"],
            index=0
        )
    with c4:
        years_back = st.number_input("History (Years)", min_value=1, max_value=30, value=20)

    active_rsi_days = RSI_DAYS_MAP[rsi_choice]
    exit_label_map = get_exit_label_map(rsi_choice)
    inv_exit_label_map = {v: k for k, v in exit_label_map.items()}

    c5, c6, c7 = st.columns([1.6, 1.6, 1.2])
    with c5:
        selected_exit_label = st.selectbox(
            "Exit Rule (Executed on Next Day's Open)",
            list(exit_label_map.values()),
            index=0,
            help="When this condition is met at the Close, the trade exits at the following day's Open."
        )
        active_exit_key = inv_exit_label_map[selected_exit_label]
    with c6:
        bucket_preset = st.selectbox(
            "Entry Bucket(s) for Deep-Dive & Yearly Table",
            [
                "All Oversold (<20: <10 & 11-20)",
                "Bucket <10 Only",
                "Bucket 11-20 Only",
                "Bucket 21-30 Only",
                "Bucket 31-40 Only",
                "Bucket 41-50 Only",
                "Bucket 51-60 Only",
                "Bucket 61-70 Only",
                "Bucket 71-80 Only",
                "Bucket 81-90 Only",
                "Bucket 91-100 Only",
                "All Buckets (<10 to 91-100)"
            ],
            index=0
        )
    with c7:
        st.write("")
        require_uptrend_study = st.checkbox(
            "Require Close > 200-Day SMA",
            value=True,
            help="Only take entry signals when the stock closes above its 200-day simple moving average."
        )

    if "All Oversold" in bucket_preset:
        active_buckets = ['<10', '11-20']
    elif "All Buckets" in bucket_preset:
        active_buckets = BUCKET_LABELS
    else:
        b_token = bucket_preset.replace("Bucket ", "").replace(" Only", "").strip()
        active_buckets = [b_token]

    start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    with st.spinner(f"Loading {study_ticker} daily history..."):
        raw_df = fetch_daily_data(study_ticker, start_date, end_date)

    if raw_df is None or len(raw_df) < 220:
        st.error(f"Insufficient history found for '{study_ticker}' (minimum 220 trading days required).")
    else:
        processed_df = prepare_stock_data(raw_df, rsi_days=active_rsi_days)
        fresh_only = ("First Day" in signal_mode)
        filtered_df = filter_signal_rows(processed_df, require_uptrend=require_uptrend_study, fresh_only=fresh_only)

        latest = processed_df.iloc[-1]
        curr_px = latest['Close_Adj']
        curr_rsi1 = latest['RSI2_1d']
        curr_rsi_n = latest['Active_RSI']
        curr_bucket = str(latest['Bucket'])
        is_bull = bool(latest['Above_200SMA'])

        st.divider()
        st.markdown(f"#### Live Status: **{study_ticker}** (as of {processed_df.index[-1].strftime('%Y-%m-%d')} Close)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Last Close", f"${curr_px:.2f}")
        m2.metric("1-Day RSI(2)", f"{curr_rsi1:.1f}")
        m3.metric(f"Current {rsi_choice}", f"{curr_rsi_n:.1f}")
        m4.metric(f"{rsi_choice} Bucket", curr_bucket)
        m5.metric("200-Day SMA Trend", "Above (Bullish)" if is_bull else "Below (Bearish)")

        # ==========================================
        # SECTION 1: ENTRY BUCKET BREAKDOWN
        # ==========================================
        st.divider()
        st.subheader(f"1. Performance by Entry Bucket — Exit: {selected_exit_label} (Next Open)")
        st.caption(
            f"Shows how each **{rsi_choice}** entry bucket performed when entering on the Next Day's Open "
            f"and exiting on the Next Day's Open after **{selected_exit_label}**."
        )

        bucket_table = build_bucket_table(filtered_df, active_exit_key)

        col_tbl, col_charts = st.columns([1.35, 1.65])
        with col_tbl:
            b_cfg = {
                "Trades": st.column_config.NumberColumn(format="%d"),
                "Win Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "Avg Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                "Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d"),
                "Avg Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%")
            }
            st.dataframe(bucket_table, column_config=b_cfg, use_container_width=True, height=390)

        with col_charts:
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=bucket_table.index,
                y=bucket_table['Avg Return (%)'],
                name='Avg Return (%)',
                marker_color=['#00E676' if v >= 0 else '#FF5252' for v in bucket_table['Avg Return (%)']],
                text=[f"{v:+.2f}%" for v in bucket_table['Avg Return (%)']],
                textposition='outside'
            ))
            fig_b.add_hline(y=0, line_dash="dash", line_color="white")
            fig_b.update_layout(
                title=f"Avg Trade Return (%) by {rsi_choice} Entry Bucket (Exit: {selected_exit_label})",
                template="plotly_dark",
                yaxis_title="Avg Return (%)",
                xaxis_title=f"{rsi_choice} Entry Bucket",
                height=390,
                margin=dict(l=20, r=20, t=45, b=20)
            )
            st.plotly_chart(fig_b, use_container_width=True)

        # ==========================================
        # SECTION 2: EXIT STRATEGY COMPARISON
        # ==========================================
        st.divider()
        bucket_title_str = ", ".join(active_buckets) if len(active_buckets) <= 3 else "All 10 Buckets"
        st.subheader(f"2. Compare All Exit Rules — Entry Bucket(s): {bucket_title_str}")

        sub_tab1, sub_tab2 = st.tabs([
            f"📋 Exit Comparison Table (Entry: {bucket_title_str})",
            "🔢 Full Entry × Exit Grid (All Buckets vs. All Exits)"
        ])

        with sub_tab1:
            st.caption(
                f"Compares all available Next-Open exit rules side-by-side when entering on **{rsi_choice}** in bucket(s): **{bucket_title_str}**."
            )
            exit_comp_df = build_exit_comparison_table(filtered_df, active_buckets, exit_label_map)
            st.dataframe(exit_comp_df, column_config=b_cfg, use_container_width=True, height=560)

        with sub_tab2:
            grid_metric = st.radio(
                "Select Metric to Display in Grid:",
                ["Avg Return (%)", "Win Rate (%)", "Profit Factor", "Avg Hold (Days)"],
                horizontal=True
            )
            matrix_df = build_entry_exit_matrix(filtered_df, grid_metric, exit_label_map)

            fmt_str = "%+.2f%%" if grid_metric == "Avg Return (%)" else (
                "%.1f%%" if grid_metric == "Win Rate (%)" else (
                    "%.2f" if grid_metric == "Profit Factor" else "%.1f d"
                )
            )
            mat_cfg = {"Trades": st.column_config.NumberColumn(format="%d")}
            for k in EXIT_KEYS:
                mat_cfg[exit_label_map[k]] = st.column_config.NumberColumn(format=fmt_str)

            st.dataframe(matrix_df, column_config=mat_cfg, use_container_width=True)

        # ==========================================
        # SECTION 3: YEAR-BY-YEAR & TRADE LOG
        # ==========================================
        st.divider()
        st.subheader(f"3. Year-by-Year Breakdown — Entry: {bucket_title_str} | Exit: {selected_exit_label}")

        yearly_df, trades_df = build_yearly_and_trades(processed_df, filtered_df, active_buckets, active_exit_key)

        if yearly_df.empty:
            st.info("No historical trades matched the selected filters.")
        else:
            y_col1, y_col2 = st.columns([1.3, 1.7])
            with y_col1:
                y_cfg = {
                    "Trades": st.column_config.NumberColumn(format="%d"),
                    "Win Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "Avg Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                    "Compounded Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d")
                }
                st.dataframe(yearly_df, column_config=y_cfg, use_container_width=True, height=400)

            with y_col2:
                chart_yr = yearly_df[yearly_df.index != "All Years"]
                fig_yr = go.Figure()
                fig_yr.add_trace(go.Bar(
                    x=chart_yr.index,
                    y=chart_yr['Compounded Return (%)'],
                    marker_color=['#00E676' if v >= 0 else '#FF5252' for v in chart_yr['Compounded Return (%)']],
                    hovertext=[
                        f"Year: {y}<br>Compounded: {r:+.2f}%<br>Trades: {t}<br>Win Rate: {w:.1f}%"
                        for y, r, t, w in zip(
                            chart_yr.index, chart_yr['Compounded Return (%)'],
                            chart_yr['Trades'], chart_yr['Win Rate (%)']
                        )
                    ],
                    hoverinfo='text'
                ))
                fig_yr.add_hline(y=0, line_dash="dash", line_color="white")
                fig_yr.update_layout(
                    title=f"{study_ticker} Annual Compounded Return (%) by Year",
                    template="plotly_dark",
                    yaxis_title="Compounded Return (%)",
                    xaxis_title="Year",
                    height=400,
                    margin=dict(l=20, r=20, t=45, b=20)
                )
                st.plotly_chart(fig_yr, use_container_width=True)

            with st.expander(f"🔎 View Individual Trade Audit Log ({len(trades_df)} Trades — Next-Open Execution)"):
                st.download_button(
                    label="Download Trade Log (CSV)",
                    data=trades_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{study_ticker}_{rsi_choice}_trades.csv",
                    mime="text/csv"
                )
                st.dataframe(trades_df.sort_values(by="Signal Date (Close)", ascending=False), use_container_width=True)

# ----------------------------------------------------
# TAB 2: CUSTOM WATCHLIST ANALYZER
# ----------------------------------------------------
with tab_watchlist:
    st.subheader("Custom Watchlist Analyzer (Next-Open Execution)")
    st.caption(
        "Enter one or more tickers to inspect their current RSI(2) bucket and historical Next-Open edge whenever they entered that exact bucket."
    )

    w_c1, w_c2, w_c3, w_c4 = st.columns([2.0, 1.1, 1.4, 0.8])
    with w_c1:
        w_input = st.text_input(
            "Watchlist Tickers (comma or space separated)",
            value="SPY, QQQ, IWM, META, NVDA, AAPL, MSFT, AMZN, GOOGL, TSLA",
            key="w_tickers_input"
        )
    with w_c2:
        w_rsi_choice = st.selectbox("RSI(2) Lookback", ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"], index=2, key="w_rsi")
        w_rsi_days = RSI_DAYS_MAP[w_rsi_choice]
        w_exit_map = get_exit_label_map(w_rsi_choice)
        w_inv_exit_map = {v: k for k, v in w_exit_map.items()}
    with w_c3:
        w_exit_label = st.selectbox("Benchmark Exit (Next Open)", list(w_exit_map.values()), index=0, key="w_exit")
        w_exit_key = w_inv_exit_map[w_exit_label]
    with w_c4:
        w_years = st.number_input("History (Yrs)", min_value=2, max_value=25, value=10, key="w_yrs")

    w_f1, w_f2 = st.columns(2)
    with w_f1:
        w_uptrend = st.checkbox("Require Historical Signals to be Above 200-Day SMA", value=True, key="w_trend")
    with w_f2:
        w_fresh = st.checkbox("First Day Entering Bucket Only (No Overlap)", value=False, key="w_fresh")

    if st.button("Analyze Custom Watchlist", type="primary", use_container_width=True):
        w_list = list(dict.fromkeys([s.strip().upper().replace('.', '-') for s in re.split(r'[, \n]+', w_input) if s.strip()]))
        if not w_list:
            st.warning("Please enter at least one valid ticker symbol.")
        else:
            with st.spinner(f"Analyzing {len(w_list)} watchlist ticker(s)..."):
                s_w = (datetime.now() - timedelta(days=w_years * 365)).strftime('%Y-%m-%d')
                e_w = datetime.now().strftime('%Y-%m-%d')
                bulk_w = yf.download(w_list, start=s_w, end=e_w, group_by="ticker", auto_adjust=False, progress=False, threads=True)

                rows = []
                for sym in w_list:
                    d_sym = extract_symbol_daily(bulk_w, sym, len(w_list))
                    if len(d_sym) < 220:
                        continue
                    p_df = prepare_stock_data(d_sym, rsi_days=w_rsi_days)
                    if p_df.empty:
                        continue

                    last = p_df.iloc[-1]
                    c_b = str(last['Bucket'])

                    hist = filter_signal_rows(p_df, require_uptrend=w_uptrend, fresh_only=w_fresh)
                    sub_b = hist[hist['Bucket'] == c_b]
                    n, wr, avg_r, pf, days, mae = summarize_series(
                        sub_b[f'Ret_{w_exit_key}'], sub_b[f'Days_{w_exit_key}'], sub_b[f'MAE_{w_exit_key}']
                    )

                    rows.append({
                        "Ticker": sym,
                        "Last Close": round(last['Close_Adj'], 2),
                        "1D RSI(2)": round(last['RSI2_1d'], 1),
                        "2D RSI(2)": round(last['RSI2_2d'], 1),
                        "3D RSI(2)": round(last['RSI2_3d'], 1),
                        "5D RSI(2)": round(last['RSI2_5d'], 1),
                        f"Current Bucket ({w_rsi_choice})": c_b,
                        "> 200 SMA": "Yes" if bool(last['Above_200SMA']) else "No",
                        "Prior 5D (%)": round(last['Prior_5d'], 2),
                        "Hist Trades (in Bucket)": n,
                        "Hist Win Rate (%)": wr,
                        "Hist Avg Ret (%)": avg_r,
                        "Hist Profit Factor": pf,
                        "Hist Avg Hold (Days)": days,
                        "Hist Avg Drawdown (%)": mae
                    })

                if not rows:
                    st.error("Could not retrieve sufficient trading history (minimum 220 days) for the entered ticker(s).")
                else:
                    scan_df = pd.DataFrame(rows).sort_values(by=f"{w_rsi_days}D RSI(2)", ascending=True)
                    w_col_cfg = {
                        "Last Close": st.column_config.NumberColumn(format="$%.2f"),
                        "1D RSI(2)": st.column_config.NumberColumn(format="%.1f"),
                        "2D RSI(2)": st.column_config.NumberColumn(format="%.1f"),
                        "3D RSI(2)": st.column_config.NumberColumn(format="%.1f"),
                        "5D RSI(2)": st.column_config.NumberColumn(format="%.1f"),
                        "Prior 5D (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                        "Hist Trades (in Bucket)": st.column_config.NumberColumn(format="%d"),
                        "Hist Win Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                        "Hist Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                        "Hist Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                        "Hist Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d"),
                        "Hist Avg Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%")
                    }
                    st.dataframe(scan_df, column_config=w_col_cfg, use_container_width=True)
                    st.download_button(
                        label="Download Watchlist Results (CSV)",
                        data=scan_df.to_csv(index=False).encode('utf-8'),
                        file_name="watchlist_next_open_results.csv",
                        mime="text/csv"
                    )

# ----------------------------------------------------
# TAB 3: LIVE UNIVERSE SCREENER
# ----------------------------------------------------
with tab_screener:
    st.subheader("Live Index Universe Screener (Next-Open Execution)")
    st.caption(
        "Scans an entire index universe for stocks currently sitting in your target RSI(2) entry buckets as of the latest Close."
    )

    s_c1, s_c2, s_c3, s_c4 = st.columns([1.2, 1.1, 1.4, 1.4])
    with s_c1:
        scr_universe = st.selectbox(
            "Select Index Universe",
            ["Major ETFs", "Dow Jones 30", "Nasdaq 100", "S&P 500", "Russell 2000 (Small Caps)"],
            index=2,
            key="scr_univ"
        )
    with s_c2:
        scr_rsi_choice = st.selectbox(
            "RSI(2) Lookback",
            ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"],
            index=2,
            key="scr_rsi"
        )
        scr_rsi_days = RSI_DAYS_MAP[scr_rsi_choice]
        scr_exit_map = get_exit_label_map(scr_rsi_choice)
        scr_inv_exit_map = {v: k for k, v in scr_exit_map.items()}
    with s_c3:
        scr_buckets = st.multiselect(
            "Filter Active Entry Buckets",
            options=BUCKET_LABELS,
            default=['<10', '11-20'],
            key="scr_buckets"
        )
    with s_c4:
        scr_exit_label = st.selectbox(
            "Benchmark Exit (Next Open)",
            list(scr_exit_map.values()),
            index=0,
            key="scr_exit"
        )
        scr_exit_key = scr_inv_exit_map[scr_exit_label]

    s_f1, s_f2, s_f3 = st.columns([1.4, 1.4, 1.0])
    with s_f1:
        scr_uptrend_gate = st.checkbox(
            "Enforce Close > 200-Day SMA Gate (Today & Historical)",
            value=True,
            key="scr_trend"
        )
    with s_f2:
        scr_fresh_gate = st.checkbox(
            "Only Show Fresh Entries Today (Just Entered Bucket on Latest Close)",
            value=False,
            key="scr_fresh"
        )
    with s_f3:
        scr_years = st.number_input("Historical Backtest Depth (Yrs)", min_value=2, max_value=15, value=5, key="scr_yrs")

    if st.button("🔎 Run Live Universe Screen", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Checking S&P 500 (SPY) broad market regime...")
        spy_df = fetch_daily_data(
            "SPY",
            (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        )
        spy_rsi5 = np.nan
        if spy_df is not None and len(spy_df) >= 200:
            spy_prep = prepare_stock_data(spy_df, rsi_days=5)
            spy_rsi5 = spy_prep['RSI2_5d'].iloc[-1]

        progress_bar.progress(15)
        status_text.text(f"Loading constituent list for {scr_universe}...")
        u_tickers = get_universe_tickers(scr_universe)

        progress_bar.progress(30)
        status_text.text(f"Downloading historical daily data for {len(u_tickers)} tickers...")
        s_sc = (datetime.now() - timedelta(days=scr_years * 365)).strftime('%Y-%m-%d')
        e_sc = datetime.now().strftime('%Y-%m-%d')
        bulk_sc = yf.download(u_tickers, start=s_sc, end=e_sc, group_by="ticker", auto_adjust=False, progress=False, threads=True)

        progress_bar.progress(75)
        status_text.text("Filtering active setups and computing Next-Open historical expectancy...")

        screen_rows = []
        for sym in u_tickers:
            d_sym = extract_symbol_daily(bulk_sc, sym, len(u_tickers))
            if len(d_sym) < 220:
                continue
            p_df = prepare_stock_data(d_sym, rsi_days=scr_rsi_days)
            if p_df.empty:
                continue

            last = p_df.iloc[-1]
            prev = p_df.iloc[-2]
            c_b = str(last['Bucket'])
            is_bull = bool(last['Above_200SMA'])

            if scr_uptrend_gate and not is_bull:
                continue
            if scr_buckets and c_b not in scr_buckets:
                continue
            if scr_fresh_gate and str(prev['Bucket']) == c_b:
                continue

            hist = filter_signal_rows(p_df, require_uptrend=scr_uptrend_gate, fresh_only=scr_fresh_gate)
            sub_b = hist[hist['Bucket'] == c_b]
            n, wr, avg_r, pf, days, mae = summarize_series(
                sub_b[f'Ret_{scr_exit_key}'], sub_b[f'Days_{scr_exit_key}'], sub_b[f'MAE_{scr_exit_key}']
            )

            screen_rows.append({
                "Ticker": sym,
                "Last Close": round(last['Close_Adj'], 2),
                "1D RSI(2)": round(last['RSI2_1d'], 1),
                "2D RSI(2)": round(last['RSI2_2d'], 1),
                "3D RSI(2)": round(last['RSI2_3d'], 1),
                "5D RSI(2)": round(last['RSI2_5d'], 1),
                f"Active Bucket ({scr_rsi_choice})": c_b,
                "> 200 SMA": "Yes" if is_bull else "No",
                "Prior 5D (%)": round(last['Prior_5d'], 2),
                "Hist Trades": n,
                "Hist Win Rate (%)": wr,
                "Hist Avg Ret (%)": avg_r,
                "Hist Profit Factor": pf,
                "Hist Avg Hold (Days)": days,
                "Hist Avg Drawdown (%)": mae
            })

        progress_bar.progress(100)
        status_text.empty()

        if pd.notna(spy_rsi5):
            if spy_rsi5 <= 25:
                st.success(f"**Macro Context**: Systemic Oversold Flush (SPY 5-Day RSI(2) = {spy_rsi5:.1f}). Broad market mean-reversion tailwinds active.")
            else:
                st.info(f"**Macro Context**: Neutral / Bullish Market Breadth (SPY 5-Day RSI(2) = {spy_rsi5:.1f}). Pullbacks are stock-specific.")

        if not screen_rows:
            st.info(f"No stocks in **{scr_universe}** currently match the selected bucket and trend filters.")
        else:
            st.subheader(f"Filtered Setups Found: {len(screen_rows)} Tickers")
            scr_df = pd.DataFrame(screen_rows).sort_values(by=f"{scr_rsi_days}D RSI(2)", ascending=True)
            scr_col_cfg = {
                "Last Close": st.column_config.NumberColumn(format="$%.2f"),
                "1D RSI(2)": st.column_config.NumberColumn(format="%.1f"),
                "2D RSI(2)": st.column_config.NumberColumn(format="%.1f"),
                "3D RSI(2)": st.column_config.NumberColumn(format="%.1f"),
                "5D RSI(2)": st.column_config.NumberColumn(format="%.1f"),
                "Prior 5D (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "Hist Trades": st.column_config.NumberColumn(format="%d"),
                "Hist Win Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "Hist Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "Hist Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                "Hist Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d"),
                "Hist Avg Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%")
            }
            st.dataframe(scr_df, column_config=scr_col_cfg, use_container_width=True)
            st.download_button(
                label="Download Screener Matches (CSV)",
                data=scr_df.to_csv(index=False).encode('utf-8'),
                file_name=f"screener_{scr_universe.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

# ----------------------------------------------------
# TAB 4: MULTI-STOCK PORTFOLIO BACKTESTER
# ----------------------------------------------------
with tab_portfolio:
    st.subheader("Multi-Stock Portfolio Backtester (Next-Open Execution)")
    st.caption(
        "Simulates a portfolio across a universe or custom basket: Entry & Exit signals evaluated at Close → Orders executed at Next Day's Open."
    )

    p_col1, p_col2, p_col3 = st.columns([1.3, 1.4, 1.3])
    with p_col1:
        port_source = st.radio("Universe Mode", ["Index Universe", "Custom Ticker List"], horizontal=True)
    with p_col2:
        if port_source == "Index Universe":
            port_univ = st.selectbox("Select Universe", ["Major ETFs", "Dow Jones 30", "Nasdaq 100", "S&P 500", "Russell 2000 (Small Caps)"], index=1)
        else:
            port_custom = st.text_input("Tickers (comma-separated)", value="META, NVDA, AAPL, MSFT, AMZN, GOOGL, TSLA, AMD, AVGO, NFLX")
    with p_col3:
        port_pool_limit = st.selectbox("Max Universe Candidates", ["Top 30 Liquid", "Top 50 Liquid", "Top 100 Liquid"], index=0)

    p_c4, p_c5, p_c6, p_c7 = st.columns([1.1, 1.3, 1.5, 0.9])
    with p_c4:
        port_rsi_choice = st.selectbox("RSI(2) Lookback", ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"], index=0, key="p_rsi")
        port_rsi_days = RSI_DAYS_MAP[port_rsi_choice]
        port_exit_map = get_exit_label_map(port_rsi_choice)
        port_inv_exit_map = {v: k for k, v in port_exit_map.items()}
    with p_c5:
        port_buckets = st.multiselect("Target Entry Buckets", options=BUCKET_LABELS, default=['<10', '11-20'], key="p_buckets")
    with p_c6:
        port_exit_label = st.selectbox("Exit Rule (Next Open)", list(port_exit_map.values()), index=0, key="p_exit")
        port_exit_key = port_inv_exit_map[port_exit_label]
    with p_c7:
        port_years = st.number_input("Years", min_value=2, max_value=25, value=10, key="p_yrs")

    p_c8, p_c9, p_c10, p_c11 = st.columns([1.0, 1.0, 1.2, 1.2])
    with p_c8:
        port_max_pos = st.number_input("Max Open Positions", min_value=1, max_value=20, value=5)
    with p_c9:
        port_capital = st.number_input("Starting Capital ($)", min_value=10000, value=100000, step=10000)
    with p_c10:
        st.write("")
        port_uptrend = st.checkbox("Require Close > 200 SMA", value=True, key="p_trend")
    with p_c11:
        st.write("")
        port_fresh = st.checkbox("First Day in Bucket Only", value=True, key="p_fresh")

    if st.button("🚀 Run Next-Open Portfolio Simulation", type="primary", use_container_width=True):
        if port_source == "Index Universe":
            roster = get_universe_tickers(port_univ)
            lim = 30 if "30" in port_pool_limit else (50 if "50" in port_pool_limit else 100)
            tickers_list = roster[:lim]
        else:
            tickers_list = list(dict.fromkeys([s.strip().upper().replace('.', '-') for s in re.split(r'[, \n]+', port_custom) if s.strip()]))

        if not tickers_list or not port_buckets:
            st.warning("Please select at least one ticker and one entry bucket.")
        else:
            with st.spinner(f"Downloading data and simulating Next-Open portfolio across {len(tickers_list)} ticker(s)..."):
                s_p = (datetime.now() - timedelta(days=port_years * 365)).strftime('%Y-%m-%d')
                e_p = datetime.now().strftime('%Y-%m-%d')
                bulk_p = yf.download(tickers_list, start=s_p, end=e_p, group_by="ticker", auto_adjust=False, progress=False, threads=True)

                prep_dict = {}
                for sym in tickers_list:
                    d_sym = extract_symbol_daily(bulk_p, sym, len(tickers_list))
                    if len(d_sym) >= 220:
                        p_df = prepare_stock_data(d_sym, rsi_days=port_rsi_days)
                        if not p_df.empty:
                            prep_dict[sym] = p_df

            if not prep_dict:
                st.error("Could not load sufficient data for the selected tickers.")
            else:
                p_yr_df, p_eq_df, p_tr_df = run_portfolio_backtest_next_open(
                    prepared_dfs=prep_dict,
                    selected_buckets=port_buckets,
                    exit_key=port_exit_key,
                    exit_label=port_exit_label,
                    max_positions=int(port_max_pos),
                    initial_capital=float(port_capital),
                    require_uptrend=port_uptrend,
                    fresh_entry_only=port_fresh
                )

                if not p_yr_df.empty:
                    ov = p_yr_df.loc["All Years"]
                    k1, k2, k3, k4, k5, k6 = st.columns(6)
                    k1.metric("Total Return", f"{ov['Annual Portfolio Ret (%)']:+,.2f}%")
                    k2.metric("Total Trades", f"{int(ov['Trades'])}")
                    k3.metric("Win Rate", f"{ov['Win Rate (%)']:.1f}%")
                    k4.metric("Avg Trade Return", f"{ov['Avg Trade Ret (%)']:+.2f}%")
                    k5.metric("Profit Factor", f"{ov['Profit Factor']}")
                    k6.metric("Max Drawdown", f"{ov['Max Drawdown (%)']:.1f}%")

                    st.dataframe(p_yr_df, use_container_width=True)

                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(x=p_eq_df.index, y=p_eq_df['Equity'], mode='lines', line=dict(color='#00E676', width=2), name='Equity'))
                    fig_eq.update_layout(title="Portfolio Equity Curve ($)", template="plotly_dark", height=360, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_eq, use_container_width=True)

                    if not p_tr_df.empty:
                        with st.expander(f"View Portfolio Trade Log ({len(p_tr_df)} Closed Trades)"):
                            st.dataframe(p_tr_df.sort_values(by="Exit Date (Open)", ascending=False), use_container_width=True)
