import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import re

st.set_page_config(page_title="RSI(2) Next-Open Backtester & Exit Monitor", layout="wide")

st.title("RSI(2) Quantitative Backtester & Live Exit Price Monitor")
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

def calculate_wilders_rsi_with_state(prices, period=2):
    """
    Calculates standard Wilder's Smoothed RSI(2) and also returns the avg_gain and avg_loss
    series so we can algebraically solve for exact next-bar RSI target exit prices.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100), avg_gain, avg_loss

def calculate_wilders_rsi(prices, period=2):
    rsi, _, _ = calculate_wilders_rsi_with_state(prices, period=period)
    return rsi

def solve_next_close_for_rsi2(last_close, avg_gain, avg_loss, current_rsi, target_rsi):
    """
    Computes the exact Next-Day Close Price required for 1-Day Wilder's RSI(2) to reach `target_rsi`.
    For period=2 (alpha=0.5), RS_next = (avg_gain + gain_next) / (avg_loss + loss_next).
    """
    if pd.isna(last_close) or pd.isna(avg_gain) or pd.isna(avg_loss):
        return np.nan
    if target_rsi <= 0.0 or target_rsi >= 100.0:
        return np.nan

    rs_target = target_rsi / (100.0 - target_rsi)

    if target_rsi > current_rsi:
        # Requires a positive price change (gain_next > 0, loss_next = 0)
        req_gain = rs_target * avg_loss - avg_gain
        return max(0.01, float(last_close + max(0.0, req_gain)))
    else:
        # Can still achieve target_rsi even if price drops by req_loss
        if rs_target <= 0:
            return np.nan
        req_loss = (avg_gain / rs_target) - avg_loss
        return max(0.01, float(last_close - max(0.0, req_loss)))

def solve_next_close_for_nday_rsi2(last_close, avg_gain, avg_loss, recent_rsi_list, n_days, target_avg_rsi):
    """
    Computes the exact Next-Day Close Price required so the N-Day Average RSI(2)
    crosses `target_avg_rsi` on tomorrow's close.
    Returns np.nan if mathematically unreachable in a single session (e.g. requires 1D RSI > 99.9).
    """
    if n_days == 1:
        return solve_next_close_for_rsi2(last_close, avg_gain, avg_loss, recent_rsi_list[-1], target_avg_rsi)

    prior_sum = sum(recent_rsi_list[-(n_days - 1):])
    req_1d_rsi = (n_days * target_avg_rsi) - prior_sum

    if req_1d_rsi <= 0.1:
        # Will exceed threshold even on a sharp drop; compute floor price where 1D RSI = 0.5
        return solve_next_close_for_rsi2(last_close, avg_gain, avg_loss, recent_rsi_list[-1], 0.5)
    elif req_1d_rsi >= 99.9:
        # Cannot reach target N-day average in a single bar even with a massive rally
        return np.nan
    else:
        return solve_next_close_for_rsi2(last_close, avg_gain, avg_loss, recent_rsi_list[-1], req_1d_rsi)

def compute_ticker_snapshot_row(raw_df, sym, active_rsi_days=2, custom_exit_rsi_days=2, custom_exit_thresh=70):
    """
    Computes a single row containing current RSI readings and all Exit Target Prices ($)
    for the Universe Monitor and Watchlist tables.
    """
    df = normalize_ohlcv_df(raw_df)
    if len(df) < 205:
        return None

    if 'Adj Close' in df.columns and 'Close' in df.columns:
        close_s = df['Adj Close']
    else:
        close_s = df['Close']

    rsi_s, avg_gain_s, avg_loss_s = calculate_wilders_rsi_with_state(close_s, period=2)
    rsi_2d = rsi_s.rolling(2).mean()
    rsi_3d = rsi_s.rolling(3).mean()
    rsi_5d = rsi_s.rolling(5).mean()

    sma5 = close_s.rolling(5).mean()
    sma200 = close_s.rolling(200).mean()

    hc3 = close_s.rolling(3).max()
    hc5 = close_s.rolling(5).max()
    hc7 = close_s.rolling(7).max()
    hc10 = close_s.rolling(10).max()

    last_c = float(close_s.iloc[-1])
    prev_c = float(close_s.iloc[-2])
    c_5d_ago = float(close_s.iloc[-6]) if len(close_s) >= 6 else prev_c

    chg_1d = (last_c - prev_c) / prev_c * 100.0
    chg_5d = (last_c - c_5d_ago) / c_5d_ago * 100.0

    r1 = float(rsi_s.iloc[-1])
    r2 = float(rsi_2d.iloc[-1])
    r3 = float(rsi_3d.iloc[-1])
    r5 = float(rsi_5d.iloc[-1])

    active_r = {1: r1, 2: r2, 3: r3, 5: r5}.get(active_rsi_days, r2)
    bins = [-0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.001]
    bucket_str = str(pd.cut([active_r], bins=bins, labels=BUCKET_LABELS)[0])

    cur_sma5 = float(sma5.iloc[-1])
    # Exact close needed tomorrow so Close[t+1] > SMA5[t+1] is the mean of the last 4 closes
    next_sma5_trigger = float(close_s.iloc[-4:].mean())
    pct_to_sma5 = (cur_sma5 - last_c) / last_c * 100.0

    cur_sma200 = float(sma200.iloc[-1])
    pct_vs_200 = (last_c - cur_sma200) / cur_sma200 * 100.0 if cur_sma200 > 0 else np.nan

    ag_last = float(avg_gain_s.iloc[-1])
    al_last = float(avg_loss_s.iloc[-1])
    recent_r_list = [float(x) for x in rsi_s.iloc[-5:].values]

    px_rsi1_50 = solve_next_close_for_rsi2(last_c, ag_last, al_last, r1, 50.0)
    px_rsi1_70 = solve_next_close_for_rsi2(last_c, ag_last, al_last, r1, 70.0)
    px_rsi1_80 = solve_next_close_for_rsi2(last_c, ag_last, al_last, r1, 80.0)
    px_rsi1_90 = solve_next_close_for_rsi2(last_c, ag_last, al_last, r1, 90.0)

    px_custom_rsi = solve_next_close_for_nday_rsi2(
        last_c, ag_last, al_last, recent_r_list, custom_exit_rsi_days, float(custom_exit_thresh)
    )

    custom_col_name = f"Exit: {custom_exit_rsi_days}D RSI>{custom_exit_thresh} ($)"

    return {
        "Ticker": sym,
        "Last Close ($)": round(last_c, 2),
        "1D Chg (%)": round(chg_1d, 2),
        "5D Chg (%)": round(chg_5d, 2),
        "1D RSI(2)": round(r1, 1),
        "2D RSI(2)": round(r2, 1),
        "3D RSI(2)": round(r3, 1),
        "5D RSI(2)": round(r5, 1),
        "Bucket": bucket_str,
        "Exit: 5D SMA ($)": round(cur_sma5, 2),
        "Next Close > SMA5 ($)": round(next_sma5_trigger, 2),
        "% to 5D SMA": round(pct_to_sma5, 2),
        "Exit: 3D High ($)": round(float(hc3.iloc[-1]), 2),
        "Exit: 5D High ($)": round(float(hc5.iloc[-1]), 2),
        "Exit: 7D High ($)": round(float(hc7.iloc[-1]), 2),
        "Exit: 10D High ($)": round(float(hc10.iloc[-1]), 2),
        "Exit: 1D RSI>50 ($)": round(px_rsi1_50, 2) if pd.notna(px_rsi1_50) else np.nan,
        "Exit: 1D RSI>70 ($)": round(px_rsi1_70, 2) if pd.notna(px_rsi1_70) else np.nan,
        "Exit: 1D RSI>80 ($)": round(px_rsi1_80, 2) if pd.notna(px_rsi1_80) else np.nan,
        "Exit: 1D RSI>90 ($)": round(px_rsi1_90, 2) if pd.notna(px_rsi1_90) else np.nan,
        custom_col_name: round(px_custom_rsi, 2) if pd.notna(px_custom_rsi) else np.nan,
        "200D SMA ($)": round(cur_sma200, 2),
        "% vs 200 SMA": round(pct_vs_200, 2) if pd.notna(pct_vs_200) else np.nan
    }

def get_snapshot_column_config(custom_col_name):
    """Returns strict numeric column configuration so every column sorts properly."""
    return {
        "Ticker": st.column_config.TextColumn("Ticker", pinned=True, width="small"),
        "Last Close ($)": st.column_config.NumberColumn("Last Close ($)", format="$%.2f"),
        "1D Chg (%)": st.column_config.NumberColumn("1D Chg (%)", format="%+.2f%%"),
        "5D Chg (%)": st.column_config.NumberColumn("5D Chg (%)", format="%+.2f%%"),
        "1D RSI(2)": st.column_config.NumberColumn("1D RSI(2)", format="%.1f"),
        "2D RSI(2)": st.column_config.NumberColumn("2D RSI(2)", format="%.1f"),
        "3D RSI(2)": st.column_config.NumberColumn("3D RSI(2)", format="%.1f"),
        "5D RSI(2)": st.column_config.NumberColumn("5D RSI(2)", format="%.1f"),
        "Bucket": st.column_config.TextColumn("Bucket"),
        "Exit: 5D SMA ($)": st.column_config.NumberColumn(
            "Exit: 5D SMA ($)", format="$%.2f", help="Current 5-Day Simple Moving Average."
        ),
        "Next Close > SMA5 ($)": st.column_config.NumberColumn(
            "Next Close > SMA5 ($)", format="$%.2f",
            help="Exact Close price needed tomorrow so tomorrow's Close finishes above tomorrow's 5-Day SMA."
        ),
        "% to 5D SMA": st.column_config.NumberColumn(
            "% to 5D SMA", format="%+.2f%%", help="Percentage move from Last Close to current 5D SMA."
        ),
        "Exit: 3D High ($)": st.column_config.NumberColumn("Exit: 3D High ($)", format="$%.2f"),
        "Exit: 5D High ($)": st.column_config.NumberColumn("Exit: 5D High ($)", format="$%.2f"),
        "Exit: 7D High ($)": st.column_config.NumberColumn("Exit: 7D High ($)", format="$%.2f"),
        "Exit: 10D High ($)": st.column_config.NumberColumn("Exit: 10D High ($)", format="$%.2f"),
        "Exit: 1D RSI>50 ($)": st.column_config.NumberColumn(
            "Exit: 1D RSI>50 ($)", format="$%.2f", help="Next Close price required for 1D RSI(2) to cross 50."
        ),
        "Exit: 1D RSI>70 ($)": st.column_config.NumberColumn(
            "Exit: 1D RSI>70 ($)", format="$%.2f", help="Next Close price required for 1D RSI(2) to cross 70."
        ),
        "Exit: 1D RSI>80 ($)": st.column_config.NumberColumn(
            "Exit: 1D RSI>80 ($)", format="$%.2f", help="Next Close price required for 1D RSI(2) to cross 80."
        ),
        "Exit: 1D RSI>90 ($)": st.column_config.NumberColumn(
            "Exit: 1D RSI>90 ($)", format="$%.2f", help="Next Close price required for 1D RSI(2) to cross 90."
        ),
        custom_col_name: st.column_config.NumberColumn(
            custom_col_name, format="$%.2f",
            help="Next Close price required for the selected N-Day RSI(2) to cross the target threshold (NaN if unreachable in 1 session)."
        ),
        "200D SMA ($)": st.column_config.NumberColumn("200D SMA ($)", format="$%.2f"),
        "% vs 200 SMA": st.column_config.NumberColumn("% vs 200 SMA", format="%+.2f%%")
    }

def prepare_stock_data(raw_df, rsi_days=5, max_hold=20):
    """
    Computes adjusted OHLC, N-day RSI(2) buckets, and Next-Open forward trade outcomes
    for all exit strategies.
    """
    df = normalize_ohlcv_df(raw_df)
    if df.empty:
        return pd.DataFrame()

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

# ==========================================
# CANDLESTICK + 2-DAY & 5-DAY RSI(2) CHART
# ==========================================

def build_candlestick_and_rsi_chart(df, ticker, lookback_bars=126):
    plot_df = df.iloc[-lookback_bars:].copy() if lookback_bars and len(df) > lookback_bars else df.copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            f"{ticker} Daily Candlestick Price Action (with 5-SMA & 200-SMA)",
            "2-Day Average RSI(2)  [Lines at 10, 20, 80, 90]",
            "5-Day Average RSI(2)  [Lines at 10, 20, 80, 90]"
        )
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open_Adj'],
            high=plot_df['High_Adj'],
            low=plot_df['Low_Adj'],
            close=plot_df['Close_Adj'],
            name="Price (Adj)",
            increasing_line_color="#00E676",
            decreasing_line_color="#FF5252"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df['SMA_5'],
            mode='lines',
            name='5-Day SMA',
            line=dict(color='#29B6F6', width=1.4)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df['SMA_200'],
            mode='lines',
            name='200-Day SMA',
            line=dict(color='#FFA726', width=1.8, dash='dot')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df['RSI2_2d'],
            mode='lines',
            name='2-Day RSI(2)',
            line=dict(color='#CE93D8', width=2.0)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df['RSI2_5d'],
            mode='lines',
            name='5-Day RSI(2)',
            line=dict(color='#4DD0E1', width=2.0)
        ),
        row=3, col=1
    )

    threshold_styles = [
        (90, "#FF1744", "solid"),
        (80, "#FF9100", "dash"),
        (20, "#00E5FF", "dash"),
        (10, "#00E676", "solid"),
    ]

    for r_idx in [2, 3]:
        fig.add_hrect(y0=80, y1=90, fillcolor="#FF5252", opacity=0.08, line_width=0, row=r_idx, col=1)
        fig.add_hrect(y0=90, y1=100, fillcolor="#FF1744", opacity=0.14, line_width=0, row=r_idx, col=1)
        fig.add_hrect(y0=10, y1=20, fillcolor="#00E5FF", opacity=0.08, line_width=0, row=r_idx, col=1)
        fig.add_hrect(y0=0,  y1=10, fillcolor="#00E676", opacity=0.14, line_width=0, row=r_idx, col=1)

        for lvl, color, dash_style in threshold_styles:
            fig.add_hline(
                y=lvl,
                line_dash=dash_style,
                line_color=color,
                line_width=1.2,
                row=r_idx,
                col=1
            )

        fig.update_yaxes(
            range=[0, 100],
            tickvals=[10, 20, 50, 80, 90],
            title_text="RSI(2)",
            row=r_idx,
            col=1
        )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        rangeslider_visible=False
    )

    fig.update_layout(
        template="plotly_dark",
        height=760,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=55, b=20)
    )

    return fig

# ==========================================
# 2. TABLE & SUMMARY BUILDERS
# ==========================================

def filter_signal_rows(df, require_uptrend=True, fresh_only=False):
    data = df.dropna(subset=['Bucket']).copy()
    if require_uptrend:
        data = data[data['Above_200SMA'] == True]
    if fresh_only:
        data = data[data['Bucket'] != data['Bucket'].shift(1)]
    return data

def summarize_series(ret_s, days_s, mae_s):
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
# 3. MULTI-STOCK PORTFOLIO ENGINE (NEXT OPEN)
# ==========================================

def run_portfolio_backtest_next_open(
    prepared_dfs, selected_buckets, exit_key, exit_label,
    max_positions=5, initial_capital=100000.0,
    require_uptrend=True, fresh_entry_only=True
):
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
# 4. DATA FETCHING & CACHING
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

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_bulk_recent_data(tickers_tuple, days_back=420):
    """Cached bulk downloader for fast Universe & Watchlist snapshot generation."""
    tickers_list = list(tickers_tuple)
    if not tickers_list:
        return pd.DataFrame()
    s_d = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    e_d = datetime.now().strftime('%Y-%m-%d')
    return yf.download(
        tickers_list,
        start=s_d,
        end=e_d,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True
    )

# ==========================================
# 5. INTERFACE TABS (4 SEPARATE TABS)
# ==========================================

tab_study, tab_screener, tab_watchlist, tab_portfolio = st.tabs([
    "📊 Single Ticker Study (Next-Open)",
    "🌐 Universe Monitor & Exit Prices",
    "📋 Custom Watchlist & Exit Prices",
    "📁 Multi-Stock Portfolio Backtester"
])

# ----------------------------------------------------
# TAB 1: SINGLE TICKER STUDY (WITH CANDLESTICK & RSI SUBPLOTS)
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
        # PRICE CANDLESTICK & 2D / 5D RSI(2) SUBPLOTS
        # ==========================================
        st.divider()
        ch_col1, ch_col2 = st.columns([2.8, 1.2])
        with ch_col1:
            st.subheader(f"📈 {study_ticker} Daily Candlestick & Multi-Day RSI(2) Oscillators")
        with ch_col2:
            chart_zoom = st.selectbox(
                "Chart Display Window",
                [
                    "Last 3 Months (~63 Bars)",
                    "Last 6 Months (~126 Bars)",
                    "Last 1 Year (~252 Bars)",
                    "Last 2 Years (~504 Bars)",
                    "Full Backtest History"
                ],
                index=1,
                key="chart_zoom_window"
            )

        zoom_map = {
            "Last 3 Months (~63 Bars)": 63,
            "Last 6 Months (~126 Bars)": 126,
            "Last 1 Year (~252 Bars)": 252,
            "Last 2 Years (~504 Bars)": 504,
            "Full Backtest History": None
        }
        fig_candles = build_candlestick_and_rsi_chart(
            processed_df, study_ticker, lookback_bars=zoom_map[chart_zoom]
        )
        st.plotly_chart(fig_candles, use_container_width=True)

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
# TAB 2: UNIVERSE MONITOR & EXIT PRICES (ALL TICKERS)
# ----------------------------------------------------
with tab_screener:
    st.subheader("🌐 Full Universe Monitor — Current RSI Readings & Target Exit Prices")
    st.caption(
        "Select any universe from the dropdown to list ALL constituent tickers, their live 1D/2D/3D/5D RSI(2) readings, "
        "and exact target exit prices ($). Click any column header in the table to sort ascending or descending."
    )

    u_c1, u_c2, u_c3, u_c4, u_c5 = st.columns([1.3, 1.1, 1.1, 1.1, 1.2])
    with u_c1:
        mon_universe = st.selectbox(
            "1. Select Universe",
            ["Major ETFs", "Dow Jones 30", "Nasdaq 100", "Russell 2000 (Small Caps)", "S&P 500"],
            index=1,
            key="mon_univ_select"
        )
    with u_c2:
        mon_bucket_rsi = st.selectbox(
            "Bucket RSI Lookback",
            ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"],
            index=0,
            key="mon_b_rsi"
        )
        mon_b_days = RSI_DAYS_MAP[mon_bucket_rsi]
    with u_c3:
        mon_custom_exit_days = st.selectbox(
            "Custom Exit RSI Lookback",
            [1, 2, 3, 5],
            index=1,
            format_func=lambda x: f"{x}-Day RSI(2)",
            key="mon_cx_days"
        )
    with u_c4:
        mon_custom_exit_thresh = st.selectbox(
            "Custom Exit RSI Threshold",
            RSI_THRESHOLDS,
            index=6,  # Default > 70
            format_func=lambda x: f"RSI > {x}",
            key="mon_cx_thresh"
        )
    with u_c5:
        mon_search = st.text_input("Quick Filter Ticker (Optional)", value="", placeholder="e.g. AAPL, NVDA", key="mon_search")

    u_tickers = get_universe_tickers(mon_universe)

    with st.spinner(f"Loading live RSI & Exit Price snapshot for all {len(u_tickers)} tickers in {mon_universe}..."):
        bulk_univ_df = fetch_bulk_recent_data(tuple(u_tickers), days_back=420)

        univ_rows = []
        for sym in u_tickers:
            d_sym = extract_symbol_daily(bulk_univ_df, sym, len(u_tickers))
            if d_sym.empty:
                continue
            row_data = compute_ticker_snapshot_row(
                d_sym,
                sym,
                active_rsi_days=mon_b_days,
                custom_exit_rsi_days=mon_custom_exit_days,
                custom_exit_thresh=mon_custom_exit_thresh
            )
            if row_data is not None:
                univ_rows.append(row_data)

    if not univ_rows:
        st.error(f"Could not load market data for {mon_universe}.")
    else:
        full_univ_df = pd.DataFrame(univ_rows)
        custom_col_label = f"Exit: {mon_custom_exit_days}D RSI>{mon_custom_exit_thresh} ($)"

        # Optional Quick-Sort & Filter Bar
        sort_c1, sort_c2, sort_c3 = st.columns([1.6, 1.0, 1.8])
        with sort_c1:
            default_sort_col = st.selectbox(
                "Sort Universe By Column (or click any column header below):",
                options=list(full_univ_df.columns),
                index=list(full_univ_df.columns).index(f"{mon_b_days}D RSI(2)"),
                key="mon_sort_col"
            )
        with sort_c2:
            sort_dir = st.radio("Sort Order", ["Ascending (Low → High)", "Descending (High → Low)"], horizontal=True, key="mon_sort_dir")
        with sort_c3:
            trend_filter_mode = st.radio(
                "Display Scope",
                ["All Tickers in Universe", "Only Above 200D SMA", "Only Oversold (<20 RSI)"],
                horizontal=True,
                key="mon_scope_mode"
            )

        display_univ_df = full_univ_df.copy()

        if mon_search.strip():
            q_tokens = [t.strip().upper() for t in re.split(r'[, \n]+', mon_search) if t.strip()]
            display_univ_df = display_univ_df[display_univ_df["Ticker"].isin(q_tokens)]

        if trend_filter_mode == "Only Above 200D SMA":
            display_univ_df = display_univ_df[display_univ_df["% vs 200 SMA"] > 0]
        elif trend_filter_mode == "Only Oversold (<20 RSI)":
            display_univ_df = display_univ_df[display_univ_df[f"{mon_b_days}D RSI(2)"] <= 20.0]

        asc_bool = ("Ascending" in sort_dir)
        display_univ_df = display_univ_df.sort_values(by=default_sort_col, ascending=asc_bool, na_position='last').reset_index(drop=True)

        # Summary Metrics Banner
        s_m1, s_m2, s_m3, s_m4 = st.columns(4)
        s_m1.metric("Tickers Listed", f"{len(display_univ_df)} / {len(full_univ_df)}")
        s_m2.metric("Oversold (2D RSI ≤ 20)", f"{int((full_univ_df['2D RSI(2)'] <= 20).sum())}")
        s_m3.metric("Oversold (5D RSI ≤ 20)", f"{int((full_univ_df['5D RSI(2)'] <= 20).sum())}")
        s_m4.metric("Above 200-Day SMA", f"{(full_univ_df['% vs 200 SMA'] > 0).mean() * 100:.1f}%")

        st.dataframe(
            display_univ_df,
            column_config=get_snapshot_column_config(custom_col_label),
            hide_index=True,
            use_container_width=True,
            height=620
        )

        st.download_button(
            label=f"📥 Download {mon_universe} RSI & Exit Price Table (CSV)",
            data=display_univ_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{mon_universe.lower().replace(' ', '_')}_exit_prices.csv",
            mime="text/csv"
        )

# ----------------------------------------------------
# TAB 3: CUSTOM WATCHLIST & EXIT PRICES
# ----------------------------------------------------
with tab_watchlist:
    st.subheader("📋 Custom Watchlist — Current RSI Readings & Target Exit Prices")
    st.caption(
        "Enter your personal list of tickers to view their current 1D/2D/3D/5D RSI(2) readings and all exact target exit prices ($). "
        "Every column is sortable by clicking its header."
    )

    w_c1, w_c2, w_c3, w_c4 = st.columns([2.2, 1.0, 1.0, 1.0])
    with w_c1:
        w_input = st.text_input(
            "Watchlist Tickers (comma or space separated)",
            value="SPY, QQQ, IWM, DIA, META, NVDA, AAPL, MSFT, AMZN, GOOGL, TSLA, AMD",
            key="w_tickers_input"
        )
    with w_c2:
        w_bucket_rsi = st.selectbox("Bucket RSI Lookback", ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"], index=0, key="w_b_rsi")
        w_b_days = RSI_DAYS_MAP[w_bucket_rsi]
    with w_c3:
        w_cx_days = st.selectbox("Custom Exit RSI", [1, 2, 3, 5], index=1, format_func=lambda x: f"{x}-Day RSI(2)", key="w_cx_days")
    with w_c4:
        w_cx_thresh = st.selectbox("Custom Exit Threshold", RSI_THRESHOLDS, index=6, format_func=lambda x: f"RSI > {x}", key="w_cx_thresh")

    w_list = list(dict.fromkeys([s.strip().upper().replace('.', '-') for s in re.split(r'[, \n]+', w_input) if s.strip()]))

    if not w_list:
        st.warning("Please enter at least one valid ticker symbol.")
    else:
        with st.spinner(f"Loading RSI readings and Exit Target Prices for {len(w_list)} watchlist ticker(s)..."):
            bulk_w_df = fetch_bulk_recent_data(tuple(w_list), days_back=420)
            w_rows = []
            for sym in w_list:
                d_sym = extract_symbol_daily(bulk_w_df, sym, len(w_list))
                if d_sym.empty:
                    continue
                row_data = compute_ticker_snapshot_row(
                    d_sym,
                    sym,
                    active_rsi_days=w_b_days,
                    custom_exit_rsi_days=w_cx_days,
                    custom_exit_thresh=w_cx_thresh
                )
                if row_data is not None:
                    w_rows.append(row_data)

        if not w_rows:
            st.error("Could not retrieve sufficient trading history for the entered ticker(s).")
        else:
            watch_df = pd.DataFrame(w_rows)
            w_custom_col = f"Exit: {w_cx_days}D RSI>{w_cx_thresh} ($)"
            watch_df = watch_df.sort_values(by=f"{w_b_days}D RSI(2)", ascending=True).reset_index(drop=True)

            st.dataframe(
                watch_df,
                column_config=get_snapshot_column_config(w_custom_col),
                hide_index=True,
                use_container_width=True,
                height=480
            )
            st.download_button(
                label="📥 Download Watchlist Exit Prices (CSV)",
                data=watch_df.to_csv(index=False).encode('utf-8'),
                file_name="watchlist_exit_prices.csv",
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
