
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

st.set_page_config(
    page_title="Multi-Stock Monthly Reversal Backtester",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Portfolio Monthly Reversal Backtester")
st.caption(
    "Rules: $N$ Consecutive Red Months → Green Inside/Lower-High Signal Month → "
    "Buy 1 Tick Above Green High → Stop Loss 1 Tick Below First Red Month's Low."
)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Portfolio & Strategy Settings")

tickers_input = st.sidebar.text_input(
    "Stock Tickers (Up to 10, comma-separated)", 
    value="SPY, QQQ, AAPL, MSFT"
)

start_year = st.sidebar.selectbox(
    "Start Year", 
    options=[1995, 2000, 2005, 2010, 2015, 2020], 
    index=1
)

consec_red_target = st.sidebar.slider(
    "Min Consecutive Red Months", 
    min_value=2, 
    max_value=8, 
    value=3
)

hold_months = st.sidebar.slider(
    "Hold Duration (Months)", 
    min_value=1, 
    max_value=12, 
    value=3
)

# Parse and clean tickers
raw_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
selected_tickers = raw_tickers[:10]  # Cap at 10

if len(raw_tickers) > 10:
    st.sidebar.warning(f"Capped at 10 tickers. Evaluating: {', '.join(selected_tickers)}")


# --- DATA FETCHING WITH CACHE & CUSTOM USER-AGENT ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker_sym, start_yr):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    ticker_obj = yf.Ticker(ticker_sym, session=session)
    monthly_df = ticker_obj.history(start=f"{start_yr}-01-01", interval="1mo")
    daily_df = ticker_obj.history(start=f"{start_yr}-01-01", interval="1d")

    if monthly_df.empty or daily_df.empty:
        monthly_df = yf.download(ticker_sym, start=f"{start_yr}-01-01", interval="1mo", progress=False)
        daily_df = yf.download(ticker_sym, start=f"{start_yr}-01-01", interval="1d", progress=False)

    return monthly_df, daily_df


# --- SINGLE-ASSET BACKTEST LOGIC ---
def backtest_single_stock(ticker_sym, monthly_raw, daily_raw, red_target, hold_len):
    monthly_df = monthly_raw.copy()
    daily_df = daily_raw.copy()

    for df in [monthly_df, daily_df]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    if monthly_df.index.tz is not None:
        monthly_df.index = monthly_df.index.tz_localize(None)
    if daily_df.index.tz is not None:
        daily_df.index = daily_df.index.tz_localize(None)

    monthly_df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    daily_df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

    if len(monthly_df) < red_target + 2 or daily_df.empty:
        return []

    monthly_df['Is_Red'] = monthly_df['Close'] < monthly_df['Open']
    monthly_df['Is_Green'] = monthly_df['Close'] > monthly_df['Open']
    monthly_df['Consec_Red'] = monthly_df['Is_Red'].groupby((~monthly_df['Is_Red']).cumsum()).cumsum()

    trades = []

    for i in range(len(monthly_df)):
        if i < red_target + 1:
            continue

        was_green_signal = bool(monthly_df['Is_Green'].iloc[i - 1])
        actual_red_streak = int(monthly_df['Consec_Red'].iloc[i - 2])
        had_enough_reds = actual_red_streak >= red_target
        lower_high_filter = float(monthly_df['High'].iloc[i - 1]) < float(monthly_df['High'].iloc[i - 2])

        if was_green_signal and had_enough_reds and lower_high_filter:
            first_red_idx = (i - 2) - actual_red_streak + 1

            tick = 0.01
            buy_trigger = float(monthly_df['High'].iloc[i - 1] + tick)
            stop_loss_level = float(monthly_df['Low'].iloc[first_red_idx] - tick)

            exec_month_start = monthly_df.index[i]
            exec_month_end = monthly_df.index[i + 1] if (i + 1 < len(monthly_df)) else daily_df.index[-1]

            execution_days = daily_df[(daily_df.index >= exec_month_start) & (daily_df.index < exec_month_end)]
            triggered_days = execution_days[execution_days['High'] >= buy_trigger]

            if triggered_days.empty:
                continue

            entry_date = triggered_days.index[0]
            entry_day_data = daily_df.loc[entry_date]
            entry_price = max(buy_trigger, float(entry_day_data['Open']))

            exit_month_idx = i + hold_len
            max_hold_date = monthly_df.index[exit_month_idx] if (exit_month_idx < len(monthly_df)) else daily_df.index[-1]

            trade_lifecycle = daily_df[(daily_df.index >= entry_date) & (daily_df.index <= max_hold_date)]

            exited = False
            for current_date, row in trade_lifecycle.iterrows():
                # Same-day execution stop check
                if current_date == entry_date:
                    if float(row['Low']) <= stop_loss_level:
                        exit_price = stop_loss_level
                        trades.append({
                            'Ticker': ticker_sym,
                            'Year': current_date.year,
                            'Entry Date': entry_date.strftime('%Y-%m-%d'),
                            'Exit Date': current_date.strftime('%Y-%m-%d'),
                            'Entry ($)': round(entry_price, 2),
                            'Exit ($)': round(exit_price, 2),
                            'Return (%)': round(((exit_price - entry_price) / entry_price) * 100, 2),
                            'Exit Type': 'Stopped Out (Day 1)'
                        })
                        exited = True
                        break
                    continue

                # Normal daily stop against first red month's low
                if float(row['Low']) <= stop_loss_level:
                    exit_price = min(stop_loss_level, float(row['Open']))
                    trades.append({
                        'Ticker': ticker_sym,
                        'Year': current_date.year,
                        'Entry Date': entry_date.strftime('%Y-%m-%d'),
                        'Exit Date': current_date.strftime('%Y-%m-%d'),
                        'Entry ($)': round(entry_price, 2),
                        'Exit ($)': round(exit_price, 2),
                        'Return (%)': round(((exit_price - entry_price) / entry_price) * 100, 2),
                        'Exit Type': 'Stopped Out (First Red Low)'
                    })
                    exited = True
                    break

                # Target time horizon exit
                if current_date == max_hold_date:
                    exit_price = float(row['Close'])
                    trades.append({
                        'Ticker': ticker_sym,
                        'Year': current_date.year,
                        'Entry Date': entry_date.strftime('%Y-%m-%d'),
                        'Exit Date': current_date.strftime('%Y-%m-%d'),
                        'Entry ($)': round(entry_price, 2),
                        'Exit ($)': round(exit_price, 2),
                        'Return (%)': round(((exit_price - entry_price) / entry_price) * 100, 2),
                        'Exit Type': 'Time Exit'
                    })
                    exited = True
                    break

            if not exited and not trade_lifecycle.empty:
                last_row = trade_lifecycle.iloc[-1]
                trades.append({
                    'Ticker': ticker_sym,
                    'Year': trade_lifecycle.index[-1].year,
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': trade_lifecycle.index[-1].strftime('%Y-%m-%d'),
                    'Entry ($)': round(entry_price, 2),
                    'Exit ($)': round(float(last_row['Close']), 2),
                    'Return (%)': round(((float(last_row['Close']) - entry_price) / entry_price) * 100, 2),
                    'Exit Type': 'End of Data'
                })

    return trades


# --- EXECUTION & DASHBOARD ---
if not selected_tickers:
    st.info("Please enter at least 1 valid stock ticker in the sidebar.")
else:
    all_portfolio_trades = []
    loading_progress = st.progress(0, text="Fetching data and running simulations...")

    for idx, sym in enumerate(selected_tickers):
        loading_progress.progress((idx + 1) / len(selected_tickers), text=f"Analyzing {sym} ({idx+1}/{len(selected_tickers)})...")
        try:
            m_raw, d_raw = fetch_stock_data(sym, start_year)
            if not m_raw.empty and not d_raw.empty:
                stock_trades = backtest_single_stock(sym, m_raw, d_raw, consec_red_target, hold_months)
                all_portfolio_trades.extend(stock_trades)
        except Exception as e:
            st.warning(f"Error evaluating {sym}: {e}")

    loading_progress.empty()

    if all_portfolio_trades:
        df_trades = pd.DataFrame(all_portfolio_trades)

        # 1. Top-Level Portfolio Metrics
        total_trades = len(df_trades)
        win_trades = (df_trades['Return (%)'] > 0).sum()
        win_rate = (win_trades / total_trades) * 100
        avg_trade_ret = df_trades['Return (%)'].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Portfolio Trades", total_trades)
        m2.metric("Overall Win Rate", f"{win_rate:.1f}%")
        m3.metric("Avg Trade Return", f"{avg_trade_ret:.2f}%")
        m4.metric("Tickers Traded", f"{df_trades['Ticker'].nunique()} / {len(selected_tickers)}")

        st.divider()

        # 2. Year-by-Year Performance Breakdown
        st.subheader("📅 Year-by-Year Performance")

        annual_summary = []
        for year, group in df_trades.groupby('Year'):
            y_total = len(group)
            y_wins = (group['Return (%)'] > 0).sum()
            y_losses = y_total - y_wins
            y_winrate = (y_wins / y_total) * 100
            y_avg_ret = group['Return (%)'].mean()
            # Compounded return across all trades closed in that year
            y_compounded = (np.prod(1 + (group['Return (%)'] / 100)) - 1) * 100

            annual_summary.append({
                'Year': int(year),
                'Trades': y_total,
                'Wins': y_wins,
                'Losses': y_losses,
                'Win Rate (%)': round(y_winrate, 1),
                'Avg Trade Ret (%)': round(y_avg_ret, 2),
                'Cumulative Annual Ret (%)': round(y_compounded, 2)
            })

        df_annual = pd.DataFrame(annual_summary).sort_values('Year', ascending=False)
        st.dataframe(df_annual, width="stretch", hide_index=True)

        st.divider()

        # 3. Individual Ticker Breakdown
        st.subheader("📊 Performance by Ticker")
        ticker_summary = []
        for sym, group in df_trades.groupby('Ticker'):
            t_total = len(group)
            t_wins = (group['Return (%)'] > 0).sum()
            t_winrate = (t_wins / t_total) * 100
            t_avg_ret = group['Return (%)'].mean()

            ticker_summary.append({
                'Ticker': sym,
                'Trades': t_total,
                'Win Rate (%)': round(t_winrate, 1),
                'Avg Trade Ret (%)': round(t_avg_ret, 2)
            })

        df_ticker_perf = pd.DataFrame(ticker_summary).sort_values('Trades', ascending=False)
        st.dataframe(df_ticker_perf, width="stretch", hide_index=True)

        st.divider()

        # 4. Filterable Global Trade Log
        st.subheader("📋 Comprehensive Trade Log")
        selected_log_filter = st.selectbox("Filter trade log by Ticker", options=["ALL"] + sorted(list(df_trades['Ticker'].unique())))
        
        if selected_log_filter == "ALL":
            st.dataframe(df_trades.sort_values('Exit Date', ascending=False), width="stretch", hide_index=True)
        else:
            filtered_df = df_trades[df_trades['Ticker'] == selected_log_filter].sort_values('Exit Date', ascending=False)
            st.dataframe(filtered_df, width="stretch", hide_index=True)

    else:
        st.info(
            f"No trades triggered across any of the selected tickers ({', '.join(selected_tickers)}) "
            f"under the current parameters (Streak ≥ {consec_red_target} red months). "
            "Try adjusting the required red months or expanding the ticker list."
        )
