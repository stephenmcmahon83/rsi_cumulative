import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import io
import re

st.set_page_config(page_title="RSI(2) Next-Open & Limit Backtester", layout="wide")

st.title("RSI(2) Quantitative Backtester (Next-Open & Limit-Below-Open Engine)")
st.caption(
    "Execution Model: Entry Signal on Close → Buy at Next Day Open (0%) or Limit X% Below Next Open (1%–7%) | "
    "Exit Signal on Close → Sell Next Day Open"
)

# ==========================================
# CONSTANTS & STRATEGY DEFINITIONS
# ==========================================

DEFAULT_LS_BASKET = [
    "SPY", "QQQ", "GLD", "SLV", "TLT",
    "EWZ", "FXI", "EWJ", "EWG",
    "SOXL", "LABU", "JNUG", "GUSH"
]

BUCKET_LABELS = [
    '<10', '11-20', '21-30', '31-40', '41-50',
    '51-60', '61-70', '71-80', '81-90', '91-100'
]
RSI_THRESHOLDS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
HIGH_CLOSE_WINDOWS = [3, 5, 7, 10]
LIMIT_DISCOUNT_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 7]

EXIT_KEYS = (
    ["SMA5"]
    + [f"HC{x}" for x in HIGH_CLOSE_WINDOWS]
    + [f"RSI_{t}" for t in RSI_THRESHOLDS]
    + ["FIX5", "FIX10"]
)

RSI_DAYS_MAP = {
    "1-Day RSI(2)": 1,
    "2-Day RSI(2)": 2,
    "3-Day RSI(2)": 3,
    "5-Day RSI(2)": 5
}

def format_limit_label(pct):
    return "At Next Day's Open (0%)" if pct == 0 else f"Limit {pct}% Below Next Open"

def get_exit_label_map(rsi_name="5-Day RSI(2)"):
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
# COMPREHENSIVE BUILT-IN UNIVERSE ROSTERS
# ==========================================

DJIA_30_TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
]

MAJOR_ETFS_105 = [
    "SPY", "QQQ", "IWM", "DIA", "MDY", "IJH", "IJR", "VTI", "VOO", "RSP",
    "QQQM", "IVV", "SCHD", "VUG", "VTV", "VIG", "QUAL", "MTUM", "USMV", "SPLV",
    "IWF", "IWD", "IWO", "IWN", "IWP", "IWS", "ARKK", "ARKW",
    "XLK", "XLF", "XLV", "XLE", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "SMH", "SOXX", "IGV", "CIBR", "HACK", "SKYY", "BOTZ", "IBB", "XBI", "IHI",
    "XPH", "KRE", "KBE", "IAI", "KIE", "ITB", "XHB", "VNQ", "IYR", "XRT",
    "RTH", "XOP", "OIH", "AMLP", "XME", "GDX", "GDXJ", "SIL", "COPX", "URA",
    "URNM", "ITA", "XAR", "JETS", "IYT", "XTN", "TAN", "ICLN", "LIT", "PAVE", "GRID",
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TIP", "EMB", "MBB",
    "GLD", "IAU", "SLV", "USO", "UNG", "DBC", "PDBC", "DBA", "UUP", "IBIT", "FBTC", "BITO",
    "EFA", "VEA", "EEM", "VWO", "FXI", "KWEB", "MCHI", "EWJ", "EWZ", "EWY",
    "INDA", "EWG", "EWU", "EWC", "EWA", "EWW", "EWT", "EZU", "FEZ",
    "SOXL", "LABU", "JNUG", "GUSH", "TQQQ", "SPXL", "TNA", "BOIL", "KOLD"
]

NASDAQ_100_TICKERS = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM", "EA", "EXC",
    "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX",
    "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR",
    "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU",
    "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD",
    "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS",
    "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY",
    "XEL", "ZS"
]

SP500_FULL_FALLBACK = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APO",
    "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
    "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB",
    "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX",
    "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD",
    "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI",
    "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB",
    "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTRA",
    "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", "DD", "DE",
    "DECK", "DELL", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR",
    "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN",
    "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES", "ESS",
    "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG",
    "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB",
    "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE",
    "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG",
    "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA",
    "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL",
    "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX",
    "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM",
    "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ",
    "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC",
    "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV",
    "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP",
    "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX",
    "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK",
    "MRNA", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH",
    "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG",
    "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O",
    "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA",
    "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR",
    "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD",
    "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM",
    "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP",
    "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB",
    "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD",
    "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", "T",
    "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX",
    "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN",
    "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA",
    "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC",
    "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA",
    "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB",
    "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS"
]

RUSSELL_SMALLCAP_FALLBACK = [
    "AAON", "AAT", "ABCB", "ABG", "ABM", "ACA", "ACAD", "ACHC", "ACIW", "ACLS",
    "ADEA", "ADMA", "ADNT", "ADUS", "AEIS", "AEO", "AESI", "AGCO", "AGYS", "AHCO",
    "AIN", "AIR", "AIT", "AKR", "AL", "ALEX", "ALG", "ALGT", "ALK", "ALKT",
    "ALRM", "ALTR", "AMBA", "AMBC", "AMED", "AMKR", "AMN", "AMPH", "AMRC", "AMRX",
    "AMSF", "AMWD", "ANDE", "ANF", "ANIP", "AORT", "AOSL", "APAM", "APGE",
    "APLE", "APOG", "APPF", "APPN", "ARCB", "ARCH", "ARCT", "ARES", "ARHS", "ARKO",
    "ARLO", "ARMK", "AROC", "ARR", "ARVN", "ARWR", "ASB", "ASGN", "ASH", "ASIX",
    "ASPN", "ATEN", "ATGE", "ATI", "ATKR", "ATMU", "ATNI", "ATRC", "ATRI",
    "ATRO", "ATSG", "AVA", "AVAV", "AVNS", "AVNT", "AVO", "AVPT", "AVXL", "AWR",
    "AX", "AXGN", "AXNX", "AXSM", "AYI", "AZEK", "AZTA", "B", "BANC", "BANF",
    "BANR", "BCC", "BCML", "BCO", "BCPC", "BCRX", "BDC", "BE", "BEAM", "BECN",
    "BELFB", "BFH", "BFS", "BGC", "BGS", "BHE", "BHLB", "BHVN", "BIGC", "BJRI",
    "BKE", "BKH", "BKU", "BL", "BLBD", "BLD", "BLFS", "BLKB", "BLMN", "BLX",
    "BMI", "BMRC", "BOH", "BOKF", "BOOT", "BOX", "BPMC", "BPOP", "PRVA", "BRBR",
    "BRC", "BRKL", "BRKR", "BRSP", "BRX", "BRZE", "BSIG", "BSM", "BTU", "BTSG",
    "BUSE", "BV", "BWA", "BWB", "BWFG", "BWIN", "BWXT", "BXMT", "BY", "BYD",
    "BYON", "CACI", "CADE", "CAKE", "CAL", "CALM", "CALX", "CAMT", "CAR", "CARG",
    "CARS", "CASY", "CATY", "CBAN", "CBRL", "CBSH", "CBT", "CBU", "CBZ", "CC",
    "CCB", "CCBG", "CCOI", "CCRN", "CCS", "CDE", "CDMO", "CDNA", "CDP", "CEIX",
    "CELH", "CENT", "CENTA", "CENX", "CERS", "CEVA", "CFB", "CFFN", "CHCO", "CHCT",
    "CHE", "CHEF", "CHH", "CHRD", "CHUY", "CHX", "CIEN", "CIM", "CIVB", "CIVI",
    "CLB", "CLBK", "CLDT", "CLFD", "CLH", "CLNE", "CLSK", "CLVT", "CMC", "CMCO",
    "CMP", "CMPR", "CMRE", "CMTG", "CNK", "CNMD", "CNNE", "CNO", "CNOB", "CNS",
    "CNX", "CNXC", "CNXN", "COHU", "COKE", "COLB", "COLL", "COLM", "COMM", "COMP",
    "COOP", "CORT", "CPF", "CPK", "CPRX", "CPS", "CR", "CRAI", "CRBG", "CRC",
    "CRDO", "CRGY", "CRI", "CRK", "CRL", "CRMT", "CRNX", "CROX", "CRS", "CRSR",
    "CRUS", "CRVL", "CSGS", "CSR", "CSTE", "CSTL", "CSTM", "CSV", "CSWI", "CTBI",
    "CTKB", "CTLP", "CTO", "CTOS", "CTRE", "CTS", "CUBI", "CVBF", "CVCO", "CVCY",
    "CVGW", "CVI", "CVLG", "CVLT", "CW", "CWAN", "CWCO", "CWEN", "CWH", "CWK",
    "CWST", "CWT", "CXM", "CXW", "CYRX", "CYTK", "DAN", "DAR", "DCOM",
    "DDS", "DEA", "DEI", "DFH", "DFIN", "DGICA", "DGII", "DHC", "DHIL", "DHT",
    "DIN", "DIOD", "DJCO", "DK", "DKL", "DLB", "DLO", "DMLP", "DMRC", "DNLI",
    "DNOW", "DNUT", "DOCN", "DOCS", "DOLE", "DOMO", "TALO", "TKR", "TXG", "UMBF",
    "URBN", "VSAT", "VVI", "WING", "WNS", "ZETA", "ZWS", "EAT", "EPRT", "EXPO",
    "FCFS", "FN", "FROG", "GKOS", "GTLS", "HAE", "HALO", "HQY", "IDCC", "ITGR",
    "JXN", "KNF", "KNSL", "KNTK", "KRYS", "KYMR", "LSTR", "LTH", "MEDP", "MOG-A",
    "MRC", "NEOG", "NSIT", "NVST", "OII", "ONB", "ORA", "OSCR", "PACS", "PBF",
    "PCVX", "POWI", "PTGX", "QTWO", "RBA", "RHP", "SAIA", "SLAB", "SM", "SPSC",
    "STRL", "TGTX", "TTAN", "TWST"
]

# ==========================================
# 1. OHLCV EXTRACTION & CORE QUANT ENGINE
# ==========================================

OHLCV_NAMES = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}

def normalize_ohlcv_df(df):
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

def prepare_ls_etf_data(raw_df):
    """
    Lightweight preparation for Long/Short Basket ETFs:
    Computes split/dividend-adjusted Open/High/Low/Close, 1D/2D/3D/5D RSI(2), SMA(5), and SMA(200).
    """
    df = normalize_ohlcv_df(raw_df)
    if df.empty or len(df) < 30:
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

    close_s = df['Close_Adj']
    df['RSI2_1d'] = calculate_wilders_rsi(close_s, period=2)
    df['RSI2_2d'] = df['RSI2_1d'].rolling(window=2).mean()
    df['RSI2_3d'] = df['RSI2_1d'].rolling(window=3).mean()
    df['RSI2_5d'] = df['RSI2_1d'].rolling(window=5).mean()

    df['SMA_5'] = close_s.rolling(window=5).mean()
    df['SMA_200'] = close_s.rolling(window=200).mean()
    return df.dropna(subset=['Open_Adj', 'Close_Adj', 'SMA_5', 'RSI2_5d'])

# ==========================================
# 2. LONG / SHORT BASKET BACKTEST ENGINE
# ==========================================

def run_long_short_basket_backtest(
    etf_dfs,
    rsi_days=2,
    initial_capital=100000.0,
    execution_mode="Independent Legs (Always Active)",
    require_pre_exit_side=True,
    max_long_rsi=100.0,
    min_short_rsi=0.0,
    max_hold_days=20,
    entry_limit_pct=0.0
):
    """
    Backtests buying the ETF with the lowest X-day RSI(2) and shorting the ETF with the highest X-day RSI(2).
    Supports Market-on-Open (0%) or Limit X% Better than Next Open:
      - Long Limit Price = Next_Open * (1 - entry_limit_pct / 100); fills if Low[d] <= Limit Price.
      - Short Limit Price = Next_Open * (1 + entry_limit_pct / 100); fills if High[d] >= Limit Price.
    """
    if not etf_dfs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_dates = sorted(list(set.union(*[set(df.index) for df in etf_dfs.values()])))
    if len(all_dates) < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rsi_col = f"RSI2_{rsi_days}d"

    long_eq = float(initial_capital)
    short_eq = float(initial_capital)
    comb_long_cap = float(initial_capital) * 0.5
    comb_short_cap = float(initial_capital) * 0.5

    long_pos = None
    short_pos = None

    trades = []
    equity_records = []

    for idx in range(1, len(all_dates)):
        d = all_dates[idx]
        prev_d = all_dates[idx - 1]

        # --------------------------------------------------
        # STEP A: CHECK EXITS FROM YESTERDAY'S CLOSE (prev_d) -> EXECUTE AT TODAY'S OPEN (d)
        # --------------------------------------------------
        if long_pos is not None:
            t_long = long_pos['ticker']
            df_l = etf_dfs[t_long]
            if prev_d in df_l.index and d in df_l.index and long_pos['entry_date'] <= prev_d:
                long_pos['days_held'] += 1
                prev_row = df_l.loc[prev_d]
                today_open = float(df_l.loc[d, 'Open_Adj'])

                exit_long = False
                exit_reason = ""
                if prev_row['Close_Adj'] > prev_row['SMA_5']:
                    exit_long = True
                    exit_reason = "Close > SMA(5)"
                elif max_hold_days and long_pos['days_held'] >= max_hold_days:
                    exit_long = True
                    exit_reason = f"Max Hold ({max_hold_days}d)"

                if exit_long:
                    ret_pct = (today_open - long_pos['entry_px']) / long_pos['entry_px'] * 100.0
                    long_eq = long_pos['leg_cap'] * (1.0 + ret_pct / 100.0)
                    comb_long_cap = long_pos['comb_cap'] * (1.0 + ret_pct / 100.0)

                    trades.append({
                        "Leg": "LONG",
                        "Ticker": t_long,
                        "Signal Date": long_pos['signal_date'].strftime('%Y-%m-%d'),
                        "Signal RSI": round(long_pos['signal_rsi'], 1),
                        "Entry Date": long_pos['entry_date'].strftime('%Y-%m-%d'),
                        "Next Open ($)": round(long_pos['next_open'], 2),
                        "Entry Fill ($)": round(long_pos['entry_px'], 2),
                        "Exit Date (Open)": d.strftime('%Y-%m-%d'),
                        "Exit Open ($)": round(today_open, 2),
                        "Exit Reason": exit_reason,
                        "Return (%)": round(ret_pct, 2),
                        "Hold (Days)": long_pos['days_held'],
                        "Year": d.year,
                        "Result": "Win" if ret_pct > 0 else "Loss"
                    })
                    long_pos = None

        if short_pos is not None:
            t_short = short_pos['ticker']
            df_s = etf_dfs[t_short]
            if prev_d in df_s.index and d in df_s.index and short_pos['entry_date'] <= prev_d:
                short_pos['days_held'] += 1
                prev_row = df_s.loc[prev_d]
                today_open = float(df_s.loc[d, 'Open_Adj'])

                exit_short = False
                exit_reason = ""
                if prev_row['Close_Adj'] < prev_row['SMA_5']:
                    exit_short = True
                    exit_reason = "Close < SMA(5)"
                elif max_hold_days and short_pos['days_held'] >= max_hold_days:
                    exit_short = True
                    exit_reason = f"Max Hold ({max_hold_days}d)"

                if exit_short:
                    raw_short_ret = (short_pos['entry_px'] - today_open) / short_pos['entry_px'] * 100.0
                    ret_pct = max(-100.0, raw_short_ret)
                    short_eq = short_pos['leg_cap'] * (1.0 + ret_pct / 100.0)
                    comb_short_cap = short_pos['comb_cap'] * (1.0 + ret_pct / 100.0)

                    trades.append({
                        "Leg": "SHORT",
                        "Ticker": t_short,
                        "Signal Date": short_pos['signal_date'].strftime('%Y-%m-%d'),
                        "Signal RSI": round(short_pos['signal_rsi'], 1),
                        "Entry Date": short_pos['entry_date'].strftime('%Y-%m-%d'),
                        "Next Open ($)": round(short_pos['next_open'], 2),
                        "Entry Fill ($)": round(short_pos['entry_px'], 2),
                        "Exit Date (Open)": d.strftime('%Y-%m-%d'),
                        "Exit Open ($)": round(today_open, 2),
                        "Exit Reason": exit_reason,
                        "Return (%)": round(ret_pct, 2),
                        "Hold (Days)": short_pos['days_held'],
                        "Year": d.year,
                        "Result": "Win" if ret_pct > 0 else "Loss"
                    })
                    short_pos = None

        if long_pos is None and short_pos is None:
            total_comb_cash = comb_long_cap + comb_short_cap
            comb_long_cap = total_comb_cash * 0.5
            comb_short_cap = total_comb_cash * 0.5

        # --------------------------------------------------
        # STEP B: RANK ETFs AT YESTERDAY'S CLOSE (prev_d) -> ENTER ON DAY (d)
        # --------------------------------------------------
        can_enter_long = (long_pos is None)
        can_enter_short = (short_pos is None)

        if "Synchronized" in execution_mode:
            if not (long_pos is None and short_pos is None):
                can_enter_long = False
                can_enter_short = False

        if can_enter_long or can_enter_short:
            valid_etfs = []
            for sym, df_sym in etf_dfs.items():
                if prev_d in df_sym.index and d in df_sym.index:
                    p_row = df_sym.loc[prev_d]
                    d_row = df_sym.loc[d]
                    t_open = float(d_row['Open_Adj'])
                    t_low = float(d_row['Low_Adj'])
                    t_high = float(d_row['High_Adj'])
                    r_val = float(p_row[rsi_col])
                    c_val = float(p_row['Close_Adj'])
                    sma5_val = float(p_row['SMA_5'])
                    if pd.notna(r_val) and pd.notna(t_open) and t_open > 0 and pd.notna(sma5_val):
                        valid_etfs.append({
                            "ticker": sym,
                            "rsi": r_val,
                            "close": c_val,
                            "sma5": sma5_val,
                            "next_open": t_open,
                            "next_low": t_low,
                            "next_high": t_high
                        })

            if len(valid_etfs) >= 2:
                valid_etfs.sort(key=lambda x: x['rsi'])

                # 1. Select Lowest RSI(2) ETF for Long Leg
                if can_enter_long:
                    long_candidates = [
                        x for x in valid_etfs
                        if x['rsi'] <= max_long_rsi
                        and (not require_pre_exit_side or x['close'] <= x['sma5'])
                        and (short_pos is None or x['ticker'] != short_pos['ticker'])
                    ]
                    if long_candidates:
                        pick_l = long_candidates[0]
                        target_l_px = pick_l['next_open'] * (1.0 - entry_limit_pct / 100.0)
                        if entry_limit_pct <= 0.0 or pick_l['next_low'] <= target_l_px:
                            long_pos = {
                                "ticker": pick_l['ticker'],
                                "signal_date": prev_d,
                                "signal_rsi": pick_l['rsi'],
                                "entry_date": d,
                                "next_open": pick_l['next_open'],
                                "entry_px": target_l_px,
                                "days_held": 0,
                                "leg_cap": long_eq,
                                "comb_cap": comb_long_cap
                            }

                # 2. Select Highest RSI(2) ETF for Short Leg
                if can_enter_short:
                    short_candidates = [
                        x for x in reversed(valid_etfs)
                        if x['rsi'] >= min_short_rsi
                        and (not require_pre_exit_side or x['close'] >= x['sma5'])
                        and (long_pos is None or x['ticker'] != long_pos['ticker'])
                    ]
                    if short_candidates:
                        pick_s = short_candidates[0]
                        target_s_px = pick_s['next_open'] * (1.0 + entry_limit_pct / 100.0)
                        if entry_limit_pct <= 0.0 or pick_s['next_high'] >= target_s_px:
                            short_pos = {
                                "ticker": pick_s['ticker'],
                                "signal_date": prev_d,
                                "signal_rsi": pick_s['rsi'],
                                "entry_date": d,
                                "next_open": pick_s['next_open'],
                                "entry_px": target_s_px,
                                "days_held": 0,
                                "leg_cap": short_eq,
                                "comb_cap": comb_short_cap
                            }

        # --------------------------------------------------
        # STEP C: MARK-TO-MARKET EQUITY AT CLOSE OF DAY (d)
        # --------------------------------------------------
        if long_pos is not None and d in etf_dfs[long_pos['ticker']].index:
            cur_c = float(etf_dfs[long_pos['ticker']].loc[d, 'Close_Adj'])
            l_ret = (cur_c - long_pos['entry_px']) / long_pos['entry_px']
            mtm_long_only = long_pos['leg_cap'] * (1.0 + l_ret)
            mtm_comb_long = long_pos['comb_cap'] * (1.0 + l_ret)
        else:
            mtm_long_only = long_eq
            mtm_comb_long = comb_long_cap

        if short_pos is not None and d in etf_dfs[short_pos['ticker']].index:
            cur_c = float(etf_dfs[short_pos['ticker']].loc[d, 'Close_Adj'])
            s_ret = max(-1.0, (short_pos['entry_px'] - cur_c) / short_pos['entry_px'])
            mtm_short_only = short_pos['leg_cap'] * (1.0 + s_ret)
            mtm_comb_short = short_pos['comb_cap'] * (1.0 + s_ret)
        else:
            mtm_short_only = short_eq
            mtm_comb_short = comb_short_cap

        equity_records.append({
            "Date": d,
            "Combined_Equity": mtm_comb_long + mtm_comb_short,
            "Long_Only_Equity": mtm_long_only,
            "Short_Only_Equity": mtm_short_only
        })

    equity_df = pd.DataFrame(equity_records).set_index("Date")
    trades_df = pd.DataFrame(trades)

    if equity_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    equity_df['Year'] = equity_df.index.year
    yearly_rows = []

    for yr, grp in equity_df.groupby('Year'):
        c_ret = ((grp['Combined_Equity'].iloc[-1] / grp['Combined_Equity'].iloc[0]) - 1.0) * 100.0
        l_ret = ((grp['Long_Only_Equity'].iloc[-1] / grp['Long_Only_Equity'].iloc[0]) - 1.0) * 100.0
        s_ret = ((grp['Short_Only_Equity'].iloc[-1] / grp['Short_Only_Equity'].iloc[0]) - 1.0) * 100.0

        c_dd = ((grp['Combined_Equity'] - grp['Combined_Equity'].cummax()) / grp['Combined_Equity'].cummax()).min() * 100.0

        yr_tr = trades_df[trades_df['Year'] == yr] if not trades_df.empty else pd.DataFrame()
        yr_l = yr_tr[yr_tr['Leg'] == 'LONG'] if not yr_tr.empty else pd.DataFrame()
        yr_s = yr_tr[yr_tr['Leg'] == 'SHORT'] if not yr_tr.empty else pd.DataFrame()

        n_tr = len(yr_tr)
        wr_all = (yr_tr['Return (%)'] > 0).mean() * 100.0 if n_tr > 0 else 0.0
        wr_l = (yr_l['Return (%)'] > 0).mean() * 100.0 if len(yr_l) > 0 else 0.0
        wr_s = (yr_s['Return (%)'] > 0).mean() * 100.0 if len(yr_s) > 0 else 0.0

        gains = yr_tr[yr_tr['Return (%)'] > 0]['Return (%)'].sum() if n_tr > 0 else 0.0
        losses = abs(yr_tr[yr_tr['Return (%)'] < 0]['Return (%)'].sum()) if n_tr > 0 else 0.0
        pf = round(gains / losses, 2) if losses > 0 else (99.0 if gains > 0 else np.nan)

        yearly_rows.append({
            "Year": str(yr),
            "Total Trades": int(n_tr),
            "Long Trades": int(len(yr_l)),
            "Short Trades": int(len(yr_s)),
            "Long Win Rate (%)": round(wr_l, 1),
            "Short Win Rate (%)": round(wr_s, 1),
            "Combined Win Rate (%)": round(wr_all, 1),
            "Profit Factor": pf,
            "Long-Only Return (%)": round(l_ret, 2),
            "Short-Only Return (%)": round(s_ret, 2),
            "Combined L/S Return (%)": round(c_ret, 2),
            "Combined Max DD (%)": round(c_dd, 1)
        })

    tot_c_ret = ((equity_df['Combined_Equity'].iloc[-1] / initial_capital) - 1.0) * 100.0
    tot_l_ret = ((equity_df['Long_Only_Equity'].iloc[-1] / initial_capital) - 1.0) * 100.0
    tot_s_ret = ((equity_df['Short_Only_Equity'].iloc[-1] / initial_capital) - 1.0) * 100.0
    tot_c_dd = ((equity_df['Combined_Equity'] - equity_df['Combined_Equity'].cummax()) / equity_df['Combined_Equity'].cummax()).min() * 100.0

    if not trades_df.empty:
        all_l = trades_df[trades_df['Leg'] == 'LONG']
        all_s = trades_df[trades_df['Leg'] == 'SHORT']
        wr_all = (trades_df['Return (%)'] > 0).mean() * 100.0
        wr_l = (all_l['Return (%)'] > 0).mean() * 100.0 if len(all_l) > 0 else 0.0
        wr_s = (all_s['Return (%)'] > 0).mean() * 100.0 if len(all_s) > 0 else 0.0
        g_all = trades_df[trades_df['Return (%)'] > 0]['Return (%)'].sum()
        l_all = abs(trades_df[trades_df['Return (%)'] < 0]['Return (%)'].sum())
        pf_all = round(g_all / l_all, 2) if l_all > 0 else np.nan
    else:
        all_l, all_s = pd.DataFrame(), pd.DataFrame()
        wr_all, wr_l, wr_s, pf_all = 0.0, 0.0, 0.0, np.nan

    yearly_rows.append({
        "Year": "All Years",
        "Total Trades": int(len(trades_df)),
        "Long Trades": int(len(all_l)),
        "Short Trades": int(len(all_s)),
        "Long Win Rate (%)": round(wr_l, 1),
        "Short Win Rate (%)": round(wr_s, 1),
        "Combined Win Rate (%)": round(wr_all, 1),
        "Profit Factor": pf_all,
        "Long-Only Return (%)": round(tot_l_ret, 2),
        "Short-Only Return (%)": round(tot_s_ret, 2),
        "Combined L/S Return (%)": round(tot_c_ret, 2),
        "Combined Max DD (%)": round(tot_c_dd, 1)
    })

    yearly_df = pd.DataFrame(yearly_rows).set_index("Year")

    ticker_attr_rows = []
    if not trades_df.empty:
        for sym in sorted(etf_dfs.keys()):
            sub_t = trades_df[trades_df['Ticker'] == sym]
            sub_l = sub_t[sub_t['Leg'] == 'LONG']
            sub_s = sub_t[sub_t['Leg'] == 'SHORT']

            l_cnt = len(sub_l)
            s_cnt = len(sub_s)
            l_wr = (sub_l['Return (%)'] > 0).mean() * 100.0 if l_cnt > 0 else 0.0
            s_wr = (sub_s['Return (%)'] > 0).mean() * 100.0 if s_cnt > 0 else 0.0
            l_avg = sub_l['Return (%)'].mean() if l_cnt > 0 else 0.0
            s_avg = sub_s['Return (%)'].mean() if s_cnt > 0 else 0.0
            tot_avg = sub_t['Return (%)'].mean() if len(sub_t) > 0 else 0.0

            ticker_attr_rows.append({
                "Ticker": sym,
                "Total Trades": int(len(sub_t)),
                "Long Trades": int(l_cnt),
                "Long Win Rate (%)": round(l_wr, 1),
                "Long Avg Ret (%)": round(l_avg, 2),
                "Short Trades": int(s_cnt),
                "Short Win Rate (%)": round(s_wr, 1),
                "Short Avg Ret (%)": round(s_avg, 2),
                "Overall Avg Ret (%)": round(tot_avg, 2)
            })
    ticker_attr_df = pd.DataFrame(ticker_attr_rows).set_index("Ticker") if ticker_attr_rows else pd.DataFrame()

    return yearly_df, equity_df, trades_df, ticker_attr_df

# ==========================================
# 3. SINGLE-TICKER & MONITOR HELPER FUNCTIONS
# ==========================================

def solve_next_close_for_rsi2(last_close, avg_gain, avg_loss, current_rsi, target_rsi):
    if pd.isna(last_close) or pd.isna(avg_gain) or pd.isna(avg_loss):
        return np.nan
    if target_rsi <= 0.0 or target_rsi >= 100.0:
        return np.nan

    rs_target = target_rsi / (100.0 - target_rsi)

    if target_rsi > current_rsi:
        req_gain = rs_target * avg_loss - avg_gain
        return max(0.01, float(last_close + max(0.0, req_gain)))
    else:
        if rs_target <= 0:
            return np.nan
        req_loss = (avg_gain / rs_target) - avg_loss
        return max(0.01, float(last_close - max(0.0, req_loss)))

def solve_next_close_for_nday_rsi2(last_close, avg_gain, avg_loss, recent_rsi_list, n_days, target_avg_rsi):
    if n_days == 1:
        return solve_next_close_for_rsi2(last_close, avg_gain, avg_loss, recent_rsi_list[-1], target_avg_rsi)

    prior_sum = sum(recent_rsi_list[-(n_days - 1):])
    req_1d_rsi = (n_days * target_avg_rsi) - prior_sum

    if req_1d_rsi <= 0.1:
        return solve_next_close_for_rsi2(last_close, avg_gain, avg_loss, recent_rsi_list[-1], 0.5)
    elif req_1d_rsi >= 99.9:
        return np.nan
    else:
        return solve_next_close_for_rsi2(last_close, avg_gain, avg_loss, recent_rsi_list[-1], req_1d_rsi)

def compute_ticker_snapshot_row(raw_df, sym, active_rsi_days=2, custom_exit_rsi_days=2, custom_exit_thresh=70):
    df = normalize_ohlcv_df(raw_df)
    if len(df) < 200:
        return None

    if 'Adj Close' in df.columns and 'Close' in df.columns:
        close_s = df['Adj Close'].dropna()
    else:
        close_s = df['Close'].dropna()

    if len(close_s) < 200:
        return None

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
        "Exit: 5D SMA ($)": st.column_config.NumberColumn("Exit: 5D SMA ($)", format="$%.2f"),
        "Next Close > SMA5 ($)": st.column_config.NumberColumn("Next Close > SMA5 ($)", format="$%.2f"),
        "% to 5D SMA": st.column_config.NumberColumn("% to 5D SMA", format="%+.2f%%"),
        "Exit: 3D High ($)": st.column_config.NumberColumn("Exit: 3D High ($)", format="$%.2f"),
        "Exit: 5D High ($)": st.column_config.NumberColumn("Exit: 5D High ($)", format="$%.2f"),
        "Exit: 7D High ($)": st.column_config.NumberColumn("Exit: 7D High ($)", format="$%.2f"),
        "Exit: 10D High ($)": st.column_config.NumberColumn("Exit: 10D High ($)", format="$%.2f"),
        "Exit: 1D RSI>50 ($)": st.column_config.NumberColumn("Exit: 1D RSI>50 ($)", format="$%.2f"),
        "Exit: 1D RSI>70 ($)": st.column_config.NumberColumn("Exit: 1D RSI>70 ($)", format="$%.2f"),
        "Exit: 1D RSI>80 ($)": st.column_config.NumberColumn("Exit: 1D RSI>80 ($)", format="$%.2f"),
        "Exit: 1D RSI>90 ($)": st.column_config.NumberColumn("Exit: 1D RSI>90 ($)", format="$%.2f"),
        custom_col_name: st.column_config.NumberColumn(custom_col_name, format="$%.2f"),
        "200D SMA ($)": st.column_config.NumberColumn("200D SMA ($)", format="$%.2f"),
        "% vs 200 SMA": st.column_config.NumberColumn("% vs 200 SMA", format="%+.2f%%")
    }

def prepare_stock_data(raw_df, rsi_days=5, max_hold=20, entry_discount_pct=0.0):
    """
    Computes adjusted OHLC, N-day RSI(2) buckets, and Next-Open / Limit-Below-Open
    forward trade outcomes for all exit strategies.
    - When entry_discount_pct == 0: enters unconditionally at Open[i+1].
    - When entry_discount_pct > 0: places a limit order at Open[i+1] * (1 - entry_discount_pct/100)
      and fills if and only if Low[i+1] <= Limit Price.
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

    df['RSI2_1d'] = calculate_wilders_rsi(close_s, period=2)
    df['RSI2_2d'] = df['RSI2_1d'].rolling(window=2).mean()
    df['RSI2_3d'] = df['RSI2_1d'].rolling(window=3).mean()
    df['RSI2_5d'] = df['RSI2_1d'].rolling(window=5).mean()

    active_rsi_col = f'RSI2_{rsi_days}d'
    df['Active_RSI'] = df[active_rsi_col]

    df['SMA_5'] = close_s.rolling(window=5).mean()
    df['SMA_200'] = close_s.rolling(window=200).mean()
    df['Above_200SMA'] = close_s > df['SMA_200']

    for x in HIGH_CLOSE_WINDOWS:
        df[f'HighClose_{x}d'] = close_s.rolling(window=x).max()

    df['Prior_5d'] = (close_s - close_s.shift(5)) / close_s.shift(5) * 100
    df['Prior_10d'] = (close_s - close_s.shift(10)) / close_s.shift(10) * 100

    bins = [-0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.001]
    df['Bucket'] = pd.cut(df['Active_RSI'], bins=bins, labels=BUCKET_LABELS)

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
    entry_px_arr = np.full(n, np.nan)
    filled_arr = np.zeros(n, dtype=bool)

    discount_mult = 1.0 - (float(entry_discount_pct) / 100.0)

    for i in range(n - 2):
        ent_idx = i + 1
        next_open_px = o_vals[ent_idx]
        next_low_px = l_vals[ent_idx]
        if np.isnan(next_open_px) or next_open_px <= 0 or np.isnan(next_low_px):
            continue

        target_entry_px = next_open_px * discount_mult

        # Check if limit order at X% below Open[i+1] fills during Day i+1
        if entry_discount_pct > 0.0 and next_low_px > target_entry_px:
            continue

        entry_px = target_entry_px
        entry_px_arr[i] = entry_px
        filled_arr[i] = True

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

        for k in range(1, max_k + 1):
            if c_vals[i + k] > sma5_vals[i + k]:
                record_exit("SMA5", k)
                break
        else:
            if i + max_hold + 1 < n:
                record_exit("SMA5", max_hold)

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

        if i + 6 < n:
            record_exit("FIX5", 5)
        if i + 11 < n:
            record_exit("FIX10", 10)

    new_cols = {
        'Entry_Px': entry_px_arr,
        'Limit_Filled': filled_arr
    }
    for k in EXIT_KEYS:
        new_cols[f'Ret_{k}'] = ret_arrays[k]
        new_cols[f'Days_{k}'] = day_arrays[k]
        new_cols[f'MAE_{k}'] = mae_arrays[k]
        new_cols[f'ExitIdx_{k}'] = exit_idx_arrays[k]

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df

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

def filter_signal_rows(df, require_uptrend=True, fresh_only=False):
    # Exclude the final row of df if it doesn't have at least 1 forward trading bar
    valid_window = df.iloc[:-2].copy() if len(df) > 2 else df.copy()
    data = valid_window.dropna(subset=['Bucket']).copy()
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
        sig_count = len(sub)
        n, wr, avg_r, pf, days, mae = summarize_series(
            sub[f'Ret_{exit_key}'], sub[f'Days_{exit_key}'], sub[f'MAE_{exit_key}']
        )
        fill_rate = round((n / sig_count) * 100.0, 1) if sig_count > 0 else 0.0
        rows.append({
            "Entry Bucket": b,
            "Signals": int(sig_count),
            "Filled Trades": n,
            "Fill Rate (%)": fill_rate,
            "Win Rate (%)": wr,
            "Avg Return (%)": avg_r,
            "Profit Factor": pf,
            "Avg Hold (Days)": days,
            "Avg Drawdown (%)": mae
        })
    return pd.DataFrame(rows).set_index("Entry Bucket")

def build_limit_offset_comparison_table(raw_df, rsi_days, selected_buckets, exit_key, require_uptrend=True, fresh_only=False):
    """
    Compares entering at Next Day's Open (0%) vs. Limit 1%, 2%, 3%, 4%, 5%, 6%, and 7% Below Next Open
    for the active Entry Bucket(s) and Exit Rule.
    """
    rows = []
    for pct in LIMIT_DISCOUNT_OPTIONS:
        prep = prepare_stock_data(raw_df, rsi_days=rsi_days, entry_discount_pct=float(pct))
        filt = filter_signal_rows(prep, require_uptrend=require_uptrend, fresh_only=fresh_only)
        sub = filt[filt['Bucket'].astype(str).isin(selected_buckets)]
        sig_count = len(sub)
        n, wr, avg_r, pf, days, mae = summarize_series(
            sub[f'Ret_{exit_key}'], sub[f'Days_{exit_key}'], sub[f'MAE_{exit_key}']
        )
        fill_rate = round((n / sig_count) * 100.0, 1) if sig_count > 0 else 0.0
        valid_rets = sub[f'Ret_{exit_key}'].dropna()
        comp_ret = ((1.0 + valid_rets / 100.0).prod() - 1.0) * 100.0 if n > 0 else 0.0

        rows.append({
            "Entry Execution Level": format_limit_label(pct),
            "Signals": int(sig_count),
            "Filled Trades": n,
            "Fill Rate (%)": fill_rate,
            "Win Rate (%)": wr,
            "Avg Return (%)": avg_r,
            "Profit Factor": pf,
            "Compounded Return (%)": round(comp_ret, 2),
            "Avg Hold (Days)": days,
            "Avg Drawdown (%)": mae
        })
    return pd.DataFrame(rows).set_index("Entry Execution Level")

def build_exit_comparison_table(filtered_df, selected_buckets, exit_label_map):
    sub = filtered_df[filtered_df['Bucket'].astype(str).isin(selected_buckets)]
    sig_count = len(sub)
    rows = []
    for k in EXIT_KEYS:
        n, wr, avg_r, pf, days, mae = summarize_series(
            sub[f'Ret_{k}'], sub[f'Days_{k}'], sub[f'MAE_{k}']
        )
        fill_rate = round((n / sig_count) * 100.0, 1) if sig_count > 0 else 0.0
        rows.append({
            "Exit Strategy (Next Open)": exit_label_map[k],
            "Signals": int(sig_count),
            "Filled Trades": n,
            "Fill Rate (%)": fill_rate,
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
        row_dict = {
            "Entry Bucket": b,
            "Filled Trades": int(sub['Ret_SMA5'].dropna().shape[0])
        }
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
    sub_all_sigs = filtered_df[filtered_df['Bucket'].astype(str).isin(selected_buckets)].copy()
    sub = sub_all_sigs.dropna(subset=[f'Ret_{exit_key}']).copy()
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

        entry_fill_px = float(row['Entry_Px']) if pd.notna(row['Entry_Px']) else float(opens[ent_loc])

        trade_rows.append({
            "Signal Date (Close)": sig_date.strftime('%Y-%m-%d'),
            "Entry Bucket": str(row['Bucket']),
            "Signal RSI": round(row['Active_RSI'], 1),
            "Entry Date": dates[ent_loc].strftime('%Y-%m-%d'),
            "Next Open ($)": round(opens[ent_loc], 2),
            "Entry Fill ($)": round(entry_fill_px, 2),
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

    sub_all_sigs['Year'] = sub_all_sigs.index.year
    sigs_by_year = sub_all_sigs.groupby('Year').size().to_dict()

    yearly_rows = []
    for yr, grp in trades_df.groupby('Year'):
        ret_s = grp['Return (%)']
        n_filled = len(grp)
        n_sigs = sigs_by_year.get(yr, n_filled)
        gains = ret_s[ret_s > 0].sum()
        losses = abs(ret_s[ret_s < 0].sum())
        pf = round(gains / losses, 2) if losses > 0 else (99.0 if gains > 0 else np.nan)
        comp_ret = ((1 + ret_s / 100.0).prod() - 1) * 100.0
        yearly_rows.append({
            "Year": str(yr),
            "Signals": int(n_sigs),
            "Filled Trades": int(n_filled),
            "Fill Rate (%)": round((n_filled / n_sigs) * 100.0, 1) if n_sigs > 0 else 0.0,
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
    tot_sigs = len(sub_all_sigs)
    tot_filled = len(trades_df)

    yearly_rows.append({
        "Year": "All Years",
        "Signals": int(tot_sigs),
        "Filled Trades": int(tot_filled),
        "Fill Rate (%)": round((tot_filled / tot_sigs) * 100.0, 1) if tot_sigs > 0 else 0.0,
        "Win Rate (%)": round((all_ret > 0).mean() * 100.0, 1),
        "Avg Return (%)": round(all_ret.mean(), 2),
        "Profit Factor": all_pf,
        "Compounded Return (%)": round(all_comp, 2),
        "Avg Hold (Days)": round(trades_df['Hold (Days)'].mean(), 1)
    })

    return pd.DataFrame(yearly_rows).set_index("Year"), trades_df

# ==========================================
# 4. MULTI-STOCK LONG-ONLY PORTFOLIO ENGINE
# ==========================================

def run_portfolio_backtest_next_open(
    prepared_dfs, selected_buckets, exit_key, exit_label,
    max_positions=5, initial_capital=100000.0,
    require_uptrend=True, fresh_entry_only=True,
    entry_discount_pct=0.0
):
    all_dates = sorted(list(set.union(*[set(df.index) for df in prepared_dfs.values()])))
    if len(all_dates) < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cash = float(initial_capital)
    open_positions = {}
    trade_log = []
    daily_equity_records = []
    discount_mult = 1.0 - (float(entry_discount_pct) / 100.0)

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
                    "Entry Date": pos['entry_date'].strftime('%Y-%m-%d'),
                    "Next Open ($)": round(pos['next_open'], 2),
                    "Entry Fill ($)": round(pos['entry_price'], 2),
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

                today_open = float(df.loc[d, 'Open_Adj'])
                today_low = float(df.loc[d, 'Low_Adj'])
                if pd.isna(today_open) or today_open <= 0 or pd.isna(today_low):
                    continue

                limit_px = today_open * discount_mult
                filled_today = (entry_discount_pct <= 0.0) or (today_low <= limit_px)

                candidates.append({
                    "ticker": ticker,
                    "next_open": today_open,
                    "entry_px": limit_px,
                    "filled": filled_today,
                    "rsi_val": prev_row['Active_RSI']
                })

            candidates.sort(key=lambda x: x['rsi_val'])

            current_equity = cash
            for t_open, p_open in open_positions.items():
                df_o = prepared_dfs[t_open]
                px = df_o.loc[d, 'Open_Adj'] if d in df_o.index else p_open['entry_price']
                current_equity += p_open['shares'] * px

            target_slot_size = current_equity / max_positions

            # Place limit orders on top available_slots candidates; fill those whose Low reached Limit Price
            for cand in candidates[:available_slots]:
                if not cand['filled']:
                    continue
                invest_amt = min(cash, target_slot_size)
                if invest_amt >= 250.0:
                    shares = invest_amt / cand['entry_px']
                    cash -= invest_amt
                    open_positions[cand['ticker']] = {
                        'signal_date': prev_d,
                        'entry_date': d,
                        'next_open': cand['next_open'],
                        'entry_price': cand['entry_px'],
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
# 5. ZERO-LXML UNIVERSE FETCHERS & CACHING
# ==========================================

def parse_wiki_tickers_regex(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        tables = re.findall(r'<table[^>]*class="[^"]*wikitable sortable[^"]*"[^>]*>(.*?)</table>', resp.text, re.DOTALL)
        for tbl in tables:
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tbl, re.DOTALL)
            if len(rows) < 25:
                continue
            header_cells = [
                re.sub(r'<[^>]+>', '', c).strip()
                for c in re.findall(r'<th[^>]*>(.*?)</th>', rows[0], re.DOTALL)
            ]
            sym_idx = None
            for idx, h in enumerate(header_cells):
                if h.lower() in ('symbol', 'ticker', 'ticker symbol'):
                    sym_idx = idx
                    break
            if sym_idx is None:
                continue
            tickers = []
            for r in rows[1:]:
                cols = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.DOTALL)
                if len(cols) > sym_idx:
                    raw_sym = re.sub(r'<[^>]+>', '', cols[sym_idx]).strip().upper().replace('.', '-')
                    if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', raw_sym):
                        tickers.append(raw_sym)
            if len(tickers) >= 25:
                return list(dict.fromkeys(tickers))
    except Exception:
        pass
    return []

def fetch_ishares_russell2000():
    url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.splitlines()
            start_idx = None
            for i, line in enumerate(lines[:30]):
                if line.startswith("Ticker,") or '"Ticker"' in line:
                    start_idx = i
                    break
            if start_idx is not None:
                csv_df = pd.read_csv(io.StringIO("\n".join(lines[start_idx:])))
                if 'Ticker' in csv_df.columns:
                    if 'Asset Class' in csv_df.columns:
                        csv_df = csv_df[csv_df['Asset Class'].astype(str).str.contains('Equity', case=False, na=False)]
                    raw_syms = csv_df['Ticker'].dropna().astype(str).tolist()
                    cleaned = []
                    for s in raw_syms:
                        s_clean = s.strip().upper().replace('.', '-').replace(' ', '-')
                        if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', s_clean) and s_clean not in ('-', 'USD'):
                            cleaned.append(s_clean)
                    if len(cleaned) >= 500:
                        return list(dict.fromkeys(cleaned))
    except Exception:
        pass
    return []

@st.cache_data(show_spinner=False, ttl=86400)
def get_universe_tickers(universe_name):
    if universe_name == "S&P 500":
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
            df_sp = pd.read_csv(url)
            if 'Symbol' in df_sp.columns and len(df_sp) >= 490:
                syms = [s.strip().upper().replace('.', '-') for s in df_sp['Symbol'].dropna().astype(str)]
                return list(dict.fromkeys(syms + SP500_FULL_FALLBACK))
        except Exception:
            pass
        wiki_sp = parse_wiki_tickers_regex("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if len(wiki_sp) >= 490:
            return list(dict.fromkeys(wiki_sp + SP500_FULL_FALLBACK))
        return list(dict.fromkeys(SP500_FULL_FALLBACK))

    elif universe_name == "Nasdaq 100":
        wiki_nq = parse_wiki_tickers_regex("https://en.wikipedia.org/wiki/Nasdaq-100")
        if len(wiki_nq) >= 95:
            return list(dict.fromkeys(wiki_nq + NASDAQ_100_TICKERS))
        return list(dict.fromkeys(NASDAQ_100_TICKERS))

    elif universe_name == "Dow Jones 30":
        wiki_dj = parse_wiki_tickers_regex("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        if len(wiki_dj) == 30:
            return wiki_dj
        return list(dict.fromkeys(DJIA_30_TICKERS))

    elif "Russell 2000" in universe_name:
        iwm_tickers = fetch_ishares_russell2000()
        if len(iwm_tickers) >= 500:
            return iwm_tickers
        wiki_sc = parse_wiki_tickers_regex("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
        return list(dict.fromkeys(wiki_sc + RUSSELL_SMALLCAP_FALLBACK))

    elif "Major ETFs" in universe_name:
        return list(dict.fromkeys(MAJOR_ETFS_105))

    return []

@st.cache_data(show_spinner=False)
def fetch_daily_data(symbol, start, end):
    raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    df = normalize_ohlcv_df(raw)
    return df if not df.empty else None

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_chunk_recent_data(tickers_tuple, days_back=340):
    t_list = list(tickers_tuple)
    if not t_list:
        return pd.DataFrame()
    s_d = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    e_d = datetime.now().strftime('%Y-%m-%d')
    return yf.download(
        t_list,
        start=s_d,
        end=e_d,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True
    )

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_bulk_history_cached(tickers_tuple, start_str, end_str):
    t_list = list(tickers_tuple)
    if not t_list:
        return pd.DataFrame()
    return yf.download(
        t_list,
        start=start_str,
        end=end_str,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True
    )

def load_universe_snapshots_batched(tickers_list, active_rsi_days, custom_exit_days, custom_exit_thresh, batch_size=125):
    total = len(tickers_list)
    if total == 0:
        return []

    rows = []
    chunks = [tickers_list[i:i + batch_size] for i in range(0, total, batch_size)]

    prog_bar = st.progress(0) if total > batch_size else None
    status_txt = st.empty() if total > batch_size else None

    for idx, chunk in enumerate(chunks):
        if status_txt is not None:
            status_txt.caption(
                f"Loading batch {idx + 1} of {len(chunks)} ({min((idx + 1) * batch_size, total)} / {total} tickers)..."
            )
        bulk_chunk = fetch_chunk_recent_data(tuple(chunk), days_back=340)
        for sym in chunk:
            d_sym = extract_symbol_daily(bulk_chunk, sym, len(chunk))
            if d_sym.empty:
                continue
            r_data = compute_ticker_snapshot_row(
                d_sym,
                sym,
                active_rsi_days=active_rsi_days,
                custom_exit_rsi_days=custom_exit_days,
                custom_exit_thresh=custom_exit_thresh
            )
            if r_data is not None:
                rows.append(r_data)

        if prog_bar is not None:
            prog_bar.progress((idx + 1) / len(chunks))

    if prog_bar is not None:
        prog_bar.empty()
    if status_txt is not None:
        status_txt.empty()

    return rows

# ==========================================
# 6. INTERFACE TABS (5 SEPARATE TABS)
# ==========================================

tab_ls, tab_study, tab_screener, tab_watchlist, tab_portfolio = st.tabs([
    "⚖️ Long/Short RSI(2) Basket",
    "📊 Single Ticker Study",
    "🌐 Universe Monitor & Exit Prices",
    "📋 Custom Watchlist & Exit Prices",
    "📁 Multi-Stock Portfolio Backtester"
])

# ----------------------------------------------------
# TAB 1: LONG/SHORT EXTREME RSI(2) BASKET BACKTESTER
# ----------------------------------------------------
with tab_ls:
    st.subheader("⚖️ Long Lowest RSI(2) / Short Highest RSI(2) Basket Backtester")
    st.caption(
        "Strategy Rules: Rank all basket ETFs at the Close → Buy the ETF with the Lowest X-Day RSI(2) (Exit when Close > 5D SMA) | "
        "Short the ETF with the Highest X-Day RSI(2) (Cover when Close < 5D SMA)."
    )

    ls_c1, ls_c2, ls_c3, ls_c4 = st.columns([2.0, 1.0, 1.2, 0.8])
    with ls_c1:
        ls_tickers_input = st.text_input(
            "ETF Basket Tickers (comma or space separated)",
            value=", ".join(DEFAULT_LS_BASKET),
            key="ls_basket_input"
        )
    with ls_c2:
        ls_rsi_choice = st.selectbox(
            "Primary X-Day RSI(2)",
            ["1-Day RSI(2)", "2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"],
            index=1,
            key="ls_rsi_lookback"
        )
        ls_rsi_days = RSI_DAYS_MAP[ls_rsi_choice]
    with ls_c3:
        ls_limit_pct = st.selectbox(
            "Entry Execution (Open vs. Limit)",
            LIMIT_DISCOUNT_OPTIONS,
            index=0,
            format_func=lambda x: "At Next Open (0%)" if x == 0 else f"Limit {x}% Better than Open",
            help="0% enters at Next Open. 1%–7% places a Limit Buy X% below Next Open for Longs and a Limit Short X% above Next Open for Shorts.",
            key="ls_limit_pct"
        )
    with ls_c4:
        ls_years = st.number_input("History (Years)", min_value=2, max_value=20, value=10, key="ls_years")

    ls_c5, ls_c6, ls_c7, ls_c8 = st.columns([1.4, 1.1, 1.1, 1.0])
    with ls_c5:
        ls_exec_mode = st.selectbox(
            "Leg Execution Mode",
            [
                "Independent Legs (Re-enter each leg as soon as it exits)",
                "Synchronized Pair (Wait until both legs exit before next pair)"
            ],
            index=0,
            key="ls_exec_mode"
        )
    with ls_c6:
        ls_max_long_rsi = st.slider(
            "Max RSI(2) Allowed for Long Entry",
            min_value=10, max_value=100, value=100, step=5
        )
    with ls_c7:
        ls_min_short_rsi = st.slider(
            "Min RSI(2) Required for Short Entry",
            min_value=0, max_value=90, value=0, step=5
        )
    with ls_c8:
        ls_exclude_3x = st.checkbox(
            "Exclude 3x ETFs (SOXL, LABU, JNUG, GUSH)",
            value=False
        )
        ls_pre_exit_check = st.checkbox(
            "Require Long < SMA5 & Short > SMA5",
            value=True
        )

    ls_tickers = list(dict.fromkeys([s.strip().upper().replace('.', '-') for s in re.split(r'[, \n]+', ls_tickers_input) if s.strip()]))
    if ls_exclude_3x:
        ls_tickers = [t for t in ls_tickers if t not in {"SOXL", "LABU", "JNUG", "GUSH"}]

    if len(ls_tickers) < 2:
        st.warning("Please enter at least 2 valid ETF tickers in the basket.")
    else:
        s_ls = (datetime.now() - timedelta(days=ls_years * 365)).strftime('%Y-%m-%d')
        e_ls = datetime.now().strftime('%Y-%m-%d')

        with st.spinner(f"Loading daily data and running Long/Short simulation across {len(ls_tickers)} ETFs..."):
            bulk_ls = fetch_bulk_history_cached(tuple(ls_tickers), s_ls, e_ls)
            etf_dfs = {}
            for sym in ls_tickers:
                d_sym = extract_symbol_daily(bulk_ls, sym, len(ls_tickers))
                prep_sym = prepare_ls_etf_data(d_sym)
                if not prep_sym.empty:
                    etf_dfs[sym] = prep_sym

        if len(etf_dfs) < 2:
            st.error("Could not retrieve sufficient historical data for at least 2 ETFs in the basket.")
        else:
            live_rank_rows = []
            for sym, df_sym in etf_dfs.items():
                last_r = df_sym.iloc[-1]
                c_px = float(last_r['Close_Adj'])
                sma5_px = float(last_r['SMA_5'])
                dist_sma5 = (c_px - sma5_px) / sma5_px * 100.0
                live_rank_rows.append({
                    "Ticker": sym,
                    "Last Date": df_sym.index[-1].strftime('%Y-%m-%d'),
                    "Last Close ($)": round(c_px, 2),
                    "5D SMA ($)": round(sma5_px, 2),
                    "Close vs 5D SMA (%)": round(dist_sma5, 2),
                    "Position vs SMA(5)": "Below SMA5 (Long Eligible)" if c_px <= sma5_px else "Above SMA5 (Short Eligible)",
                    "1-Day RSI(2)": round(float(last_r['RSI2_1d']), 1),
                    "2-Day RSI(2)": round(float(last_r['RSI2_2d']), 1),
                    "3-Day RSI(2)": round(float(last_r['RSI2_3d']), 1),
                    "5-Day RSI(2)": round(float(last_r['RSI2_5d']), 1),
                    "Active_RSI": float(last_r[f"RSI2_{ls_rsi_days}d"])
                })

            rank_df = pd.DataFrame(live_rank_rows).sort_values(by="Active_RSI", ascending=True).reset_index(drop=True)
            lowest_row = rank_df.iloc[0]
            highest_row = rank_df.iloc[-1]

            st.divider()
            st.markdown(f"#### 🎯 Current Basket Extremes (Ranked by **{ls_rsi_choice}** as of {lowest_row['Last Date']} Close)")
            sig_c1, sig_c2, sig_c3, sig_c4 = st.columns(4)
            sig_c1.metric(
                f"🟢 LONG Candidate (Lowest {ls_rsi_choice})",
                f"{lowest_row['Ticker']} (${lowest_row['Last Close ($)']:.2f})",
                f"RSI: {lowest_row['Active_RSI']:.1f}"
            )
            sig_c2.metric(
                f"{lowest_row['Ticker']} Long Exit Target (> 5D SMA)",
                f"${lowest_row['5D SMA ($)']:.2f}",
                f"{lowest_row['Close vs 5D SMA (%)']:+.2f}% vs SMA5"
            )
            sig_c3.metric(
                f"🔴 SHORT Candidate (Highest {ls_rsi_choice})",
                f"{highest_row['Ticker']} (${highest_row['Last Close ($)']:.2f})",
                f"RSI: {highest_row['Active_RSI']:.1f}"
            )
            sig_c4.metric(
                f"{highest_row['Ticker']} Short Cover Target (< 5D SMA)",
                f"${highest_row['5D SMA ($)']:.2f}",
                f"{highest_row['Close vs 5D SMA (%)']:+.2f}% vs SMA5"
            )

            with st.expander("📋 View Full 13-ETF Live RSI(2) Ranking Table (Lowest → Highest)", expanded=False):
                display_rank_df = rank_df.drop(columns=["Active_RSI"])
                st.dataframe(display_rank_df, hide_index=True, use_container_width=True)

            st.divider()
            st.subheader("1. Compare X-Day RSI(2) Lookback Horizons (1D vs. 2D vs. 3D vs. 5D)")

            comp_lookback_rows = []
            cached_runs = {}
            for lb_label, lb_days in RSI_DAYS_MAP.items():
                y_df, eq_df, tr_df, attr_df = run_long_short_basket_backtest(
                    etf_dfs=etf_dfs,
                    rsi_days=lb_days,
                    initial_capital=100000.0,
                    execution_mode=ls_exec_mode,
                    require_pre_exit_side=ls_pre_exit_check,
                    max_long_rsi=float(ls_max_long_rsi),
                    min_short_rsi=float(ls_min_short_rsi),
                    max_hold_days=20,
                    entry_limit_pct=float(ls_limit_pct)
                )
                cached_runs[lb_days] = (y_df, eq_df, tr_df, attr_df)
                if not y_df.empty:
                    ov = y_df.loc["All Years"]
                    comp_lookback_rows.append({
                        "RSI(2) Lookback": lb_label,
                        "Total Trades": int(ov["Total Trades"]),
                        "Long Win Rate (%)": ov["Long Win Rate (%)"],
                        "Short Win Rate (%)": ov["Short Win Rate (%)"],
                        "Combined Win Rate (%)": ov["Combined Win Rate (%)"],
                        "Profit Factor": ov["Profit Factor"],
                        "Long-Only Total Ret (%)": ov["Long-Only Return (%)"],
                        "Short-Only Total Ret (%)": ov["Short-Only Return (%)"],
                        "Combined L/S Total Ret (%)": ov["Combined L/S Return (%)"],
                        "Combined Max DD (%)": ov["Combined Max DD (%)"]
                    })

            if comp_lookback_rows:
                comp_lb_df = pd.DataFrame(comp_lookback_rows).set_index("RSI(2) Lookback")
                lb_cfg = {
                    "Total Trades": st.column_config.NumberColumn(format="%d"),
                    "Long Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Short Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Combined Win Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                    "Long-Only Total Ret (%)": st.column_config.NumberColumn(format="%+,.2f%%"),
                    "Short-Only Total Ret (%)": st.column_config.NumberColumn(format="%+,.2f%%"),
                    "Combined L/S Total Ret (%)": st.column_config.NumberColumn(format="%+,.2f%%"),
                    "Combined Max DD (%)": st.column_config.NumberColumn(format="%.1f%%")
                }
                st.dataframe(comp_lb_df, column_config=lb_cfg, use_container_width=True)

            ls_yearly_df, ls_equity_df, ls_trades_df, ls_attr_df = cached_runs[ls_rsi_days]

            if not ls_yearly_df.empty:
                st.divider()
                st.subheader(f"2. Detailed Performance & Leg Attribution — **{ls_rsi_choice}**")
                ov_sel = ls_yearly_df.loc["All Years"]

                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Combined L/S Return", f"{ov_sel['Combined L/S Return (%)']:+,.2f}%")
                k2.metric("Long-Only Return", f"{ov_sel['Long-Only Return (%)']:+,.2f}%")
                k3.metric("Short-Only Return", f"{ov_sel['Short-Only Return (%)']:+,.2f}%")
                k4.metric("Long / Short Win Rate", f"{ov_sel['Long Win Rate (%)']:.1f}% / {ov_sel['Short Win Rate (%)']:.1f}%")
                k5.metric("Profit Factor", f"{ov_sel['Profit Factor']}")
                k6.metric("Combined Max DD", f"{ov_sel['Combined Max DD (%)']:.1f}%")

                fig_ls_eq = go.Figure()
                fig_ls_eq.add_trace(go.Scatter(
                    x=ls_equity_df.index, y=ls_equity_df['Combined_Equity'],
                    mode='lines', name='Combined 50/50 Long/Short ($)',
                    line=dict(color='#00E676', width=2.6)
                ))
                fig_ls_eq.add_trace(go.Scatter(
                    x=ls_equity_df.index, y=ls_equity_df['Long_Only_Equity'],
                    mode='lines', name='Long Leg Only (Lowest RSI)',
                    line=dict(color='#29B6F6', width=1.8, dash='dot')
                ))
                fig_ls_eq.add_trace(go.Scatter(
                    x=ls_equity_df.index, y=ls_equity_df['Short_Only_Equity'],
                    mode='lines', name='Short Leg Only (Highest RSI)',
                    line=dict(color='#FF5252', width=1.8, dash='dash')
                ))
                fig_ls_eq.update_layout(
                    title=f"Equity Curve Attribution ($100,000 Start) — {ls_rsi_choice}",
                    template="plotly_dark",
                    yaxis_title="Account Equity ($)",
                    xaxis_title="Date",
                    height=430,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_ls_eq, use_container_width=True)

                col_y_ls, col_attr_ls = st.columns([1.45, 1.35])
                with col_y_ls:
                    st.markdown(f"##### 📅 Year-by-Year Results ({ls_rsi_choice})")
                    yr_ls_cfg = {
                        "Total Trades": st.column_config.NumberColumn(format="%d"),
                        "Long Trades": st.column_config.NumberColumn(format="%d"),
                        "Short Trades": st.column_config.NumberColumn(format="%d"),
                        "Long Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Short Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Combined Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                        "Long-Only Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                        "Short-Only Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                        "Combined L/S Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                        "Combined Max DD (%)": st.column_config.NumberColumn(format="%.1f%%")
                    }
                    st.dataframe(ls_yearly_df, column_config=yr_ls_cfg, use_container_width=True, height=420)

                with col_attr_ls:
                    st.markdown(f"##### 📊 Per-ETF Long vs. Short Breakdown ({ls_rsi_choice})")
                    if not ls_attr_df.empty:
                        attr_cfg = {
                            "Total Trades": st.column_config.NumberColumn(format="%d"),
                            "Long Trades": st.column_config.NumberColumn(format="%d"),
                            "Long Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                            "Long Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                            "Short Trades": st.column_config.NumberColumn(format="%d"),
                            "Short Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                            "Short Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                            "Overall Avg Ret (%)": st.column_config.NumberColumn(format="%+.2f%%")
                        }
                        st.dataframe(ls_attr_df, column_config=attr_cfg, use_container_width=True, height=420)

                with st.expander(f"🔎 View Complete Long/Short Trade Audit Log ({len(ls_trades_df)} Trades)"):
                    st.download_button(
                        label="📥 Download Long/Short Trade Log (CSV)",
                        data=ls_trades_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"long_short_rsi2_{ls_rsi_days}d_trades.csv",
                        mime="text/csv"
                    )
                    st.dataframe(ls_trades_df.sort_values(by="Exit Date (Open)", ascending=False), use_container_width=True)

# ----------------------------------------------------
# TAB 2: SINGLE TICKER STUDY (WITH LIMIT ENTRY & CANDLESTICK CHART)
# ----------------------------------------------------
with tab_study:
    st.subheader("Strategy Configuration")

    c1, c2, c3, c4, c5 = st.columns([0.9, 1.1, 1.35, 1.35, 0.8])
    with c1:
        study_ticker = st.text_input("Ticker Symbol", value="META", key="study_sym").upper().strip()
    with c2:
        rsi_choice = st.selectbox(
            "RSI(2) Lookback",
            ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"],
            index=0
        )
    with c3:
        study_limit_pct = st.selectbox(
            "Entry Execution (Open vs. Limit)",
            LIMIT_DISCOUNT_OPTIONS,
            index=0,
            format_func=format_limit_label,
            help="Select 0% to enter at the Next Day's Open, or 1%–7% to enter on a Limit Order set X% below the Next Day's Open (fills only if Next Day's Low <= Limit Price).",
            key="study_limit_pct"
        )
    with c4:
        signal_mode = st.selectbox(
            "Entry Signal Counting",
            ["All Days in Bucket", "First Day Entering Bucket Only (No Overlap)"],
            index=0
        )
    with c5:
        years_back = st.number_input("History (Years)", min_value=1, max_value=30, value=20)

    active_rsi_days = RSI_DAYS_MAP[rsi_choice]
    exit_label_map = get_exit_label_map(rsi_choice)
    inv_exit_label_map = {v: k for k, v in exit_label_map.items()}
    active_entry_label = format_limit_label(study_limit_pct)

    c6, c7, c8 = st.columns([1.6, 1.6, 1.2])
    with c6:
        selected_exit_label = st.selectbox(
            "Exit Rule (Executed on Next Day's Open)",
            list(exit_label_map.values()),
            index=0
        )
        active_exit_key = inv_exit_label_map[selected_exit_label]
    with c7:
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
    with c8:
        st.write("")
        require_uptrend_study = st.checkbox("Require Close > 200-Day SMA", value=True)

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
        processed_df = prepare_stock_data(
            raw_df,
            rsi_days=active_rsi_days,
            entry_discount_pct=float(study_limit_pct)
        )
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
        st.subheader(f"1. Performance by Entry Bucket — Entry: {active_entry_label} | Exit: {selected_exit_label}")
        bucket_table = build_bucket_table(filtered_df, active_exit_key)

        col_tbl, col_charts = st.columns([1.45, 1.55])
        with col_tbl:
            b_cfg = {
                "Signals": st.column_config.NumberColumn(format="%d"),
                "Filled Trades": st.column_config.NumberColumn(format="%d"),
                "Fill Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
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
                title=f"Avg Return (%) by {rsi_choice} Bucket ({active_entry_label})",
                template="plotly_dark",
                yaxis_title="Avg Return (%)",
                xaxis_title=f"{rsi_choice} Entry Bucket",
                height=390,
                margin=dict(l=20, r=20, t=45, b=20)
            )
            st.plotly_chart(fig_b, use_container_width=True)

        # ==========================================
        # SECTION 2: LIMIT ENTRY COMPARISON (0% vs 1%..7% BELOW NEXT OPEN)
        # ==========================================
        st.divider()
        bucket_title_str = ", ".join(active_buckets) if len(active_buckets) <= 3 else "All 10 Buckets"
        st.subheader(f"2. Limit Order Entry Comparison (0% Open vs. 1%–7% Below Next Open) — Bucket(s): {bucket_title_str}")
        st.caption(
            f"Compares entering at the Next Day's Open (0%) against placing a Limit Buy Order 1% to 7% below the Next Day's Open "
            f"(Exit: **{selected_exit_label}**)."
        )

        limit_comp_df = build_limit_offset_comparison_table(
            raw_df=raw_df,
            rsi_days=active_rsi_days,
            selected_buckets=active_buckets,
            exit_key=active_exit_key,
            require_uptrend=require_uptrend_study,
            fresh_only=fresh_only
        )

        lim_cfg = {
            "Signals": st.column_config.NumberColumn(format="%d"),
            "Filled Trades": st.column_config.NumberColumn(format="%d"),
            "Fill Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Win Rate (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Avg Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
            "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
            "Compounded Return (%)": st.column_config.NumberColumn(format="%+,.2f%%"),
            "Avg Hold (Days)": st.column_config.NumberColumn(format="%.1f d"),
            "Avg Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%")
        }
        st.dataframe(limit_comp_df, column_config=lim_cfg, use_container_width=True)

        # ==========================================
        # SECTION 3: EXIT STRATEGY COMPARISON
        # ==========================================
        st.divider()
        st.subheader(f"3. Compare All Exit Rules — Entry: {active_entry_label} | Bucket(s): {bucket_title_str}")

        sub_tab1, sub_tab2 = st.tabs([
            f"📋 Exit Comparison Table (Entry: {bucket_title_str})",
            "🔢 Full Entry × Exit Grid (All Buckets vs. All Exits)"
        ])

        with sub_tab1:
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
            mat_cfg = {"Filled Trades": st.column_config.NumberColumn(format="%d")}
            for k in EXIT_KEYS:
                mat_cfg[exit_label_map[k]] = st.column_config.NumberColumn(format=fmt_str)

            st.dataframe(matrix_df, column_config=mat_cfg, use_container_width=True)

        # ==========================================
        # SECTION 4: YEAR-BY-YEAR & TRADE LOG
        # ==========================================
        st.divider()
        st.subheader(
            f"4. Year-by-Year Breakdown — Bucket(s): {bucket_title_str} | Entry: {active_entry_label} | Exit: {selected_exit_label}"
        )
        yearly_df, trades_df = build_yearly_and_trades(processed_df, filtered_df, active_buckets, active_exit_key)

        if yearly_df.empty:
            st.info("No historical trades filled under the selected limit and bucket filters.")
        else:
            y_col1, y_col2 = st.columns([1.4, 1.6])
            with y_col1:
                y_cfg = {
                    "Signals": st.column_config.NumberColumn(format="%d"),
                    "Filled Trades": st.column_config.NumberColumn(format="%d"),
                    "Fill Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
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
                    marker_color=['#00E676' if v >= 0 else '#FF5252' for v in chart_yr['Compounded Return (%)']]
                ))
                fig_yr.add_hline(y=0, line_dash="dash", line_color="white")
                fig_yr.update_layout(
                    title=f"{study_ticker} Annual Compounded Return (%) by Year ({active_entry_label})",
                    template="plotly_dark",
                    yaxis_title="Compounded Return (%)",
                    xaxis_title="Year",
                    height=400,
                    margin=dict(l=20, r=20, t=45, b=20)
                )
                st.plotly_chart(fig_yr, use_container_width=True)

            with st.expander(f"🔎 View Individual Trade Audit Log ({len(trades_df)} Filled Trades — {active_entry_label})"):
                st.download_button(
                    label="Download Trade Log (CSV)",
                    data=trades_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{study_ticker}_{rsi_choice}_limit_{study_limit_pct}pct_trades.csv",
                    mime="text/csv"
                )
                st.dataframe(trades_df.sort_values(by="Signal Date (Close)", ascending=False), use_container_width=True)

# ----------------------------------------------------
# TAB 3: UNIVERSE MONITOR & EXIT PRICES
# ----------------------------------------------------
with tab_screener:
    st.subheader("🌐 Full Universe Monitor — Current RSI Readings & Target Exit Prices")
    u_c1, u_c2, u_c3, u_c4, u_c5, u_c6 = st.columns([1.3, 1.1, 1.0, 1.0, 1.0, 1.1])
    with u_c1:
        mon_universe = st.selectbox(
            "1. Select Universe",
            [
                "Major ETFs (105+ ETFs)",
                "Dow Jones 30",
                "Nasdaq 100",
                "S&P 500",
                "Russell 2000 (Small Caps)"
            ],
            index=2,
            key="mon_univ_select"
        )
    with u_c2:
        univ_key = "Major ETFs" if "Major ETFs" in mon_universe else mon_universe
        full_u_roster = get_universe_tickers(univ_key)
        total_avail = len(full_u_roster)

        if total_avail > 150:
            size_options = [f"All Constituents ({total_avail})", "Top 100", "Top 250", "Top 500"]
            size_choice = st.selectbox("Constituent Batch Limit", size_options, index=0, key="mon_batch_lim")
            if "100" in size_choice:
                u_tickers = full_u_roster[:100]
            elif "250" in size_choice:
                u_tickers = full_u_roster[:250]
            elif "500" in size_choice:
                u_tickers = full_u_roster[:500]
            else:
                u_tickers = full_u_roster
        else:
            st.text_input("Universe Size", value=f"{total_avail} Tickers", disabled=True)
            u_tickers = full_u_roster

    with u_c3:
        mon_bucket_rsi = st.selectbox("Bucket RSI Lookback", ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"], index=0, key="mon_b_rsi")
        mon_b_days = RSI_DAYS_MAP[mon_bucket_rsi]
    with u_c4:
        mon_custom_exit_days = st.selectbox("Custom Exit RSI", [1, 2, 3, 5], index=1, format_func=lambda x: f"{x}-Day RSI(2)", key="mon_cx_days")
    with u_c5:
        mon_custom_exit_thresh = st.selectbox("Custom Exit Threshold", RSI_THRESHOLDS, index=6, format_func=lambda x: f"RSI > {x}", key="mon_cx_thresh")
    with u_c6:
        mon_search = st.text_input("Filter Ticker (Optional)", value="", placeholder="e.g. AAPL, NVDA", key="mon_search")

    univ_rows = load_universe_snapshots_batched(
        u_tickers,
        active_rsi_days=mon_b_days,
        custom_exit_days=mon_custom_exit_days,
        custom_exit_thresh=mon_custom_exit_thresh,
        batch_size=125
    )

    if univ_rows:
        full_univ_df = pd.DataFrame(univ_rows)
        custom_col_label = f"Exit: {mon_custom_exit_days}D RSI>{mon_custom_exit_thresh} ($)"

        sort_c1, sort_c2, sort_c3 = st.columns([1.6, 1.0, 1.8])
        with sort_c1:
            default_sort_col = st.selectbox(
                "Sort Universe By Column:",
                options=list(full_univ_df.columns),
                index=list(full_univ_df.columns).index(f"{mon_b_days}D RSI(2)"),
                key="mon_sort_col"
            )
        with sort_c2:
            sort_dir = st.radio("Sort Order", ["Ascending (Low → High)", "Descending (High → Low)"], horizontal=True, key="mon_sort_dir")
        with sort_c3:
            trend_filter_mode = st.radio(
                "Display Scope",
                ["All Tickers in Universe", "Only Above 200D SMA", "Only Oversold (≤20 RSI)"],
                horizontal=True,
                key="mon_scope_mode"
            )

        display_univ_df = full_univ_df.copy()
        if mon_search.strip():
            q_tokens = [t.strip().upper() for t in re.split(r'[, \n]+', mon_search) if t.strip()]
            display_univ_df = display_univ_df[display_univ_df["Ticker"].isin(q_tokens)]

        if trend_filter_mode == "Only Above 200D SMA":
            display_univ_df = display_univ_df[display_univ_df["% vs 200 SMA"] > 0]
        elif trend_filter_mode == "Only Oversold (≤20 RSI)":
            display_univ_df = display_univ_df[display_univ_df[f"{mon_b_days}D RSI(2)"] <= 20.0]

        asc_bool = ("Ascending" in sort_dir)
        display_univ_df = display_univ_df.sort_values(by=default_sort_col, ascending=asc_bool, na_position='last').reset_index(drop=True)

        st.dataframe(
            display_univ_df,
            column_config=get_snapshot_column_config(custom_col_label),
            hide_index=True,
            use_container_width=True,
            height=620
        )

# ----------------------------------------------------
# TAB 4: CUSTOM WATCHLIST & EXIT PRICES
# ----------------------------------------------------
with tab_watchlist:
    st.subheader("📋 Custom Watchlist — Current RSI Readings & Target Exit Prices")
    w_c1, w_c2, w_c3, w_c4 = st.columns([2.2, 1.0, 1.0, 1.0])
    with w_c1:
        w_input = st.text_input(
            "Watchlist Tickers (comma or space separated)",
            value=", ".join(DEFAULT_LS_BASKET),
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
    if w_list:
        w_rows = load_universe_snapshots_batched(
            w_list,
            active_rsi_days=w_b_days,
            custom_exit_days=w_cx_days,
            custom_exit_thresh=w_cx_thresh,
            batch_size=100
        )
        if w_rows:
            watch_df = pd.DataFrame(w_rows)
            w_custom_col = f"Exit: {w_cx_days}D RSI>{w_cx_thresh} ($)"
            watch_df = watch_df.sort_values(by=f"{w_b_days}D RSI(2)", ascending=True).reset_index(drop=True)
            st.dataframe(
                watch_df,
                column_config=get_snapshot_column_config(w_custom_col),
                hide_index=True,
                use_container_width=True,
                height=500
            )

# ----------------------------------------------------
# TAB 5: MULTI-STOCK LONG-ONLY PORTFOLIO BACKTESTER
# ----------------------------------------------------
with tab_portfolio:
    st.subheader("Multi-Stock Portfolio Backtester (Next-Open & Limit-Below-Open)")
    p_col1, p_col2, p_col3 = st.columns([1.3, 1.4, 1.3])
    with p_col1:
        port_source = st.radio("Universe Mode", ["Index Universe", "Custom Ticker List"], horizontal=True)
    with p_col2:
        if port_source == "Index Universe":
            port_univ = st.selectbox("Select Universe", ["Major ETFs", "Dow Jones 30", "Nasdaq 100", "S&P 500", "Russell 2000 (Small Caps)"], index=1)
        else:
            port_custom = st.text_input("Tickers (comma-separated)", value=", ".join(DEFAULT_LS_BASKET))
    with p_col3:
        port_pool_limit = st.selectbox("Max Universe Candidates", ["Top 50 Liquid", "Top 100 Liquid", "Top 250 Liquid", "Full Universe"], index=1)

    p_c4, p_c5, p_c6, p_c7, p_c7b = st.columns([1.0, 1.2, 1.3, 1.3, 0.7])
    with p_c4:
        port_rsi_choice = st.selectbox("RSI(2) Lookback", ["2-Day RSI(2)", "3-Day RSI(2)", "5-Day RSI(2)"], index=0, key="p_rsi")
        port_rsi_days = RSI_DAYS_MAP[port_rsi_choice]
        port_exit_map = get_exit_label_map(port_rsi_choice)
        port_inv_exit_map = {v: k for k, v in port_exit_map.items()}
    with p_c5:
        port_buckets = st.multiselect("Target Entry Buckets", options=BUCKET_LABELS, default=['<10', '11-20'], key="p_buckets")
    with p_c6:
        port_limit_pct = st.selectbox(
            "Entry Execution (Open vs. Limit)",
            LIMIT_DISCOUNT_OPTIONS,
            index=0,
            format_func=format_limit_label,
            key="p_limit_pct"
        )
    with p_c7:
        port_exit_label = st.selectbox("Exit Rule (Next Open)", list(port_exit_map.values()), index=0, key="p_exit")
        port_exit_key = port_inv_exit_map[port_exit_label]
    with p_c7b:
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

    if st.button("🚀 Run Portfolio Simulation", type="primary", use_container_width=True):
        if port_source == "Index Universe":
            roster = get_universe_tickers(port_univ)
            if "50" in port_pool_limit:
                tickers_list = roster[:50]
            elif "100" in port_pool_limit:
                tickers_list = roster[:100]
            elif "250" in port_pool_limit:
                tickers_list = roster[:250]
            else:
                tickers_list = roster
        else:
            tickers_list = list(dict.fromkeys([s.strip().upper().replace('.', '-') for s in re.split(r'[, \n]+', port_custom) if s.strip()]))

        if tickers_list and port_buckets:
            with st.spinner(f"Downloading data and simulating portfolio across {len(tickers_list)} ticker(s)..."):
                s_p = (datetime.now() - timedelta(days=port_years * 365)).strftime('%Y-%m-%d')
                e_p = datetime.now().strftime('%Y-%m-%d')
                bulk_p = fetch_bulk_history_cached(tuple(tickers_list), s_p, e_p)

                prep_dict = {}
                for sym in tickers_list:
                    d_sym = extract_symbol_daily(bulk_p, sym, len(tickers_list))
                    if len(d_sym) >= 220:
                        p_df = prepare_stock_data(
                            d_sym,
                            rsi_days=port_rsi_days,
                            entry_discount_pct=float(port_limit_pct)
                        )
                        if not p_df.empty:
                            prep_dict[sym] = p_df

            if prep_dict:
                p_yr_df, p_eq_df, p_tr_df = run_portfolio_backtest_next_open(
                    prepared_dfs=prep_dict,
                    selected_buckets=port_buckets,
                    exit_key=port_exit_key,
                    exit_label=port_exit_label,
                    max_positions=int(port_max_pos),
                    initial_capital=float(port_capital),
                    require_uptrend=port_uptrend,
                    fresh_entry_only=port_fresh,
                    entry_discount_pct=float(port_limit_pct)
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
                    fig_eq.update_layout(title=f"Portfolio Equity Curve ($) — {format_limit_label(port_limit_pct)}", template="plotly_dark", height=360, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_eq, use_container_width=True)

                    if not p_tr_df.empty:
                        with st.expander(f"View Portfolio Trade Log ({len(p_tr_df)} Closed Trades)"):
                            st.dataframe(p_tr_df.sort_values(by="Exit Date (Open)", ascending=False), use_container_width=True)
