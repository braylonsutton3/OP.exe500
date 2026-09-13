from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None


# ============================================================
# OP.exe — HIGH-CONFIDENCE FULL REBUILD
# Primary goal: live-data LONG / SHORT / WAIT with strict confluence gating
#
# Included concepts:
# - 1m / 5m / 30m / 4h / 1d / 1w
# - configurable 09:00–09:30 ET or 09:30–10:00 ET ORB
# - SMA / EMA trend
# - RSI momentum state
# - ADX trend strength
# - VWAP (intraday)
# - ATR volatility
# - volatility regime
# - volume expansion
# - swing highs / lows
# - prior day / prior week highs and lows
# - session highs and lows
# - Break of Structure (BOS)
# - Change of Character / structure shift (CHoCH)
# - liquidity sweeps
# - displacement candles
# - Fair Value Gaps (FVGs)
# - objective order-block proxy
# - structural TP / SL
# - reward:risk
# - projected dollar P&L by contract
# - $200 minimum projected-profit filter for 10 contracts
#
# IMPORTANT:
# This is a rules-based analytical tool, not a guarantee.
# ============================================================

st.set_page_config(
    page_title="OP.exe",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SETTINGS
# ============================================================

TIMEFRAMES = {
    "1m":  {"px_unit": 2, "px_num": 1,  "lookback_days": 7,    "limit": 1500, "yf_period": "7d",  "yf_interval": "1m"},
    "5m":  {"px_unit": 2, "px_num": 5,  "lookback_days": 45,   "limit": 1500, "yf_period": "60d", "yf_interval": "5m"},
    "15m": {"px_unit": 2, "px_num": 15, "lookback_days": 60,   "limit": 1500, "yf_period": "60d", "yf_interval": "15m"},
    "30m": {"px_unit": 2, "px_num": 30, "lookback_days": 120,  "limit": 1200, "yf_period": "60d", "yf_interval": "30m"},
    "4h":  {"px_unit": 3, "px_num": 4,  "lookback_days": 365,  "limit": 1200, "yf_period": "2y",  "yf_interval": "1h"},
    "1d":  {"px_unit": 4, "px_num": 1,  "lookback_days": 1000, "limit": 1000, "yf_period": "5y",  "yf_interval": "1d"},
    "1w":  {"px_unit": 5, "px_num": 1,  "lookback_days": 2200, "limit": 500,  "yf_period": "10y", "yf_interval": "1wk"},
}

# fallback tick specs if ProjectX metadata is unavailable
FALLBACK_SPECS = {
    "MES": {"tick_size": 0.25, "tick_value": 1.25,  "yf": "MES=F"},
    "ES":  {"tick_size": 0.25, "tick_value": 12.50, "yf": "ES=F"},
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50,  "yf": "MNQ=F"},
    "NQ":  {"tick_size": 0.25, "tick_value": 5.00,  "yf": "NQ=F"},
    "M2K": {"tick_size": 0.10, "tick_value": 0.50,  "yf": "M2K=F"},
    "RTY": {"tick_size": 0.10, "tick_value": 5.00,  "yf": "RTY=F"},
    "MYM": {"tick_size": 1.00, "tick_value": 0.50,  "yf": "MYM=F"},
    "YM":  {"tick_size": 1.00, "tick_value": 5.00,  "yf": "YM=F"},
    "MGC": {"tick_size": 0.10, "tick_value": 1.00,  "yf": "MGC=F"},
    "GC":  {"tick_size": 0.10, "tick_value": 10.00, "yf": "GC=F"},
    "MCL": {"tick_size": 0.01, "tick_value": 1.00,  "yf": "MCL=F"},
    "CL":  {"tick_size": 0.01, "tick_value": 10.00, "yf": "CL=F"},
}

PX_BASE = "https://api.topstepx.com/api"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1220px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px;
        padding: 10px 12px;
    }
    .op-card {
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .op-long { border-left: 6px solid #22c55e; }
    .op-short { border-left: 6px solid #ef4444; }
    .op-wait { border-left: 6px solid #9ca3af; }
    .muted { opacity: .72; font-size: .88rem; }
    .tiny { opacity: .68; font-size: .80rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_root(symbol):
    s = (symbol or "").upper().strip().replace("!", "")
    for root in sorted(FALLBACK_SPECS.keys(), key=len, reverse=True):
        if s.startswith(root):
            return root
    return s




def classify_market(symbol, bundle_contract=None):
    s=(symbol or "").upper().strip()
    if bundle_contract is not None or clean_root(s) in FALLBACK_SPECS or s.endswith("=F"):
        return "futures"
    if s.endswith("-USD") or s.endswith("-USDT"):
        return "crypto"
    return "equity"


def market_session_status(symbol, bundle_contract=None):
    """Return (is_open, label, reason) using Topstep permitted futures hours,
    NYSE calendar for equities, and 24/7 for crypto. Freshness is checked separately.
    """
    kind = classify_market(symbol, bundle_contract)
    now_utc = pd.Timestamp.now(tz="UTC")

    if kind == "crypto":
        return True, "MARKET OPEN", "Crypto trades continuously; live-feed freshness is checked separately."

    if kind == "futures":
        # Topstep general permitted hours (Central Time): Sunday 17:00 open;
        # Mon-Fri flatten by 15:10; Mon-Thu reopen 17:00; Friday closed until Sunday.
        ct = now_utc.tz_convert("America/Chicago")
        wd = ct.weekday()  # Mon=0 ... Sun=6
        mins = ct.hour * 60 + ct.minute
        close_m = 15 * 60 + 10
        reopen_m = 17 * 60
        if wd == 5:
            return False, "MARKET CLOSED", "Topstep futures are closed Saturday."
        if wd == 6:
            if mins < reopen_m:
                return False, "MARKET CLOSED", "Topstep futures reopen Sunday at 5:00 PM CT."
            return True, "MARKET OPEN", "Topstep Sunday session is open."
        if wd == 4:
            if mins >= close_m:
                return False, "MARKET CLOSED", "Topstep Friday trading ends at 3:10 PM CT and reopens Sunday at 5:00 PM CT."
            return True, "MARKET OPEN", "Topstep weekday session is open."
        if close_m <= mins < reopen_m:
            return False, "MARKET CLOSED", "Topstep daily close window: trading resumes at 5:00 PM CT."
        return True, "MARKET OPEN", "Topstep permitted futures session is open (product-specific pauses/holidays may still apply)."

    # Equities/ETFs: use official exchange calendar when available.
    et = now_utc.tz_convert("America/New_York")
    if mcal is not None:
        try:
            nyse = mcal.get_calendar("NYSE")
            d = et.date()
            sched = nyse.schedule(start_date=d, end_date=d)
            if sched.empty:
                return False, "MARKET CLOSED", "NYSE calendar is closed today."
            op = pd.Timestamp(sched.iloc[0]["market_open"]).tz_convert("UTC")
            cl = pd.Timestamp(sched.iloc[0]["market_close"]).tz_convert("UTC")
            if not (op <= now_utc < cl):
                return False, "MARKET CLOSED", f"NYSE regular session is {op.tz_convert('America/New_York').strftime('%-I:%M %p')}–{cl.tz_convert('America/New_York').strftime('%-I:%M %p')} ET today."
            return True, "MARKET OPEN", "NYSE regular session is open."
        except Exception:
            pass
    # Safe fallback if exchange calendar package is unavailable.
    if et.weekday() >= 5 or not ((et.hour > 9 or (et.hour == 9 and et.minute >= 30)) and et.hour < 16):
        return False, "MARKET CLOSED", "U.S. equity regular session is not open."
    return True, "MARKET OPEN", "U.S. equity regular session appears open; exchange-calendar check unavailable."


def round_to_tick(price, tick_size):
    if price is None or not np.isfinite(price):
        return price
    if tick_size <= 0:
        return float(price)
    return round(round(float(price) / tick_size) * tick_size, 10)


def pnl_between(entry, exit_price, tick_size, tick_value, contracts):
    if tick_size <= 0:
        return 0.0
    ticks = abs(float(exit_price) - float(entry)) / tick_size
    return ticks * tick_value * int(contracts)


def safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df.rename(columns={
        "o": "Open", "h": "High", "l": "Low",
        "c": "Close", "v": "Volume", "t": "Datetime"
    })

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df = df.set_index("Datetime")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ============================================================
# PROJECTX / TOPSTEPX LIVE DATA
# ============================================================

def get_topstep_credentials():
    try:
        return (
            st.secrets.get("TOPSTEP_USERNAME", ""),
            st.secrets.get("TOPSTEP_API_KEY", ""),
        )
    except Exception:
        return "", ""


@st.cache_data(ttl=20 * 60, show_spinner=False)
def px_login(username, api_key):
    if not username or not api_key:
        return None, "TopstepX credentials are not configured."

    try:
        r = requests.post(
            f"{PX_BASE}/Auth/loginKey",
            json={"userName": username, "apiKey": api_key},
            timeout=12,
        )
        r.raise_for_status()
        js = r.json()
        if not js.get("success") or not js.get("token"):
            return None, js.get("errorMessage") or "TopstepX authentication failed."
        return js["token"], None
    except Exception as e:
        return None, f"TopstepX login error: {e}"


def px_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "accept": "text/plain",
        "Content-Type": "application/json",
    }


@st.cache_data(ttl=60, show_spinner=False)
def px_find_contract(token, search_text, live=True):
    r = requests.post(
        f"{PX_BASE}/Contract/search",
        headers=px_headers(token),
        json={"searchText": search_text, "live": live},
        timeout=12,
    )
    r.raise_for_status()
    js = r.json()
    contracts = js.get("contracts") or []
    if not contracts:
        return None

    active = [c for c in contracts if c.get("activeContract")]
    pool = active or contracts
    root = clean_root(search_text)
    exactish = [
        c for c in pool
        if str(c.get("name", "")).upper().startswith(root)
    ]
    return (exactish or pool)[0]


def px_retrieve_bars(token, contract_id, timeframe, live=True):
    cfg = TIMEFRAMES[timeframe]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cfg["lookback_days"])

    payload = {
        "contractId": contract_id,
        "live": live,
        "startTime": start.isoformat().replace("+00:00", "Z"),
        "endTime": end.isoformat().replace("+00:00", "Z"),
        "unit": cfg["px_unit"],
        "unitNumber": cfg["px_num"],
        "limit": cfg["limit"],
        "includePartialBar": True,
    }

    r = requests.post(
        f"{PX_BASE}/History/retrieveBars",
        headers=px_headers(token),
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    js = r.json()
    return normalize_df(pd.DataFrame(js.get("bars") or []))


# ============================================================
# YAHOO FALLBACK
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def yahoo_bars(symbol, timeframe):
    root = clean_root(symbol)
    yf_symbol = FALLBACK_SPECS.get(root, {}).get("yf", symbol)
    cfg = TIMEFRAMES[timeframe]

    data = yf.download(
        yf_symbol,
        period=cfg["yf_period"],
        interval=cfg["yf_interval"],
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = normalize_df(data)

    if timeframe == "4h" and not data.empty:
        data = (
            data.resample("4h")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna(subset=["Open", "High", "Low", "Close"])
        )

    return data


# ============================================================
# CORE INDICATORS
# ============================================================

def atr_series(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_atr(df, period=14):
    if df is None or len(df) < period + 2:
        return np.nan
    return safe_float(atr_series(df, period).iloc[-1])


def calculate_rsi(series, period=14):
    if len(series) < period + 2:
        return np.nan
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = (100 - (100 / (1 + rs))).iloc[-1]
    return 50.0 if not np.isfinite(value) else float(value)


def calculate_adx(df, period=14):
    if df is None or len(df) < period * 2 + 2:
        return np.nan

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_smoothed = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_smoothed.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_smoothed.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return safe_float(dx.rolling(period).mean().iloc[-1])


def intraday_vwap(df):
    if df is None or df.empty:
        return np.nan
    if "Volume" not in df.columns or df["Volume"].fillna(0).sum() <= 0:
        return np.nan

    local = df.tz_convert("America/New_York")
    latest_date = local.index[-1].date()
    day = local[local.index.date == latest_date]
    if day.empty:
        return np.nan

    typical = (day["High"] + day["Low"] + day["Close"]) / 3
    vol = day["Volume"].fillna(0)
    denom = vol.cumsum().iloc[-1]
    if denom <= 0:
        return np.nan
    return float((typical * vol).cumsum().iloc[-1] / denom)


def volume_expansion(df, lookback=20):
    if df is None or len(df) < lookback + 2:
        return 1.0
    vol = df["Volume"].fillna(0)
    baseline = vol.iloc[-lookback-1:-1].mean()
    if baseline <= 0:
        return 1.0
    return float(vol.iloc[-1] / baseline)


def volatility_regime(df):
    if df is None or len(df) < 60:
        return "UNKNOWN", 1.0

    atr14 = atr_series(df, 14)
    current = safe_float(atr14.iloc[-1])
    baseline = safe_float(atr14.iloc[-50:].median())

    if not np.isfinite(current) or not np.isfinite(baseline) or baseline <= 0:
        return "UNKNOWN", 1.0

    ratio = current / baseline

    if ratio >= 1.35:
        return "EXPANSION", ratio
    if ratio <= 0.75:
        return "COMPRESSION", ratio
    return "NORMAL", ratio


# ============================================================
# SWING STRUCTURE
# ============================================================

def swing_levels(df, window=3, lookback=180):
    data = df.tail(lookback)
    highs, lows = [], []

    if len(data) < window * 2 + 2:
        return highs, lows

    hv = data["High"].to_numpy()
    lv = data["Low"].to_numpy()
    idx = data.index

    for i in range(window, len(data) - window):
        if hv[i] >= np.max(hv[i-window:i+window+1]):
            highs.append((idx[i], float(hv[i])))
        if lv[i] <= np.min(lv[i-window:i+window+1]):
            lows.append((idx[i], float(lv[i])))

    return highs[-20:], lows[-20:]


def detect_bos_choch(df):
    """
    Objective proxy:
    - Bullish BOS = latest close breaks above latest confirmed swing high.
    - Bearish BOS = latest close breaks below latest confirmed swing low.
    - CHoCH = break is opposite the direction implied by the last two swing legs.
    """
    highs, lows = swing_levels(df)
    if not highs or not lows:
        return {"bos": "NONE", "choch": "NONE", "broken_level": None}

    close = float(df["Close"].iloc[-1])
    last_high = highs[-1][1]
    last_low = lows[-1][1]

    # estimate prior structure direction from last two highs/lows
    structure = "NEUTRAL"
    if len(highs) >= 2 and len(lows) >= 2:
        higher_high = highs[-1][1] > highs[-2][1]
        higher_low = lows[-1][1] > lows[-2][1]
        lower_high = highs[-1][1] < highs[-2][1]
        lower_low = lows[-1][1] < lows[-2][1]
        if higher_high and higher_low:
            structure = "UP"
        elif lower_high and lower_low:
            structure = "DOWN"

    bos = "NONE"
    choch = "NONE"
    level = None

    if close > last_high:
        bos = "BULLISH"
        level = last_high
        if structure == "DOWN":
            choch = "BULLISH"
    elif close < last_low:
        bos = "BEARISH"
        level = last_low
        if structure == "UP":
            choch = "BEARISH"

    return {"bos": bos, "choch": choch, "broken_level": level}


# ============================================================
# LIQUIDITY SWEEPS
# ============================================================

def detect_liquidity_sweep(df):
    """
    Bearish sweep:
    latest wick trades above a prior confirmed swing high,
    but closes back below that level.

    Bullish sweep:
    latest wick trades below a prior confirmed swing low,
    but closes back above that level.
    """
    if df is None or len(df) < 20:
        return {"type": "NONE", "level": None}

    highs, lows = swing_levels(df.iloc[:-1])
    if not highs or not lows:
        return {"type": "NONE", "level": None}

    bar = df.iloc[-1]
    prior_high = highs[-1][1]
    prior_low = lows[-1][1]

    if bar["High"] > prior_high and bar["Close"] < prior_high:
        return {"type": "BEARISH_SWEEP", "level": prior_high}

    if bar["Low"] < prior_low and bar["Close"] > prior_low:
        return {"type": "BULLISH_SWEEP", "level": prior_low}

    return {"type": "NONE", "level": None}


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(df):
    if df is None or len(df) < 20:
        return {"direction": "NONE", "strength": 0.0}

    a = calculate_atr(df)
    if not np.isfinite(a) or a <= 0:
        return {"direction": "NONE", "strength": 0.0}

    bar = df.iloc[-1]
    body = abs(float(bar["Close"] - bar["Open"]))
    full_range = max(float(bar["High"] - bar["Low"]), 1e-9)
    body_ratio = body / full_range
    strength = body / a

    if strength >= 0.80 and body_ratio >= 0.60:
        direction = "BULLISH" if bar["Close"] > bar["Open"] else "BEARISH"
        return {"direction": direction, "strength": strength}

    return {"direction": "NONE", "strength": strength}


# ============================================================
# FAIR VALUE GAPS
# ============================================================

def recent_fvgs(df, lookback=160):
    out = []
    if df is None or len(df) < 3:
        return out

    start = max(2, len(df) - lookback)

    for i in range(start, len(df)):
        left = df.iloc[i - 2]
        current = df.iloc[i]

        # bullish 3-candle imbalance
        if current["Low"] > left["High"]:
            out.append({
                "direction": "BULL",
                "low": float(left["High"]),
                "high": float(current["Low"]),
                "time": df.index[i],
            })

        # bearish 3-candle imbalance
        if current["High"] < left["Low"]:
            out.append({
                "direction": "BEAR",
                "low": float(current["High"]),
                "high": float(left["Low"]),
                "time": df.index[i],
            })

    return out[-40:]


# ============================================================
# ORDER BLOCK PROXY
# ============================================================

def recent_order_blocks(df, lookback=80):
    """
    Objective order-block proxy:
    - bullish OB = last bearish candle immediately before a strong bullish
      displacement that closes above a recent local high
    - bearish OB = last bullish candle immediately before a strong bearish
      displacement that closes below a recent local low

    This is deliberately given LOW weight because "order block" definitions
    are not standardized and evidence is weaker than basic trend/volatility/risk.
    """
    out = []
    if df is None or len(df) < 25:
        return out

    a_series = atr_series(df, 14)
    start = max(15, len(df) - lookback)

    for i in range(start, len(df)):
        a = safe_float(a_series.iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue

        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        body = abs(float(bar["Close"] - bar["Open"]))

        local_high = float(df["High"].iloc[max(0, i-8):i].max())
        local_low = float(df["Low"].iloc[max(0, i-8):i].min())

        bullish_displacement = (
            bar["Close"] > bar["Open"]
            and body >= 0.8 * a
            and bar["Close"] > local_high
        )

        bearish_displacement = (
            bar["Close"] < bar["Open"]
            and body >= 0.8 * a
            and bar["Close"] < local_low
        )

        if bullish_displacement and prev["Close"] < prev["Open"]:
            out.append({
                "direction": "BULL",
                "low": float(prev["Low"]),
                "high": float(prev["High"]),
                "time": df.index[i - 1],
            })

        if bearish_displacement and prev["Close"] > prev["Open"]:
            out.append({
                "direction": "BEAR",
                "low": float(prev["Low"]),
                "high": float(prev["High"]),
                "time": df.index[i - 1],
            })

    return out[-20:]


# ============================================================
# PRIOR DAY / WEEK / SESSION LEVELS
# ============================================================

def prior_day_week_levels(df):
    if df is None or df.empty:
        return {}

    local = df.tz_convert("America/New_York")
    daily = local.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    weekly = local.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()

    result = {}

    if len(daily) >= 2:
        result["PDH"] = float(daily["High"].iloc[-2])
        result["PDL"] = float(daily["Low"].iloc[-2])

    if len(weekly) >= 2:
        result["PWH"] = float(weekly["High"].iloc[-2])
        result["PWL"] = float(weekly["Low"].iloc[-2])

    latest_date = local.index[-1].date()
    current_day = local[local.index.date == latest_date]
    if not current_day.empty:
        result["SESSION_HIGH"] = float(current_day["High"].max())
        result["SESSION_LOW"] = float(current_day["Low"].min())

    return result


# ============================================================
# ORB
# ============================================================

def latest_orb(df_1m, mode):
    if df_1m is None or df_1m.empty:
        return None

    if mode == "09:30–10:00 ET":
        start_t, end_t = "09:30", "09:59"
    else:
        start_t, end_t = "09:00", "09:29"

    local = df_1m.tz_convert("America/New_York")
    unique_dates = list(dict.fromkeys(local.index.date))[::-1]

    for d in unique_dates:
        day = local[local.index.date == d]
        orb = day.between_time(start_t, end_t)
        if len(orb) >= 10:
            hi = float(orb["High"].max())
            lo = float(orb["Low"].min())
            return {
                "date": d,
                "high": hi,
                "low": lo,
                "mid": (hi + lo) / 2,
                "mode": mode,
            }
    return None


# ============================================================
# EVIDENCE ENGINE
# ============================================================

def evidence_engine(df, timeframe):
    """
    Produces a directional evidence score and a 0-100 CONFLUENCE INDEX.

    IMPORTANT:
    The confluence index is NOT a probability of winning.
    It only measures how much of OP.exe's own evidence agrees in one direction.
    """
    if df is None or len(df) < 60:
        return {
            "raw": "WAIT",
            "score": 0,
            "bull": 0,
            "bear": 0,
            "confluence_index": 0.0,
            "dominance": 0.0,
            "reasons": ["Not enough data"],
            "atr": np.nan,
            "rsi": np.nan,
            "adx": np.nan,
            "regime": "UNKNOWN",
            "bos": "NONE",
            "choch": "NONE",
            "sweep": "NONE",
            "displacement": "NONE",
            "volume_ratio": 1.0,
            "vwap": np.nan,
        }

    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    current = float(close.iloc[-1])
    a = calculate_atr(df)
    rv = calculate_rsi(close)
    adx = calculate_adx(df)
    regime, vol_ratio = volatility_regime(df)
    vol_exp = volume_expansion(df)
    vwap = intraday_vwap(df) if timeframe in ("1m", "5m", "30m") else np.nan
    structure = detect_bos_choch(df)
    sweep = detect_liquidity_sweep(df)
    displacement = detect_displacement(df)
    gaps = recent_fvgs(df)
    obs = recent_order_blocks(df)

    bull = 0.0
    bear = 0.0
    max_points = 0.0
    reasons = []

    def vote(direction, points, label):
        nonlocal bull, bear, max_points
        max_points += float(points)
        if direction == "BULL":
            bull += float(points)
            reasons.append(label)
        elif direction == "BEAR":
            bear += float(points)
            reasons.append(label)

    # Trend / momentum — strongest broad empirical base among these inputs
    vote("BULL" if current > sma20.iloc[-1] else "BEAR", 1.5, "price vs SMA20")
    vote("BULL" if sma20.iloc[-1] > sma50.iloc[-1] else "BEAR", 2.0, "SMA20 vs SMA50")
    vote("BULL" if ema9.iloc[-1] > ema21.iloc[-1] else "BEAR", 1.5, "EMA9 vs EMA21")

    momentum_n = min(5, len(close) - 1)
    momentum = float(close.iloc[-1] - close.iloc[-1 - momentum_n])
    if momentum > 0:
        vote("BULL", 1.5, "recent momentum up")
    elif momentum < 0:
        vote("BEAR", 1.5, "recent momentum down")
    else:
        max_points += 1.5

    if rv >= 55:
        vote("BULL", 1.0, "RSI bullish state")
    elif rv <= 45:
        vote("BEAR", 1.0, "RSI bearish state")
    else:
        max_points += 1.0

    # Intraday VWAP
    if timeframe in ("1m", "5m", "30m"):
        if np.isfinite(vwap):
            vote("BULL" if current > vwap else "BEAR", 1.5, "VWAP location")
        else:
            max_points += 1.5

    # Market structure
    if structure["bos"] == "BULLISH":
        vote("BULL", 2.5, "bullish BOS")
    elif structure["bos"] == "BEARISH":
        vote("BEAR", 2.5, "bearish BOS")
    else:
        max_points += 2.5

    if structure["choch"] == "BULLISH":
        vote("BULL", 1.5, "bullish CHoCH")
    elif structure["choch"] == "BEARISH":
        vote("BEAR", 1.5, "bearish CHoCH")
    else:
        max_points += 1.5

    # Liquidity sweep
    if sweep["type"] == "BULLISH_SWEEP":
        vote("BULL", 1.5, "bullish liquidity sweep")
    elif sweep["type"] == "BEARISH_SWEEP":
        vote("BEAR", 1.5, "bearish liquidity sweep")
    else:
        max_points += 1.5

    # Displacement
    if displacement["direction"] == "BULLISH":
        vote("BULL", 1.5, "bullish displacement")
    elif displacement["direction"] == "BEARISH":
        vote("BEAR", 1.5, "bearish displacement")
    else:
        max_points += 1.5

    # FVG context — supporting evidence only
    recent_gaps = gaps[-8:]
    has_bull_fvg = any(g["direction"] == "BULL" for g in recent_gaps)
    has_bear_fvg = any(g["direction"] == "BEAR" for g in recent_gaps)
    if has_bull_fvg and not has_bear_fvg:
        vote("BULL", 1.0, "bullish FVG context")
    elif has_bear_fvg and not has_bull_fvg:
        vote("BEAR", 1.0, "bearish FVG context")
    else:
        max_points += 1.0

    # Order-block proxy — intentionally low weight
    recent_obs = obs[-6:]
    has_bull_ob = any(o["direction"] == "BULL" for o in recent_obs)
    has_bear_ob = any(o["direction"] == "BEAR" for o in recent_obs)
    if has_bull_ob and not has_bear_ob:
        vote("BULL", 0.75, "bullish order-block proxy")
    elif has_bear_ob and not has_bull_ob:
        vote("BEAR", 0.75, "bearish order-block proxy")
    else:
        max_points += 0.75

    # ADX / volume / volatility are confidence modifiers, not direction generators
    directional_side = "BULL" if bull > bear else "BEAR" if bear > bull else None

    if np.isfinite(adx):
        max_points += 1.25
        if adx >= 25 and directional_side:
            if directional_side == "BULL":
                bull += 1.25
            else:
                bear += 1.25
            reasons.append("ADX confirms trend strength")

    max_points += 1.0
    if vol_exp >= 1.5 and directional_side:
        if directional_side == "BULL":
            bull += 1.0
        else:
            bear += 1.0
        reasons.append("volume expansion confirms move")

    # Expansion helps; compression penalizes directional confidence.
    max_points += 1.0
    if regime == "EXPANSION" and directional_side:
        if directional_side == "BULL":
            bull += 1.0
        else:
            bear += 1.0
        reasons.append("volatility expansion")
    elif regime == "COMPRESSION":
        if directional_side == "BULL":
            bull = max(0.0, bull - 1.0)
        elif directional_side == "BEAR":
            bear = max(0.0, bear - 1.0)
        reasons.append("volatility compression penalty")

    score = bull - bear
    total_directional = bull + bear
    dominant = max(bull, bear)

    confluence_index = 0.0 if max_points <= 0 else 100.0 * dominant / max_points
    dominance = 0.0 if total_directional <= 0 else 100.0 * abs(bull - bear) / total_directional

    # Strict raw gate. Final gate below is even stricter.
    if score >= 5.0 and confluence_index >= 65 and dominance >= 35:
        raw = "LONG"
    elif score <= -5.0 and confluence_index >= 65 and dominance >= 35:
        raw = "SHORT"
    else:
        raw = "WAIT"

    return {
        "raw": raw,
        "score": score,
        "bull": bull,
        "bear": bear,
        "confluence_index": confluence_index,
        "dominance": dominance,
        "reasons": reasons,
        "atr": a,
        "rsi": rv,
        "adx": adx,
        "regime": regime,
        "regime_ratio": vol_ratio,
        "bos": structure["bos"],
        "choch": structure["choch"],
        "sweep": sweep["type"],
        "displacement": displacement["direction"],
        "volume_ratio": vol_exp,
        "vwap": vwap,
    }


# ============================================================
# TARGET / STOP CANDIDATE ENGINE
# ============================================================

def add_candidate(candidates, price, label, weight, entry, direction):
    if price is None or not np.isfinite(price):
        return
    if direction == "LONG" and price <= entry:
        return
    if direction == "SHORT" and price >= entry:
        return
    candidates.append({
        "price": float(price),
        "label": label,
        "weight": float(weight),
        "distance": abs(float(price) - float(entry)),
    })


def add_stop_candidate(candidates, price, label, entry, direction):
    if price is None or not np.isfinite(price):
        return
    if direction == "LONG" and price >= entry:
        return
    if direction == "SHORT" and price <= entry:
        return
    candidates.append({
        "price": float(price),
        "label": label,
        "distance": abs(float(price) - float(entry)),
    })


def choose_target_and_stop(df, direction, timeframe, tick_size, orb=None):
    if direction not in ("LONG", "SHORT") or df is None or df.empty:
        return None

    entry = float(df["Close"].iloc[-1])
    a = calculate_atr(df)
    if not np.isfinite(a) or a <= 0:
        return None

    highs, lows = swing_levels(df)
    gaps = recent_fvgs(df)
    obs = recent_order_blocks(df)
    levels = prior_day_week_levels(df)

    targets = []
    stops = []

    # ---------- swing highs/lows ----------
    for _, p in highs[-8:]:
        if direction == "LONG":
            add_candidate(targets, p, "swing high liquidity", 3.0, entry, direction)
        else:
            add_stop_candidate(stops, p, "swing-high invalidation", entry, direction)

    for _, p in lows[-8:]:
        if direction == "SHORT":
            add_candidate(targets, p, "swing low liquidity", 3.0, entry, direction)
        else:
            add_stop_candidate(stops, p, "swing-low invalidation", entry, direction)

    # ---------- PDH / PDL / PWH / PWL / session ----------
    if direction == "LONG":
        add_candidate(targets, levels.get("PDH"), "prior-day high", 4.0, entry, direction)
        add_candidate(targets, levels.get("PWH"), "prior-week high", 4.0, entry, direction)
        add_candidate(targets, levels.get("SESSION_HIGH"), "session high", 2.0, entry, direction)
        add_stop_candidate(stops, levels.get("PDL"), "prior-day low invalidation", entry, direction)
        add_stop_candidate(stops, levels.get("SESSION_LOW"), "session-low invalidation", entry, direction)
    else:
        add_candidate(targets, levels.get("PDL"), "prior-day low", 4.0, entry, direction)
        add_candidate(targets, levels.get("PWL"), "prior-week low", 4.0, entry, direction)
        add_candidate(targets, levels.get("SESSION_LOW"), "session low", 2.0, entry, direction)
        add_stop_candidate(stops, levels.get("PDH"), "prior-day high invalidation", entry, direction)
        add_stop_candidate(stops, levels.get("SESSION_HIGH"), "session-high invalidation", entry, direction)

    # ---------- FVGs ----------
    for g in gaps[-15:]:
        if direction == "LONG" and g["low"] > entry:
            add_candidate(targets, g["low"], "FVG near edge", 2.0, entry, direction)
            add_candidate(targets, g["high"], "FVG far edge", 1.5, entry, direction)
        elif direction == "SHORT" and g["high"] < entry:
            add_candidate(targets, g["high"], "FVG near edge", 2.0, entry, direction)
            add_candidate(targets, g["low"], "FVG far edge", 1.5, entry, direction)

        if direction == "LONG" and g["high"] < entry:
            add_stop_candidate(stops, g["low"], "FVG invalidation", entry, direction)
        elif direction == "SHORT" and g["low"] > entry:
            add_stop_candidate(stops, g["high"], "FVG invalidation", entry, direction)

    # ---------- order blocks ----------
    for ob in obs[-8:]:
        if direction == "LONG":
            if ob["direction"] == "BEAR" and ob["low"] > entry:
                add_candidate(targets, ob["low"], "opposing order block", 1.0, entry, direction)
            if ob["direction"] == "BULL" and ob["high"] < entry:
                add_stop_candidate(stops, ob["low"], "bullish order-block invalidation", entry, direction)
        else:
            if ob["direction"] == "BULL" and ob["high"] < entry:
                add_candidate(targets, ob["high"], "opposing order block", 1.0, entry, direction)
            if ob["direction"] == "BEAR" and ob["low"] > entry:
                add_stop_candidate(stops, ob["high"], "bearish order-block invalidation", entry, direction)

    # ---------- ORB ----------
    if orb and timeframe in ("1m", "5m", "30m"):
        if direction == "LONG":
            if orb["low"] < entry < orb["high"]:
                add_candidate(targets, orb["high"], "ORB high", 4.0, entry, direction)
                add_stop_candidate(stops, orb["low"], "ORB low invalidation", entry, direction)
            elif entry > orb["high"]:
                add_stop_candidate(stops, orb["high"], "ORB breakout invalidation", entry, direction)
        else:
            if orb["low"] < entry < orb["high"]:
                add_candidate(targets, orb["low"], "ORB low", 4.0, entry, direction)
                add_stop_candidate(stops, orb["high"], "ORB high invalidation", entry, direction)
            elif entry < orb["low"]:
                add_stop_candidate(stops, orb["low"], "ORB breakout invalidation", entry, direction)

    # ---------- ATR extension fallback ----------
    if direction == "LONG":
        add_candidate(targets, entry + 1.5 * a, "1.5 ATR extension", 1.0, entry, direction)
    else:
        add_candidate(targets, entry - 1.5 * a, "1.5 ATR extension", 1.0, entry, direction)

    # Score targets: favor meaningful structure without choosing something absurdly far away.
    for t in targets:
        dist_atr = t["distance"] / a
        distance_penalty = max(0.0, dist_atr - 3.0) * 0.75
        t["rank"] = t["weight"] - distance_penalty

    targets = sorted(
        targets,
        key=lambda x: (-x["rank"], x["distance"])
    )

    if not targets:
        return None

    chosen_target = targets[0]

    # nearest valid structure stop, but not unrealistically tight
    min_stop_distance = 0.35 * a
    valid_stops = [s for s in stops if s["distance"] >= min_stop_distance]

    if valid_stops:
        chosen_stop = min(valid_stops, key=lambda x: x["distance"])
        raw_stop = chosen_stop["price"]
        stop_label = chosen_stop["label"]
    else:
        raw_stop = entry - a if direction == "LONG" else entry + a
        stop_label = "1 ATR fallback invalidation"

    # modest volatility buffer beyond structure
    buffer = 0.10 * a
    stop = raw_stop - buffer if direction == "LONG" else raw_stop + buffer

    entry = round_to_tick(entry, tick_size)
    target = round_to_tick(chosen_target["price"], tick_size)
    stop = round_to_tick(stop, tick_size)

    reward = abs(target - entry)
    risk = abs(entry - stop)
    rr = reward / risk if risk > 0 else 0.0
    atr_multiple = reward / a if a > 0 else 0.0

    return {
        "entry": entry,
        "target": target,
        "target_label": chosen_target["label"],
        "stop": stop,
        "stop_label": stop_label,
        "reward_points": reward,
        "risk_points": risk,
        "rr": rr,
        "atr_multiple": atr_multiple,
    }


# ============================================================
# FINAL QUALIFICATION FILTER
# ============================================================

def qualify_signal(
    raw_direction,
    plan,
    tick_size,
    tick_value,
    contracts,
    minimum_profit,
    minimum_rr,
    minimum_atr_multiple,
    confluence_index,
    dominance,
    minimum_confluence_index,
    minimum_dominance,
):
    """
    Final OP.exe gate:
    LONG/SHORT survives only when BOTH market direction and trade structure are strong.
    The confluence index is not a win probability.
    """
    if raw_direction not in ("LONG", "SHORT") or not plan:
        return {
            "signal": "WAIT",
            "reason": "Direction is not strong enough",
            "projected_profit": 0.0,
            "projected_risk": 0.0,
        }

    profit = pnl_between(
        plan["entry"], plan["target"],
        tick_size, tick_value, contracts
    )
    risk = pnl_between(
        plan["entry"], plan["stop"],
        tick_size, tick_value, contracts
    )

    failures = []

    if confluence_index < minimum_confluence_index:
        failures.append(f"confluence < {minimum_confluence_index:.0f}/100")

    if dominance < minimum_dominance:
        failures.append(f"directional dominance < {minimum_dominance:.0f}/100")

    if profit < minimum_profit:
        failures.append(f"projected TP < ${minimum_profit:,.0f}")

    if plan["rr"] < minimum_rr:
        failures.append(f"R:R < {minimum_rr:.2f}:1")

    if plan["atr_multiple"] < minimum_atr_multiple:
        failures.append(f"target < {minimum_atr_multiple:.2f} ATR")

    if failures:
        return {
            "signal": "WAIT",
            "reason": "; ".join(failures),
            "projected_profit": profit,
            "projected_risk": risk,
        }

    return {
        "signal": raw_direction,
        "reason": "STRICT CONFLUENCE PASSED",
        "projected_profit": profit,
        "projected_risk": risk,
    }


# ============================================================
# DATA LOADER
# ============================================================

def load_all_data(symbol, prefer_topstep=True):
    username, api_key = get_topstep_credentials()
    token = None
    contract = None
    error = None

    if prefer_topstep and username and api_key:
        token, error = px_login(username, api_key)
        if token:
            try:
                contract = px_find_contract(token, symbol, live=True)
            except Exception as e:
                error = f"Contract search failed: {e}"
                contract = None

    frames = {}

    if token and contract:
        mode = "TopstepX / ProjectX live bars"

        for tf in TIMEFRAMES:
            try:
                frames[tf] = px_retrieve_bars(
                    token, contract["id"], tf, live=True
                )
            except Exception as e:
                frames[tf] = pd.DataFrame()
                error = f"{tf} live-data error: {e}"

        root = clean_root(symbol)
        fallback = FALLBACK_SPECS.get(root, {})

        tick_size = float(
            contract.get("tickSize")
            or fallback.get("tick_size", 0.25)
        )
        tick_value = float(
            contract.get("tickValue")
            or fallback.get("tick_value", 1.0)
        )
        resolved = contract.get("name") or symbol

    else:
        mode = "Yahoo fallback — proxy/delayed"

        for tf in TIMEFRAMES:
            try:
                frames[tf] = yahoo_bars(symbol, tf)
            except Exception:
                frames[tf] = pd.DataFrame()

        root = clean_root(symbol)
        spec = FALLBACK_SPECS.get(root, {
            "tick_size": 0.01,
            "tick_value": 1.0,
            "yf": symbol,
        })
        tick_size = float(spec["tick_size"])
        tick_value = float(spec["tick_value"])
        resolved = spec.get("yf", symbol)

    return {
        "frames": frames,
        "mode": mode,
        "error": error,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "resolved": resolved,
        "contract": contract,
    }



# ============================================================
# CROSS-TIMEFRAME CONFLUENCE
# ============================================================

TF_ORDER = ["1m", "5m", "15m", "30m", "4h", "1d", "1w"]

def higher_timeframe_alignment(raw_results, tf):
    """
    Returns a small supporting adjustment based on adjacent/higher timeframes.
    This does NOT override a timeframe's own structure.
    """
    if tf not in TF_ORDER:
        return 0.0, "none"

    i = TF_ORDER.index(tf)
    direction = raw_results.get(tf, {}).get("raw", "WAIT")
    if direction not in ("LONG", "SHORT"):
        return 0.0, "none"

    checks = []
    for higher in TF_ORDER[i+1:i+3]:
        hdir = raw_results.get(higher, {}).get("raw", "WAIT")
        if hdir in ("LONG", "SHORT"):
            checks.append(hdir)

    if not checks:
        return 0.0, "no higher-timeframe confirmation"

    same = sum(1 for x in checks if x == direction)
    opposite = sum(1 for x in checks if x != direction)

    if same >= 2:
        return 7.5, "strong higher-timeframe alignment"
    if same == 1 and opposite == 0:
        return 4.0, "higher-timeframe alignment"
    if opposite >= 1:
        return -7.5, "higher-timeframe conflict"
    return 0.0, "mixed higher-timeframe evidence"



# ============================================================
# CATALYST CONTEXT + USER'S 1-MINUTE FVG MIDPOINT SCALPER
# ============================================================

POSITIVE_WORDS = {
    "beat", "beats", "upgrade", "upgraded", "record", "growth", "surge",
    "raises", "raised", "approval", "approved", "partnership", "contract",
    "strong", "profit", "profits", "bullish", "buyback"
}
NEGATIVE_WORDS = {
    "miss", "misses", "downgrade", "downgraded", "cuts", "cut", "lawsuit",
    "probe", "investigation", "weak", "loss", "losses", "warning", "recall",
    "bearish", "bankruptcy", "fraud"
}

@st.cache_data(ttl=180, show_spinner=False)
def headline_catalyst(symbol):
    """Small capped headline tilt for supporting higher-timeframe context only."""
    try:
        root = clean_root(symbol)
        query_symbol = FALLBACK_SPECS.get(root, {}).get("yf", symbol)
        items = yf.Ticker(query_symbol).news or []
    except Exception:
        return 0.0, [], "Catalyst feed unavailable"

    score = 0.0
    headlines = []
    for item in items[:8]:
        title = str(item.get("title") or item.get("content", {}).get("title") or "").strip()
        if not title:
            continue
        headlines.append(title)
        words = {w.strip(".,:;!?()[]{}\"'").lower() for w in title.split()}
        score += 0.18 * len(words & POSITIVE_WORDS)
        score -= 0.18 * len(words & NEGATIVE_WORDS)
    score = max(-0.75, min(0.75, score))
    note = "positive tilt" if score > 0.15 else "negative tilt" if score < -0.15 else "neutral / mixed"
    return score, headlines[:6], note


def is_bullish(row):
    return float(row["Close"]) > float(row["Open"])


def is_bearish(row):
    return float(row["Close"]) < float(row["Open"])


def hammer(row, wick_body_ratio=2.0):
    body = max(abs(float(row["Close"]) - float(row["Open"])), 1e-12)
    lower = min(float(row["Open"]), float(row["Close"])) - float(row["Low"])
    upper = float(row["High"]) - max(float(row["Open"]), float(row["Close"]))
    return lower >= wick_body_ratio * body and lower > upper


def shooting_star(row, wick_body_ratio=2.0):
    body = max(abs(float(row["Close"]) - float(row["Open"])), 1e-12)
    upper = float(row["High"]) - max(float(row["Open"]), float(row["Close"]))
    lower = min(float(row["Open"]), float(row["Close"])) - float(row["Low"])
    return upper >= wick_body_ratio * body and upper > lower


def scalp_confirmation(completed, direction, wick_body_ratio=2.0):
    if completed is None or len(completed) < 2:
        return False, "Need two completed 1-minute candles"
    a, b = completed.iloc[-2], completed.iloc[-1]
    if direction == "LONG":
        if is_bullish(a) and is_bullish(b):
            return True, "Two consecutive bullish candles"
        if hammer(a, wick_body_ratio) and is_bullish(b):
            return True, "Hammer + bullish follow-through"
        return False, "Waiting for two bullish candles or hammer + bullish candle"
    if direction == "SHORT":
        if is_bearish(a) and is_bearish(b):
            return True, "Two consecutive bearish candles"
        if shooting_star(a, wick_body_ratio) and is_bearish(b):
            return True, "Shooting star + bearish follow-through"
        return False, "Waiting for two bearish candles or shooting star + bearish candle"
    return False, "No directional target"


def unfilled_1m_fvgs(completed, min_volume_ratio=1.5, min_size_ratio=1.25,
                     micro_min_size_ratio=0.55, min_displacement_ratio=1.2, avg_window=20):
    """Return active 3-candle 1m FVGs, tagged CORE or MICRO.

    CORE gaps meet the normal size filter. MICRO gaps may be smaller but still must
    meet the same volume/displacement filters. A gap is considered consumed once its
    midpoint has traded through after formation.
    """
    if completed is None or len(completed) < avg_window + 5:
        return []
    d = completed.copy()
    d["Body"] = (d["Close"] - d["Open"]).abs()
    d["AvgVolume"] = d["Volume"].rolling(avg_window).mean().shift(1)
    d["AvgBody"] = d["Body"].rolling(avg_window).mean().shift(1)

    raw = []
    prior_sizes = []
    for i in range(2, len(d)):
        low = high = None
        kind = None
        if float(d.iloc[i]["Low"]) > float(d.iloc[i-2]["High"]):
            low, high, kind = float(d.iloc[i-2]["High"]), float(d.iloc[i]["Low"]), "BULLISH"
        elif float(d.iloc[i]["High"]) < float(d.iloc[i-2]["Low"]):
            low, high, kind = float(d.iloc[i]["High"]), float(d.iloc[i-2]["Low"]), "BEARISH"
        if kind is None:
            continue
        size = high - low
        ref_size = np.mean(prior_sizes[-20:]) if prior_sizes else size
        prior_sizes.append(size)
        av = safe_float(d.iloc[i]["AvgVolume"], 0.0)
        ab = safe_float(d.iloc[i]["AvgBody"], 0.0)
        vr = safe_float(d.iloc[i]["Volume"], 0.0) / av if av > 0 else 0.0
        dr = safe_float(d.iloc[i]["Body"], 0.0) / ab if ab > 0 else 0.0
        sr = size / ref_size if ref_size > 0 else 1.0
        if vr < min_volume_ratio or dr < min_displacement_ratio or sr < micro_min_size_ratio:
            continue
        tier = "CORE" if sr >= min_size_ratio else "MICRO"
        midpoint = (low + high) / 2.0
        later = d.iloc[i+1:]
        consumed = False if later.empty else bool(((later["Low"] <= midpoint) & (later["High"] >= midpoint)).any())
        if not consumed:
            raw.append({
                "formed_at": d.index[i], "low": low, "high": high, "midpoint": midpoint,
                "kind": kind, "tier": tier, "volume_ratio": vr, "size_ratio": sr,
                "displacement_ratio": dr,
            })
    return raw


def nearest_fvg_midpoint_target(fvgs, price):
    if not fvgs or not np.isfinite(price):
        return None, None
    valid = [g for g in fvgs if abs(g["midpoint"] - price) > 1e-12]
    if not valid:
        return None, None
    g = min(valid, key=lambda x: abs(x["midpoint"] - price))
    return g, "LONG" if g["midpoint"] > price else "SHORT"


def feed_age_seconds(df):
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return np.inf
    ts = df.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds())


def max_risk_stop(entry, direction, max_risk, quantity, tick_size, tick_value, is_futures):
    quantity = max(1, int(quantity))
    if is_futures:
        point_value = (tick_value / tick_size) if tick_size > 0 else 0.0
        distance = max_risk / (point_value * quantity) if point_value > 0 else np.nan
    else:
        distance = max_risk / quantity  # $1 move = $1/share
    if not np.isfinite(distance) or distance <= 0:
        return np.nan
    stop = entry - distance if direction == "LONG" else entry + distance
    return round_to_tick(stop, tick_size)


def one_minute_scalp(df1, price, max_risk, quantity, tick_size, tick_value, is_futures,
                     proximity_value, proximity_units, min_volume_ratio, min_size_ratio,
                     micro_min_size_ratio, micro_entry_value, micro_entry_units,
                     max_micro_target_dollars, min_displacement_ratio, wick_body_ratio,
                     max_feed_age=180):
    if df1 is None or len(df1) < 30:
        return {"signal":"WAIT", "reason":"Not enough 1-minute data", "target":None, "stop":None, "fvg":None}

    # newest bar is used for current price; completed bars alone generate candle confirmation.
    completed = df1.iloc[:-1].copy() if len(df1) > 1 else df1.copy()
    fvgs = unfilled_1m_fvgs(
        completed, min_volume_ratio=min_volume_ratio, min_size_ratio=min_size_ratio,
        micro_min_size_ratio=micro_min_size_ratio, min_displacement_ratio=min_displacement_ratio,
    )
    target_fvg, direction = nearest_fvg_midpoint_target(fvgs, price)
    if not target_fvg:
        return {"signal":"WAIT", "reason":"No qualifying unfilled 1-minute FVG midpoint available", "target":None, "stop":None, "fvg":None, "fvgs":fvgs}

    target = round_to_tick(target_fvg["midpoint"], tick_size)
    distance = abs(target - price)
    radius = float(proximity_value) * tick_size if proximity_units == "ticks" else float(proximity_value)
    micro_radius = float(micro_entry_value) * tick_size if micro_entry_units == "ticks" else float(micro_entry_value)
    confirmed, confirmation_note = scalp_confirmation(completed, direction, wick_body_ratio)
    age = feed_age_seconds(df1)
    tier = target_fvg.get("tier", "CORE")
    projected_target_dollars = pnl_between(price, target, tick_size, tick_value, quantity) if is_futures else distance * max(1, int(quantity))

    # CORE = normal confirmation rule. MICRO = may enter without full candle confirmation,
    # but only when price is already very close and the intended dollar target is small.
    micro_override = (
        tier == "MICRO"
        and distance <= micro_radius
        and projected_target_dollars <= float(max_micro_target_dollars)
    )

    if age > max_feed_age:
        signal, reason = "WAIT", f"Market data is stale ({age:.0f}s old)"
    elif micro_override:
        signal = direction
        reason = f"MICRO FVG scalp: price is very close to midpoint; reduced-target entry allowed without full confirmation (${projected_target_dollars:,.0f} projected)"
    elif distance > radius:
        signal, reason = "WAIT", f"Nearest FVG is {distance:.2f} away; confirmation zone begins within {radius:.2f}"
    elif not confirmed:
        signal, reason = "WAIT", confirmation_note
    else:
        signal, reason = direction, confirmation_note

    stop = max_risk_stop(price, direction, max_risk, quantity, tick_size, tick_value, is_futures) if direction else np.nan
    return {
        "signal": signal, "candidate": direction, "reason": reason, "target": target, "stop": stop,
        "fvg": target_fvg, "fvgs": fvgs, "distance": distance, "radius": radius,
        "micro_radius": micro_radius, "tier": tier, "micro_override": micro_override,
        "projected_target_dollars": projected_target_dollars,
        "confirmed": confirmed, "confirmation_note": confirmation_note, "age": age,
    }


def catalyst_adjusted_higher_tf(evidence, catalyst_score):
    """Catalyst is supporting context; it cannot independently manufacture a trade."""
    raw = evidence.get("raw", "WAIT")
    score = float(evidence.get("score", 0.0)) + float(catalyst_score) * 1.25
    if raw == "WAIT":
        if score >= 5.25:
            raw = "LONG"
        elif score <= -5.25:
            raw = "SHORT"
    return raw, score



# ============================================================
# USER INTERFACE — 1m FIRST, EVERYTHING ELSE SUPPORTING
# ============================================================

st.title("OP.exe · 1-Minute FVG Scalper")
st.caption("Primary output = your live 1-minute FVG-midpoint scalp. Higher timeframes, structure and catalysts stay visible as supporting data.")

with st.form("op_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([2.1, 1.0, 1.1, 1.15])
    symbol = c1.text_input("Ticker / contract", value=st.session_state.get("symbol", "MNQ"), placeholder="MNQ, MES, MGC, AAPL, SPY, BTC-USD")
    quantity = int(c2.number_input("Contracts / shares", min_value=1, max_value=10000, value=1, step=1))
    max_risk = float(c3.number_input("Max planned risk ($)", min_value=1.0, max_value=100000.0, value=300.0, step=25.0))
    run_button = c4.form_submit_button("Analyze", use_container_width=True)

if run_button:
    st.session_state["symbol"] = symbol.upper().strip()
symbol = st.session_state.get("symbol", symbol.upper().strip() or "MNQ")

with st.expander("1-minute strategy settings", expanded=False):
    s1,s2,s3,s4 = st.columns(4)
    proximity_value = s1.number_input("Normal confirmation distance", min_value=0.25, value=3.0, step=0.25)
    proximity_units = s2.selectbox("Normal distance units", ["ticks", "price points"], index=0)
    min_volume_ratio = s3.number_input("Min FVG volume × avg", min_value=1.0, value=1.5, step=0.1)
    min_size_ratio = s4.number_input("CORE FVG size × recent", min_value=0.5, value=1.25, step=0.05)
    s5,s6,s7,s8 = st.columns(4)
    micro_min_size_ratio = s5.number_input("MICRO FVG min size × recent", min_value=0.10, value=0.55, step=0.05)
    micro_entry_value = s6.number_input("No-confirmation micro distance", min_value=0.25, value=1.0, step=0.25)
    micro_entry_units = s7.selectbox("Micro distance units", ["ticks", "price points"], index=0)
    max_micro_target_dollars = s8.number_input("Max no-confirmation target $", min_value=1.0, value=100.0, step=10.0)
    s9,s10,s11,s12 = st.columns(4)
    min_displacement_ratio = s9.number_input("Min displacement × avg body", min_value=0.5, value=1.2, step=0.1)
    wick_body_ratio = s10.number_input("Hammer/star wick ÷ body", min_value=1.0, value=2.0, step=0.25)
    max_feed_age = int(s11.number_input("Max 1m feed age (sec)", min_value=60, max_value=1800, value=180, step=30))
    auto_refresh = s12.toggle("Auto refresh", value=False)
    prefer_topstep = st.toggle("Prefer TopstepX / ProjectX live bars when credentials are configured", value=True)

if auto_refresh:
    st_autorefresh(interval=15000, key="op_refresh")

with st.spinner("Reading market data and running OP.exe..."):
    bundle = load_all_data(symbol, prefer_topstep=prefer_topstep)

frames = bundle["frames"]
df1 = frames.get("1m", pd.DataFrame())
tick_size = float(bundle["tick_size"])
tick_value = float(bundle["tick_value"])
mode = bundle["mode"]
is_futures = bundle.get("contract") is not None or clean_root(symbol) in FALLBACK_SPECS
market_open, market_label, market_reason = market_session_status(symbol, bundle.get("contract"))

last_price = np.nan
for tf in ["1m","5m","15m","30m","4h","1d","1w"]:
    d = frames.get(tf, pd.DataFrame())
    if d is not None and not d.empty:
        last_price = float(d["Close"].iloc[-1]); break

cat_score, headlines, cat_note = headline_catalyst(symbol)

scalp = one_minute_scalp(
    df1, last_price, max_risk, quantity, tick_size, tick_value, is_futures,
    proximity_value, proximity_units, min_volume_ratio, min_size_ratio,
    micro_min_size_ratio, micro_entry_value, micro_entry_units, max_micro_target_dollars,
    min_displacement_ratio, wick_body_ratio, max_feed_age=max_feed_age,
)

# Never issue an actionable scalp signal when the relevant market/session is closed.
if not market_open:
    scalp["signal"] = "WAIT"
    scalp["reason"] = f"{market_label} — {market_reason}"

# top status strip
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("1m SIGNAL", scalp.get("signal","WAIT"))
m2.metric("Current price", f"{last_price:,.2f}" if np.isfinite(last_price) else "—")
m3.metric("Take profit", f"{scalp['target']:,.2f}" if scalp.get("target") is not None else "—")
m4.metric("Max-risk stop", f"{scalp['stop']:,.2f}" if scalp.get("stop") is not None and np.isfinite(scalp.get("stop",np.nan)) else "—")
m5.metric("Market", market_label)
feed_age = feed_age_seconds(df1)
if mode.startswith("TopstepX") and market_open and np.isfinite(feed_age) and feed_age <= max_feed_age:
    data_label = "LIVE / FRESH"
elif market_open and np.isfinite(feed_age) and feed_age <= max_feed_age:
    data_label = "FRESH FALLBACK"
elif not market_open:
    data_label = "CLOSED"
else:
    data_label = "STALE / NO LIVE"
m6.metric("Data status", data_label)

sig = scalp.get("signal","WAIT")
if not market_open:
    st.warning(f"MARKET CLOSED — {market_reason} OP.exe will not issue LONG/SHORT while closed.")
elif np.isfinite(feed_age) and feed_age > max_feed_age:
    st.warning(f"LIVE DATA NOT FRESH — newest 1m bar is {feed_age:.0f}s old. OP.exe forces WAIT.")

if sig == "LONG":
    st.success(f"LONG — {scalp['reason']} · Target nearest unfilled 1m FVG midpoint at {scalp['target']:,.2f}.")
elif sig == "SHORT":
    st.error(f"SHORT — {scalp['reason']} · Target nearest unfilled 1m FVG midpoint at {scalp['target']:,.2f}.")
else:
    st.warning(f"WAIT — {scalp.get('reason','No valid confirmation')}")

if mode.startswith("TopstepX"):
    st.caption("TopstepX / ProjectX is the futures data source. OP.exe labels it LIVE only when the market is open and the newest 1-minute bar passes the freshness limit. Completed candles generate confirmation; the newest bar supplies current price.")
else:
    st.caption("Yahoo/fallback data can be delayed or proxy data. It is never labeled Topstep LIVE. OP.exe forces WAIT if the market is closed or the newest 1-minute bar is stale.")

if bundle.get("error"):
    st.caption(bundle["error"])

fvg = scalp.get("fvg")
if fvg:
    f1,f2,f3,f4,f5 = st.columns(5)
    f1.metric("Nearest FVG", f"{fvg['low']:,.2f}–{fvg['high']:,.2f}")
    f2.metric("Midpoint target", f"{fvg['midpoint']:,.2f}")
    f3.metric("Tier", fvg.get("tier", "CORE"))
    f4.metric("Formation volume", f"{fvg['volume_ratio']:.2f}× avg")
    f5.metric("Target value", f"${scalp.get('projected_target_dollars',0):,.0f}")

st.caption(f"Risk setting: ${max_risk:,.0f} max planned loss across {quantity} contract(s)/share(s). The displayed stop is the price-distance equivalent before slippage/fees; exit earlier if the setup is invalidated.")

# ============================================================
# LIVE TRADE TRACKER — intermediate FVG exits and re-entry
# ============================================================

if "tracked_trade" not in st.session_state:
    st.session_state["tracked_trade"] = None

st.subheader("Live scalp management")
track_c1, track_c2, track_c3 = st.columns([1.1, 1.1, 2.8])
can_track = sig in ("LONG", "SHORT") and scalp.get("target") is not None and np.isfinite(last_price)
if track_c1.button("Track this trade", disabled=not can_track, use_container_width=True):
    st.session_state["tracked_trade"] = {
        "direction": sig,
        "entry": float(last_price),
        "original_target": float(scalp["target"]),
        "started_at": pd.Timestamp.now(tz="UTC"),
        "source_fvg_formed_at": scalp.get("fvg", {}).get("formed_at"),
    }
if track_c2.button("Reset tracked trade", use_container_width=True):
    st.session_state["tracked_trade"] = None

tracked = st.session_state.get("tracked_trade")
if not tracked:
    track_c3.caption("Track a LONG/SHORT signal to make OP.exe watch for newly formed FVG midpoints as temporary profit-taking levels before the original target.")
else:
    direction = tracked["direction"]
    entry_price = float(tracked["entry"])
    original_target = float(tracked["original_target"])
    in_profit = (last_price > entry_price) if direction == "LONG" else (last_price < entry_price)
    live_completed = df1.iloc[:-1].copy() if df1 is not None and len(df1) > 1 else pd.DataFrame()
    all_live_fvgs = unfilled_1m_fvgs(
        live_completed, min_volume_ratio=min_volume_ratio, min_size_ratio=min_size_ratio,
        micro_min_size_ratio=micro_min_size_ratio, min_displacement_ratio=min_displacement_ratio,
    )
    new_gaps = [g for g in all_live_fvgs if pd.Timestamp(g["formed_at"]) >= pd.Timestamp(tracked["started_at"])]

    # A newly formed FVG midpoint is an intermediate profit level only if it lies between
    # the entry and the original target in the direction of the trade.
    if direction == "LONG":
        intermediate = [g for g in new_gaps if entry_price < g["midpoint"] < original_target]
    else:
        intermediate = [g for g in new_gaps if original_target < g["midpoint"] < entry_price]

    reached = []
    if not live_completed.empty:
        for g in intermediate:
            after = live_completed.loc[live_completed.index >= pd.Timestamp(g["formed_at"])]
            if not after.empty and ((after["Low"] <= g["midpoint"]) & (after["High"] >= g["midpoint"])).any():
                reached.append(g)

    tc1,tc2,tc3,tc4 = st.columns(4)
    tc1.metric("Tracked side", direction)
    tc2.metric("Tracked entry", f"{entry_price:,.2f}")
    tc3.metric("Original midpoint", f"{original_target:,.2f}")
    tc4.metric("Open P/L direction", "PROFIT" if in_profit else "NOT PROFIT")

    if in_profit and reached:
        latest_exit = sorted(reached, key=lambda g: pd.Timestamp(g["formed_at"]))[-1]
        st.success(
            f"TAKE PROFIT / PAUSE — price reached a newly formed FVG midpoint at {latest_exit['midpoint']:,.2f} while the tracked trade is in profit. "
            f"After exiting, wait for a fresh {direction} setup and re-enter only if valid toward the original midpoint {original_target:,.2f}."
        )
    elif intermediate:
        nearest_intermediate = min(intermediate, key=lambda g: abs(g["midpoint"]-last_price))
        st.info(
            f"Intermediate FVG formed during the trade. Its midpoint is {nearest_intermediate['midpoint']:,.2f}. "
            "If the trade is in profit when price reaches that midpoint, treat it as a temporary profit-taking level, then re-confirm before targeting the original midpoint again."
        )
    else:
        st.caption("No qualifying intermediate FVG midpoint has formed between the tracked entry and original target yet.")

# supporting timeframe engine
raw_results = {}
rows = []
for tf in ["5m","15m","30m","4h","1d","1w"]:
    d = frames.get(tf, pd.DataFrame())
    if d is None or d.empty:
        rows.append({"Timeframe":tf,"Signal":"NO DATA","Score":"—","Catalyst":"—","Top evidence":"Unavailable"})
        continue
    e = evidence_engine(d, tf)
    adjusted_signal, adjusted_score = catalyst_adjusted_higher_tf(e, cat_score)
    raw_results[tf] = e
    rows.append({
        "Timeframe":tf,
        "Signal":adjusted_signal,
        "Score":round(adjusted_score,2),
        "Catalyst":f"{cat_score:+.2f}",
        "Top evidence":"; ".join(e.get("reasons",[])[:4]) or "Mixed / neutral",
    })

st.subheader("Supporting timeframe data")
st.caption("These do not override your 1-minute scalp signal. They preserve the older OP.exe market-structure and catalyst context.")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("All active unfilled 1-minute FVGs", expanded=False):
    active = scalp.get("fvgs", [])
    if not active:
        st.write("No qualifying CORE or MICRO unfilled 1-minute FVGs passed the current filters.")
    else:
        frows=[]
        for g in sorted(active, key=lambda x: abs(x["midpoint"]-last_price))[:20]:
            frows.append({
                "Direction from price":"LONG" if g["midpoint"] > last_price else "SHORT",
                "Tier":g.get("tier","CORE"),
                "Low":round(g["low"],4), "High":round(g["high"],4), "Midpoint":round(g["midpoint"],4),
                "Distance":round(abs(g["midpoint"]-last_price),4), "Volume ×":round(g["volume_ratio"],2),
                "Size ×":round(g["size_ratio"],2), "Displacement ×":round(g["displacement_ratio"],2),
                "Formed":str(g["formed_at"]),
            })
        st.dataframe(pd.DataFrame(frows), use_container_width=True, hide_index=True)

with st.expander("Catalysts, structure and diagnostics", expanded=False):
    st.write(f"**Catalyst context:** {cat_score:+.2f} — {cat_note}. Catalyst is supporting evidence only and cannot create the 1-minute scalp signal by itself.")
    if headlines:
        for h in headlines:
            st.write("•", h)
    else:
        st.write("No usable recent headlines were returned.")

    diag=[]
    for tf in ["1m","5m","15m","30m","4h","1d","1w"]:
        d=frames.get(tf,pd.DataFrame())
        if d is None or d.empty: continue
        e=evidence_engine(d,tf)
        diag.append({
            "TF":tf,"Raw":e.get("raw"),"BOS":e.get("bos"),"CHoCH":e.get("choch"),"Sweep":e.get("sweep"),
            "Displacement":e.get("displacement"),"RSI":round(e.get("rsi",np.nan),1) if np.isfinite(e.get("rsi",np.nan)) else None,
            "ADX":round(e.get("adx",np.nan),1) if np.isfinite(e.get("adx",np.nan)) else None,
            "Volume ×":round(e.get("volume_ratio",1.0),2),"Regime":e.get("regime"),
        })
    st.dataframe(pd.DataFrame(diag), use_container_width=True, hide_index=True)

with st.expander("Data setup / secrets", expanded=False):
    st.code('TOPSTEP_USERNAME = "your username"\nTOPSTEP_API_KEY = "your API key"', language="toml")
    st.caption("Put these in Streamlit Secrets, never in GitHub. Stocks/ETFs/crypto use Yahoo-compatible symbols; futures prefer TopstepX when credentials resolve a contract.")

st.divider()
st.caption("OP.exe is a rules-based analysis tool, not an order-entry bot. LONG/SHORT/WAIT and catalyst/confluence scores are not guarantees. Slippage can make realized loss exceed a planned stop amount.")

