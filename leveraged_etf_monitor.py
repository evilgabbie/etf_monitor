"""
Leveraged ETF Relationship Monitor
-----------------------------------
Tracks the intraday relationship between an underlying index/stock and its
leveraged bull/bear ETF pair (default: SOX / SOXL / SOXS).

Two independent views, side by side:
  1. Correlation & rolling beta  -> "are they moving together, and how hard?"
  2. Tracking error vs theoretical leverage -> "is the ETF actually delivering
     its stated multiple today, or has daily-reset decay knocked it off?"

Data source: yfinance (unofficial, delayed by seconds-to-~1min, NOT a
real-time feed). If you need true real-time, this is the wrong data source --
you need Interactive Brokers with an active real-time market data
subscription, run locally (not on Streamlit Cloud).

KNOWN GAP: market-hours check below is time-of-day/day-of-week only. It does
NOT know about NYSE/NASDAQ holidays or early closes. On a holiday it will
think the market is open and poll uselessly until it gets empty data back.
Fix this properly with a real market calendar (e.g. `pandas_market_calendars`)
if that gap matters to you.
"""

import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Leveraged ETF Monitor", layout="wide")

NY_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)


def market_status(now_et: datetime | None = None) -> tuple[bool, datetime]:
    """Returns (is_open, next_open_datetime_et). Weekday/time-of-day only --
    does not account for holidays or early closes. See module docstring."""
    now_et = now_et or datetime.now(NY_TZ)
    open_t = now_et.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    close_t = now_et.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)

    is_weekday = now_et.weekday() < 5
    is_open = is_weekday and open_t <= now_et <= close_t

    # crude "next open" estimate for display purposes only
    candidate = now_et
    if is_weekday and now_et < open_t:
        next_open = open_t
    else:
        candidate = now_et + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        next_open = candidate.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)

    return is_open, next_open


# ---------------------------------------------------------------------------
# Sidebar: ticker + parameter selection
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

ticker_index = st.sidebar.text_input("Underlying ticker", value="^SOX").strip().upper()
ticker_bull = st.sidebar.text_input("Bull (long) leveraged ETF", value="SOXL").strip().upper()
ticker_bear = st.sidebar.text_input("Bear (short) leveraged ETF", value="SOXS").strip().upper()
leverage = st.sidebar.number_input(
    "Stated leverage multiple", min_value=1.0, max_value=5.0, value=3.0, step=0.5
)
rolling_window = st.sidebar.slider(
    "Rolling window for correlation/beta (minutes)", min_value=5, max_value=60, value=30
)
refresh_seconds = st.sidebar.slider(
    "Refresh interval (seconds)", min_value=1, max_value=60, value=15,
    help="Below 10s you are pushing against yfinance's undocumented rate limits. "
         "You asked for this floor -- it's here, but expect throttling/blank data "
         "if you actually run it at 1-3s for any length of time."
)
if refresh_seconds < 10:
    st.sidebar.warning(
        f"{refresh_seconds}s refresh: high risk of Yahoo rate-limiting on sustained "
        "polling. If data starts going blank/stale, this is why -- raise it back up."
    )

st.sidebar.divider()

if "running" not in st.session_state:
    st.session_state.running = True

btn_label = "Stop live updates" if st.session_state.running else "Resume live updates"
if st.sidebar.button(btn_label, use_container_width=True):
    st.session_state.running = not st.session_state.running

st.sidebar.caption(
    "Data via yfinance (unofficial Yahoo Finance scrape). Not a real-time feed -- "
    "treat as directional monitoring, not execution-grade pricing."
)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=8, show_spinner=False)
def fetch_intraday(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch today's 1-minute bars for all tickers in a single batched call."""
    data = yf.download(
        tickers=list(tickers),
        period="1d",
        interval="1m",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    return data


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prev_close(tickers: tuple[str, ...]) -> dict[str, float]:
    """Previous session's close for each ticker -- the anchor for theoretical
    leveraged price. Cached longer since this only changes once a day."""
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            out[t] = float(info["previous_close"])
        except Exception:
            out[t] = np.nan
    return out


def extract_close_series(raw: pd.DataFrame, ticker: str, tickers: tuple[str, ...]) -> pd.Series:
    """yfinance returns either a flat frame (single ticker) or a MultiIndex
    frame (multiple tickers) -- handle both, matching the pattern used in the
    RSI screener's _extract_hist helper."""
    if len(tickers) == 1:
        return raw["Close"].dropna()
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            return raw[ticker]["Close"].dropna()
        if ticker in raw.columns.get_level_values(1):
            return raw["Close"][ticker].dropna()
    raise KeyError(f"Could not extract Close series for {ticker}")


# ---------------------------------------------------------------------------
# Main computation + render, wrapped in a fragment so only this section
# re-runs on the refresh timer -- not the whole page.
#
# run_every is resolved OUTSIDE the fragment, at full-script level, so the
# Stop/Start button (which lives outside the fragment and triggers a full
# rerun) can actually change the polling behavior. When paused, run_every=None
# and the fragment simply stops auto-refreshing -- last rendered state stays
# on screen. When the market is closed, we drop to a slow 60s "heartbeat" so
# the app notices when it reopens, instead of hammering Yahoo for data that
# isn't changing.
# ---------------------------------------------------------------------------
is_open, next_open_et = market_status()

if not st.session_state.running:
    effective_run_every = None
elif not is_open:
    effective_run_every = "60s"
else:
    effective_run_every = f"{refresh_seconds}s"


@st.fragment(run_every=effective_run_every)
def live_panel():
    tickers = (ticker_index, ticker_bull, ticker_bear)

    status_col1, status_col2 = st.columns([3, 1])
    with status_col2:
        if not st.session_state.running:
            st.info("PAUSED", icon="⏸️")
        elif not is_open:
            st.warning("MARKET CLOSED", icon="🌙")
        else:
            st.success("LIVE", icon="🟢")

    if not is_open:
        now_et = datetime.now(NY_TZ)
        st.info(
            f"Market is closed (as of {now_et:%Y-%m-%d %H:%M} ET, weekday/hours check only -- "
            f"holidays not accounted for). Next expected open: {next_open_et:%A %Y-%m-%d %H:%M} ET. "
            "Not polling Yahoo Finance while closed."
        )
        return

    if not all(tickers):
        st.warning("Enter all three tickers in the sidebar.")
        return

    try:
        raw = fetch_intraday(tickers)
        prev_close = fetch_prev_close(tickers)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return

    try:
        s_idx = extract_close_series(raw, ticker_index, tickers)
        s_bull = extract_close_series(raw, ticker_bull, tickers)
        s_bear = extract_close_series(raw, ticker_bear, tickers)
    except KeyError as e:
        st.error(f"{e}. Check that the tickers are valid and the market has traded today.")
        return

    if s_idx.empty or s_bull.empty or s_bear.empty:
        st.warning("No intraday data yet -- market may be closed, or tickers are invalid.")
        return

    df = pd.DataFrame({"idx": s_idx, "bull": s_bull, "bear": s_bear}).dropna()
    if df.empty:
        st.warning("No overlapping timestamps across the three tickers yet.")
        return

    prev_idx = prev_close.get(ticker_index, np.nan)
    prev_bull = prev_close.get(ticker_bull, np.nan)
    prev_bear = prev_close.get(ticker_bear, np.nan)

    # ---- Theoretical leverage & tracking error ----
    cum_ret_idx = df["idx"] / prev_idx - 1.0
    theo_bull = prev_bull * (1 + leverage * cum_ret_idx)
    theo_bear = prev_bear * (1 - leverage * cum_ret_idx)

    track_err_bull = (df["bull"] - theo_bull) / theo_bull * 100
    track_err_bear = (df["bear"] - theo_bear) / theo_bear * 100

    # ---- Rolling correlation & beta (on 1-min returns) ----
    ret = df.pct_change().dropna()
    roll_corr_bull = ret["idx"].rolling(rolling_window).corr(ret["bull"])
    roll_corr_bear = ret["idx"].rolling(rolling_window).corr(ret["bear"])

    roll_cov_bull = ret["idx"].rolling(rolling_window).cov(ret["bull"])
    roll_var_idx = ret["idx"].rolling(rolling_window).var()
    roll_beta_bull = roll_cov_bull / roll_var_idx

    roll_cov_bear = ret["idx"].rolling(rolling_window).cov(ret["bear"])
    roll_beta_bear = roll_cov_bear / roll_var_idx

    last_ts = df.index[-1]
    now_utc = datetime.now(timezone.utc)

    with status_col1:
        st.caption(
            f"Last bar: {last_ts} | App refreshed: {now_utc:%H:%M:%S} UTC | "
            f"Refresh every {refresh_seconds}s | yfinance data (not real-time)"
        )

    # ---- Header metrics ----
    # NOTE: delta_color="off" on the leveraged ETFs deliberately -- tracking
    # error isn't a "good=green/bad=red" number, it's just a deviation
    # direction. Using the default red/green here previously implied a
    # negative tracking error was "bad performance," which is misleading:
    # the sign just reflects which way decay pushed the price.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{ticker_index}", f"{df['idx'].iloc[-1]:.2f}",
              f"{cum_ret_idx.iloc[-1]*100:+.2f}% vs prev close")
    c2.metric(f"{ticker_bull}", f"{df['bull'].iloc[-1]:.2f}",
              f"track err {track_err_bull.iloc[-1]:+.2f}%", delta_color="off")
    c3.metric(f"{ticker_bear}", f"{df['bear'].iloc[-1]:.2f}",
              f"track err {track_err_bear.iloc[-1]:+.2f}%", delta_color="off")
    c4.metric("Rolling beta (bull)",
              f"{roll_beta_bull.iloc[-1]:.2f}" if pd.notna(roll_beta_bull.iloc[-1]) else "warming up",
              f"corr {roll_corr_bull.iloc[-1]:.2f}" if pd.notna(roll_corr_bull.iloc[-1]) else "",
              delta_color="off")

    tab1, tab2, tab3 = st.tabs(["Price (normalized)", "Correlation & Beta", "Tracking Error"])

    with tab1:
        norm = df / df.iloc[0] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=norm.index, y=norm["idx"], name=ticker_index))
        fig.add_trace(go.Scatter(x=norm.index, y=norm["bull"], name=ticker_bull))
        fig.add_trace(go.Scatter(x=norm.index, y=norm["bear"], name=ticker_bear))
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="Normalized to 100 at session open",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll_corr_bull.index, y=roll_corr_bull, name=f"Corr: {ticker_bull}"))
        fig.add_trace(go.Scatter(x=roll_corr_bear.index, y=roll_corr_bear, name=f"Corr: {ticker_bear}"))
        fig.add_trace(go.Scatter(x=roll_beta_bull.index, y=roll_beta_bull, name=f"Beta: {ticker_bull}", yaxis="y2"))
        fig.add_trace(go.Scatter(x=roll_beta_bear.index, y=roll_beta_bear, name=f"Beta: {ticker_bear}", yaxis="y2"))
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Rolling correlation", range=[-1.05, 1.05]),
            yaxis2=dict(title="Rolling beta", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{rolling_window}-minute rolling window on 1-min returns. First "
            f"{rolling_window} minutes of the session will show no value (warm-up)."
        )

    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=track_err_bull.index, y=track_err_bull, name=f"{ticker_bull} vs theoretical {leverage:g}x"))
        fig.add_trace(go.Scatter(x=track_err_bear.index, y=track_err_bear, name=f"{ticker_bear} vs theoretical -{leverage:g}x"))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="Tracking error (%) vs theoretical leverage from prior close",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Theoretical price = prior close x (1 +/- leverage x cumulative index return "
            "since prior close). Deviation reflects daily-reset compounding decay, "
            "not a data error -- expect drift to grow on choppy/sideways days."
        )

    with st.expander("Raw data (last 20 bars)"):
        display_df = pd.DataFrame({
            f"{ticker_index}": df["idx"],
            f"{ticker_bull}": df["bull"],
            f"{ticker_bear}": df["bear"],
            "Track err bull %": track_err_bull,
            "Track err bear %": track_err_bear,
            "Roll corr bull": roll_corr_bull,
            "Roll beta bull": roll_beta_bull,
        }).tail(20)
        st.dataframe(display_df, use_container_width=True)


st.title("Leveraged ETF Relationship Monitor")
live_panel()
