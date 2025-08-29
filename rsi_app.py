import time
from typing import List, Dict

import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="RSI & Moving Averages (MVP)", layout="wide")
st.title("📈 RSI • Moving Averages • 52-Week Levels (MVP)")

# ----------------------------
# Helpers
# ----------------------------
def _clean_tickers(s: str, cap: int = 10) -> List[str]:
    tickers = [t.strip().upper() for t in s.split(",") if t.strip()]
    if len(tickers) > cap:
        st.warning(f"Limiting to the first {cap} tickers to avoid rate limits.")
        tickers = tickers[:cap]
    # yfinance struggles with spaces or weird chars — filter
    tickers = [t for t in tickers if t.replace(".", "").isalnum()]
    return tickers

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def fetch_data_batch(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Batch download to minimize rate limiting. Cached for 1 hour.
    Returns a DataFrame; if multiple tickers, columns are a MultiIndex (ticker, field).
    """
    # A couple of polite retries in case Yahoo is grumpy
    delays = [0, 3, 6]
    last_err = None
    for d in delays:
        try:
            if d:
                time.sleep(d)
            df = yf.download(
                tickers,
                period=period,
                group_by="ticker",
                auto_adjust=True,
                threads=False,  # reduce hammering
                progress=False,
            )
            return df
        except Exception as e:
            last_err = e
    raise last_err

@st.cache_data(ttl=1800)
def get_fast_info_safe(ticker: str) -> Dict:
    """Tiny fundamentals snapshot using fast_info; cached to avoid repeated calls."""
    try:
        fi = yf.Ticker(ticker).fast_info
        # Convert to regular dict (fast_info is a SimpleNamespace-like)
        return {
            "last_price": getattr(fi, "last_price", None),
            "market_cap": getattr(fi, "market_cap", None),
            "trailing_pe": getattr(fi, "trailing_pe", None),
            "forward_pe": getattr(fi, "forward_pe", None),
            "dividend_yield": getattr(fi, "dividend_yield", None),
        }
    except Exception:
        return {}

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a single-ticker DataFrame with columns [Open, High, Low, Close, Volume],
    compute RSI(14), SMA50, SMA200, 52w High/Low, and a Cross signal.
    """
    out = pd.DataFrame(index=df.index.copy())
    out["Close"] = df["Close"]
    out["RSI14"] = rsi(df["Close"], 14)
    out["SMA50"] = df["Close"].rolling(50).mean()
    out["SMA200"] = df["Close"].rolling(200).mean()

    # Last-year window (approx 252 trading days)
    last_252 = df.tail(252)
    out.attrs["52w_high"] = float(last_252["High"].max()) if not last_252.empty else None
    out.attrs["52w_low"] = float(last_252["Low"].min()) if not last_252.empty else None

    # Cross signal (use last 2 points if available)
    def cross(sig50, sig200):
        if len(sig50.dropna()) < 2 or len(sig200.dropna()) < 2:
            return "N/A"
        if sig50.iloc[-1] > sig200.iloc[-1] and sig50.iloc[-2] <= sig200.iloc[-2]:
            return "🟢 Golden Cross"
        if sig50.iloc[-1] < sig200.iloc[-1] and sig50.iloc[-2] >= sig200.iloc[-2]:
            return "🔴 Death Cross"
        return "⚪ No recent cross"

    out.attrs["cross"] = cross(out["SMA50"], out["SMA200"])
    return out

def extract_single(df_all: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    If multiple tickers were downloaded, df_all has MultiIndex columns (ticker, field).
    If only one ticker, df_all already has flat columns.
    This returns a flat single-ticker frame with columns Open/High/Low/Close/Volume.
    """
    if isinstance(df_all.columns, pd.MultiIndex):
        if ticker not in df_all.columns.get_level_values(0):
            raise KeyError(f"{ticker} not in downloaded data.")
        df = df_all[ticker].copy()
    else:
        df = df_all.copy()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")

def display_ticker_panel(ticker: str, df_all: pd.DataFrame):
    try:
        df = extract_single(df_all, ticker)
        if df.empty:
            st.warning(f"{ticker}: no data returned.")
            return

        sig = compute_signals(df)
        latest_close = sig["Close"].iloc[-1]
        latest_rsi = sig["RSI14"].iloc[-1]
        cross = sig.attrs["cross"]
        high52 = sig.attrs["52w_high"]
        low52 = sig.attrs["52w_low"]

        st.markdown(f"### {ticker}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", f"${latest_close:,.2f}")
        c2.metric("RSI(14)", f"{latest_rsi:,.2f}")
        c3.metric("52w High / Low", f"${high52:,.2f} / ${low52:,.2f}" if high52 and low52 else "—")
        c4.metric("MA Signal", cross)

        # Price + MAs
        st.line_chart(sig[["Close", "SMA50", "SMA200"]])

        # RSI
        st.line_chart(sig[["RSI14"]])
        st.caption("RSI guides: >70 overbought, <30 oversold.")

    except KeyError as e:
        st.error(f"{ticker}: {e}")
    except Exception as e:
        st.error(f"{ticker}: {e}")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Fundamentals (light)", "🗒️ Watchlists"])

# ----------------------------
# Tab 1: Analysis
# ----------------------------
with tab1:
    st.subheader("🔍 RSI(14), SMA(50/200), 52-Week Levels")
    default = st.session_state.get("selected_tickers", "AAPL, MSFT, TSLA")
    t_input = st.text_input("Enter tickers (comma-separated):", default)
    tickers = _clean_tickers(t_input, cap=10)

    if st.button("Run Analysis", type="primary"):
        if not tickers:
            st.warning("Please enter at least one ticker.")
        else:
            try:
                data = fetch_data_batch(tickers, period="1y")
                for t in tickers:
                    display_ticker_panel(t, data)
            except Exception as e:
                st.error(f"Download error: {e}")

# ----------------------------
# Tab 2: Fundamentals (light)
# ----------------------------
with tab2:
    st.subheader("📊 Quick Fundamentals (cached & rate-limit friendly)")

    t_input_f = st.text_input("Tickers (comma-separated):", "AAPL, MSFT", key="funds_input")
    tickers_f = _clean_tickers(t_input_f, cap=8)

    if st.button("Get Fundamentals"):
        rows = []
        for t in tickers_f:
            fi = get_fast_info_safe(t)
            if not fi:
                rows.append({"Ticker": t, "Last Price": "—", "Mkt Cap": "—", "Trailing PE": "—", "Forward PE": "—", "Div Yield": "—"})
                continue
            rows.append({
                "Ticker": t,
                "Last Price": fi.get("last_price"),
                "Mkt Cap": fi.get("market_cap"),
                "Trailing PE": fi.get("trailing_pe"),
                "Forward PE": fi.get("forward_pe"),
                "Div Yield": fi.get("dividend_yield"),
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

# ----------------------------
# Tab 3: Watchlists
# ----------------------------
with tab3:
    st.subheader("🗒️ Build a Watchlist and Rank by RSI")
    st.caption("Paste tickers or upload a CSV with a 'Symbol' column.")

    colA, colB = st.columns(2)
    with colA:
        paste = st.text_area("Paste tickers (comma-separated):", "AAPL, MSFT, NVDA, META, AMZN")
    with colB:
        uploaded = st.file_uploader("Upload CSV (with 'Symbol' column)", type=["csv"])

    tickers_w = _clean_tickers(paste) if paste else []
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "Symbol" in df_up.columns:
                extra = _clean_tickers(",".join(df_up["Symbol"].astype(str).tolist()))
                tickers_w = list(dict.fromkeys(tickers_w + extra))  # dedupe + preserve order
            else:
                st.warning("CSV missing 'Symbol' column.")
        except Exception as e:
            st.error(f"CSV error: {e}")

    if st.button("Analyze Watchlist"):
        if not tickers_w:
            st.warning("No tickers detected.")
        else:
            try:
                data = fetch_data_batch(tickers_w, period="6mo")
                rows = []
                for t in tickers_w[:25]:  # keep it snappy
                    try:
                        df = extract_single(data, t)
                        if df.empty:
                            continue
                        rs = rsi(df["Close"], 14).iloc[-1]
                        rows.append({"Ticker": t, "RSI(14)": float(rs)})
                    except Exception:
                        continue
                if rows:
                    out = pd.DataFrame(rows).sort_values("RSI(14)")
                    st.dataframe(out, use_container_width=True)
                    # Save selection for Analysis tab
                    if st.button("Open lowest RSI in Analysis"):
                        low_list = out["Ticker"].head(5).tolist()
                        st.session_state["selected_tickers"] = ",".join(low_list)
                        st.success(f"Loaded {', '.join(low_list)} into Analysis tab input.")
                else:
                    st.info("No RSI results computed.")
            except Exception as e:
                st.error(f"Download error: {e}")

