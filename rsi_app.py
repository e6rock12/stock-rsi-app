# rsi_app.py — RSI • MAs • Factor Watchlist with FactorBlend & dark-theme color bands
import time
from typing import List, Dict

import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="RSI & Factor Watchlist", layout="wide")
st.title("📈 RSI • Moving Averages • Factor Watchlist (Value • Growth • Momentum)")

# ============================
# Helpers
# ============================
def _clean_tickers(s: str, cap: int = 10) -> List[str]:
    tickers = [t.strip().upper() for t in s.split(",") if t.strip()]
    if len(tickers) > cap:
        st.warning(f"Limiting to the first {cap} tickers to avoid rate limits.")
        tickers = tickers[:cap]
    tickers = [t for t in tickers if t.replace(".", "").isalnum()]
    return tickers

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def fetch_data_batch(tickers: List[str], period: str = "1y") -> pd.DataFrame:
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
                threads=False,
                progress=False,
            )
            return df
        except Exception as e:
            last_err = e
    raise last_err

@st.cache_data(ttl=1800)
def get_fast_info_safe(ticker: str) -> Dict:
    try:
        fi = yf.Ticker(ticker).fast_info
        return {
            "last_price": getattr(fi, "last_price", None),
            "market_cap": getattr(fi, "market_cap", None),
            "trailing_pe": getattr(fi, "trailing_pe", None),
            "forward_pe": getattr(fi, "forward_pe", None),
            "dividend_yield": getattr(fi, "dividend_yield", None),
            "price_to_book": getattr(fi, "price_to_book", None),
        }
    except Exception:
        return {}

@st.cache_data(ttl=21600)  # 6 hours
def get_info_safe(ticker: str) -> Dict:
    delays = [0, 1.5]
    for d in delays:
        try:
            if d:
                time.sleep(d)
            info = yf.Ticker(ticker).get_info()
            if isinstance(info, dict):
                return info
        except Exception:
            pass
    return {}

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())
    out["Close"] = df["Close"]
    out["RSI14"] = rsi(df["Close"], 14)
    out["SMA50"] = df["Close"].rolling(50).mean()
    out["SMA200"] = df["Close"].rolling(200).mean()

    last_252 = df.tail(252)
    out.attrs["52w_high"] = float(last_252["High"].max()) if not last_252.empty else None
    out.attrs["52w_low"] = float(last_252["Low"].min()) if not last_252.empty else None

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

        st.line_chart(sig[["Close", "SMA50", "SMA200"]])
        st.line_chart(sig[["RSI14"]])
        st.caption("RSI guides: >70 overbought, <30 oversold.")
    except KeyError as e:
        st.error(f"{ticker}: {e}")
    except Exception as e:
        st.error(f"{ticker}: {e}")

def safe_num(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None

# ============================
# Factor scoring helpers
# ============================
def band(value, good, mid, reverse=False):
    """
    Return 0/50/100 score using thresholds.
    reverse=True => lower better (e.g., P/E). reverse=False => higher better (e.g., ROIC).
    For reverse=True: good=upper bound green, mid=upper bound yellow.
    For reverse=False: good=lower bound green, mid=lower bound yellow.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    try:
        v = float(value)
    except Exception:
        return 0

    if reverse:
        if v <= good:
            return 100
        elif v <= mid:
            return 50
        else:
            return 0
    else:
        if v >= good:
            return 100
        elif v >= mid:
            return 50
        else:
            return 0

def score_value(pe, pb, peg, de):
    s_pe  = band(pe, 15, 25, reverse=True)
    s_pb  = band(pb, 2, 4, reverse=True)
    s_peg = band(peg, 1, 2, reverse=True)
    s_de  = band(de, 0.5, 1, reverse=True)
    vals = [s for s in [s_pe, s_pb, s_peg, s_de] if s is not None]
    return round(sum(vals)/len(vals), 1) if vals else 0.0

def score_growth(roic):
    return float(band(roic, 0.15, 0.08, reverse=False))

def score_momentum(rsi_val, ret_1m, ret_3m):
    if rsi_val is None or pd.isna(rsi_val):
        s_rsi = 0
    else:
        r = float(rsi_val)
        if 55 <= r <= 65: s_rsi = 100
        elif 45 <= r < 55 or 65 < r <= 70: s_rsi = 50
        elif r < 30 or r > 70: s_rsi = 0
        else: s_rsi = 50
    s_1m = 100 if (ret_1m is not None and ret_1m > 0) else 0
    s_3m = 100 if (ret_3m is not None and ret_3m > 0) else 0
    return round((s_rsi + s_1m + s_3m) / 3, 1)

# Dark-theme friendly color utilities
def cell_css(bg, fg):
    return f"background-color:{bg};color:{fg}"

DARK_GREEN = "#1b5e20"
DARK_YELLOW = "#f9a825"  # black text readable
DARK_RED = "#b71c1c"
NEUTRAL = "#424242"      # only if needed

def color_band_val(v, green, yellow, reverse=False):
    try:
        x = float(v)
    except:
        return ""
    if reverse:  # lower better
        if x <= green: return cell_css(DARK_GREEN, "#ffffff")
        if x <= yellow: return cell_css(DARK_YELLOW, "#000000")
        return cell_css(DARK_RED, "#ffffff")
    else:        # higher better
        if x >= green: return cell_css(DARK_GREEN, "#ffffff")
        if x >= yellow: return cell_css(DARK_YELLOW, "#000000")
        return cell_css(DARK_RED, "#ffffff")

def color_band_rsi(v):
    try:
        r = float(v)
    except:
        return ""
    if 55 <= r <= 65: return cell_css(DARK_GREEN, "#ffffff")
    if 45 <= r < 55 or 65 < r <= 70: return cell_css(DARK_YELLOW, "#000000")
    if r < 30 or r > 70: return cell_css(DARK_RED, "#ffffff")
    return ""

# ============================
# Tabs
# ============================
tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Fundamentals (light)", "🗒️ Watchlist (Factors)"])

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
            rows.append({
                "Ticker": t,
                "Last Price": fi.get("last_price", "—"),
                "Mkt Cap": fi.get("market_cap", "—"),
                "Trailing PE": fi.get("trailing_pe", "—"),
                "Forward PE": fi.get("forward_pe", "—"),
                "Div Yield": fi.get("dividend_yield", "—"),
                "P/B": fi.get("price_to_book", "—"),
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

# ----------------------------
# Tab 3: Watchlist — factor view + FactorBlend
# ----------------------------
with tab3:
    st.subheader("🗒️ Watchlist — Factor View (Value • Growth • Momentum)")
    st.caption("Paste/upload tickers. We’ll compute RSI, 1M/3M returns, and valuation/quality factors with clear color bands.")

    colA, colB = st.columns(2)
    with colA:
        paste = st.text_area("Paste tickers (comma-separated):", "AAPL, MSFT, NVDA, META, AMZN")
    with colB:
        uploaded = st.file_uploader("Upload CSV (with 'Symbol' column)", type=["csv"])

    tickers_w = _clean_tickers(paste, cap=50) if paste else []
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "Symbol" in df_up.columns:
                extra = _clean_tickers(",".join(df_up["Symbol"].astype(str).tolist()), cap=100)
                tickers_w = list(dict.fromkeys(tickers_w + extra))[:50]
            else:
                st.warning("CSV missing 'Symbol' column.")
        except Exception as e:
            st.error(f"CSV error: {e}")

    # Weight presets & custom
    st.markdown("#### Factor Weights")
    preset = st.radio(
        "Choose weights",
        ["Balanced (V40 / G20 / M40)", "Value Tilt (V60 / G20 / M20)", "Momentum Tilt (V20 / G20 / M60)", "Custom"],
        horizontal=True,
    )

    if preset == "Balanced (V40 / G20 / M40)":
        wV, wG, wM = 40, 20, 40
    elif preset == "Value Tilt (V60 / G20 / M20)":
        wV, wG, wM = 60, 20, 20
    elif preset == "Momentum Tilt (V20 / G20 / M60)":
        wV, wG, wM = 20, 20, 60
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            wV = st.slider("Value %", 0, 100, 40)
        with c2:
            wG = st.slider("Growth %", 0, 100, 20)
        with c3:
            wM = st.slider("Momentum %", 0, 100, 40)
        total = wV + wG + wM
        if total == 0:
            st.warning("Weights sum to 0 — defaulting to Balanced.")
            wV, wG, wM = 40, 20, 40

    if st.button("Analyze Watchlist (Factors + Blend)"):
        if not tickers_w:
            st.warning("No tickers detected.")
        else:
            # 1) Batch price for RSI + returns
            try:
                price_df = fetch_data_batch(tickers_w, period="6mo")
            except Exception as e:
                st.error(f"Download error: {e}")
                price_df = pd.DataFrame()

            rows = []
            for t in tickers_w[:50]:
                # --- RSI + returns
                rsival = None
                ret_1m = None
                ret_3m = None
                try:
                    if not price_df.empty:
                        dfp = extract_single(price_df, t)
                        if not dfp.empty and len(dfp) > 64:
                            close = dfp["Close"]
                            rsival = rsi(close, 14).iloc[-1]
                            ret_1m = float((close.iloc[-1] / close.iloc[-22] - 1.0) * 100.0) if len(close) > 21 else None
                            ret_3m = float((close.iloc[-1] / close.iloc[-64] - 1.0) * 100.0) if len(close) > 63 else None
                except Exception:
                    pass

                # --- Fundamentals (heavier): get_info() cached
                info = get_info_safe(t)
                pe  = safe_num(info.get("trailingPE"))
                peg = safe_num(info.get("pegRatio"))
                pb  = safe_num(info.get("priceToBook"))
                de  = safe_num(info.get("debtToEquity"))
                roic = safe_num(info.get("returnOnInvestedCapital") or info.get("returnOnCapitalEmployed"))

                value_score    = score_value(pe, pb, peg, de)
                growth_score   = score_growth(roic)
                momentum_score = score_momentum(rsival, ret_1m, ret_3m)

                # FactorBlend (weights % → 0..1)
                w_sum = wV + wG + wM
                wVn, wGn, wMn = wV / w_sum, wG / w_sum, wM / w_sum
                factor_blend = round(value_score * wVn + growth_score * wGn + momentum_score * wMn, 1)

                rows.append({
                    "Ticker": t,
                    "RSI(14)": round(rsival, 2) if rsival is not None else "—",
                    "1M %": round(ret_1m, 2) if ret_1m is not None else "—",
                    "3M %": round(ret_3m, 2) if ret_3m is not None else "—",
                    "P/E": round(pe, 2) if pe is not None else "—",
                    "PEG": round(peg, 2) if peg is not None else "—",
                    "P/B": round(pb, 2) if pb is not None else "—",
                    "D/E": round(de, 2) if de is not None else "—",
                    "ROIC": round(roic, 3) if roic is not None else "—",
                    "ValueScore": value_score,
                    "GrowthScore": growth_score,
                    "MomentumScore": momentum_score,
                    "FactorBlend": factor_blend,
                })

            if not rows:
                st.info("No results computed.")
            else:
                df = pd.DataFrame(rows)

                sort_choice = st.radio(
                    "Sort by",
                    ["FactorBlend", "Lowest RSI", "Highest MomentumScore", "Highest ValueScore", "Highest GrowthScore"],
                    horizontal=True,
                )
                if sort_choice == "FactorBlend":
                    df = df.sort_values("FactorBlend", ascending=False)
                elif sort_choice == "Lowest RSI":
                    df["_sort"] = pd.to_numeric(df["RSI(14)"], errors="coerce")
                    df = df.sort_values("_sort", ascending=True).drop(columns=["_sort"])
                elif sort_choice == "Highest MomentumScore":
                    df = df.sort_values("MomentumScore", ascending=False)
                elif sort_choice == "Highest ValueScore":
                    df = df.sort_values("ValueScore", ascending=False)
                else:
                    df = df.sort_values("GrowthScore", ascending=False)

                # Dark-theme friendly color bands
                def styler(row: pd.Series):
                    styles = []
                    for col, val in row.items():
                        if col == "P/E":              styles.append(color_band_val(val, 15, 25, reverse=True))
                        elif col == "P/B":            styles.append(color_band_val(val, 2, 4, reverse=True))
                        elif col == "PEG":            styles.append(color_band_val(val, 1, 2, reverse=True))
                        elif col == "D/E":            styles.append(color_band_val(val, 0.5, 1, reverse=True))
                        elif col == "ROIC":           styles.append(color_band_val(val, 0.15, 0.08, reverse=False))
                        elif col == "RSI(14)":        styles.append(color_band_rsi(val))
                        elif col in ["1M %", "3M %"]:
                            try:
                                styles.append(cell_css(DARK_GREEN, "#ffffff") if float(val) > 0 else cell_css(DARK_RED, "#ffffff"))
                            except:
                                styles.append("")
                        elif col in ["ValueScore", "GrowthScore", "MomentumScore", "FactorBlend"]:
                            styles.append(color_band_val(val, 75, 50, reverse=False))
                        else:
                            styles.append("")
                    return styles

                styled = df.style.apply(styler, axis=1).format(precision=2)
                st.dataframe(styled, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Open top 5 by FactorBlend in Analysis"):
                        top = df.sort_values("FactorBlend", ascending=False)["Ticker"].head(5).tolist()
                        st.session_state["selected_tickers"] = ",".join(top)
                        st.success(f"Loaded {', '.join(top)} into Analysis tab.")
                with c2:
                    st.download_button(
                        "Download CSV",
                        data=df.to_csv(index=False),
                        file_name="watchlist_factors.csv",
                        mime="text/csv",
                    )

