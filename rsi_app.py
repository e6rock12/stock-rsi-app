# rsi_app.py — RSI • MAs • Factor Watchlist with FactorBlend, dark-theme bands, Tips/Guides, and My Portfolio + Sector Breakdown
import time
from typing import List, Dict

import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="RSI & Factor Watchlist", layout="wide")
st.title("📈 RSI • Moving Averages • Factor Watchlist (Value • Growth • Momentum)")

# ============================
# Tips / Guides content
# ============================
TIPS = {
    "rsi": "RSI(14): 0–100 momentum oscillator. >70 often overbought; <30 oversold.",
    "sma": "Moving Average (SMA): 50/200-day trend lines; longer MAs smooth more.",
    "cross": "Golden Cross: 50-day SMA crosses above 200-day (bullish). Death Cross: opposite (bearish).",
    "52w": "52-week High/Low: highest and lowest close over ~1 year (~252 trading days).",
    "pe": "P/E: Price ÷ trailing 12m earnings per share. Lower can indicate better value (but context matters).",
    "peg": "PEG: P/E ÷ earnings growth. ~1 is ‘fair’; <1 may indicate good value for growth.",
    "pb": "P/B: Price ÷ book value per share. Lower can indicate value; sector norms differ.",
    "de": "D/E: Total debt ÷ shareholder equity. Lower generally means less leverage risk.",
    "roic": "ROIC: Return on invested capital. >15% often indicates high-quality business.",
    "ret": "Returns: 1M/3M % price change (approx. 21/63 trading days).",
    "valuescore": "ValueScore: Composite of P/E, P/B, PEG, D/E (lower is better).",
    "growthscore": "GrowthScore: Based on ROIC (higher is better).",
    "momentumscore": "MomentumScore: Combines RSI ‘sweet spot’ and positive 1M/3M returns.",
    "blend": "FactorBlend: Weighted average of Value, Growth, Momentum (your chosen weights).",
    "fundamentals": "Quick Fundamentals: uses fast_info first; fills missing fields with get_info().",
}

def tip(text_key: str):
    if st.session_state.get("show_inline_tips", True):
        msg = TIPS.get(text_key, "")
        if msg:
            st.caption(f"ℹ️ {msg}")

# Sidebar controls
with st.sidebar:
    st.markdown("### Settings")
    st.session_state["show_inline_tips"] = st.toggle(
        "Show inline tips", value=True, help="Show one-line tips under metrics/tables."
    )
    with st.expander("📘 Guides & Definitions", expanded=False):
        st.markdown(
            """
**RSI(14):** Momentum oscillator (0–100). >70 = overbought; <30 = oversold.  
**SMA 50/200:** 50/200-day trend lines. Crossovers can signal trend shifts.  
**Golden/Death Cross:** 50 > 200 = bullish; 50 < 200 = bearish.  
**52-week High/Low:** Highest/lowest close in ~1 year (~252 trading days).  
**P/E:** Price ÷ trailing earnings per share. Lower can imply value.  
**PEG:** P/E ÷ earnings growth. ≈1 fair; <1 often attractive.  
**P/B:** Price ÷ book value per share. Lower may imply value (sector-specific).  
**D/E:** Debt ÷ equity. Lower often safer.  
**ROIC:** Profitability on all invested capital; >15% often high quality.  
**1M/3M returns:** Approximate price momentum (21/63 trading days).  
**ValueScore:** Composite of P/E, P/B, PEG, D/E (lower better).  
**GrowthScore:** Based on ROIC (higher better).  
**MomentumScore:** RSI ‘sweet spot’ (55–65) + positive 1M/3M returns.  
**FactorBlend:** Weighted average of Value/Growth/Momentum (choose weights).
            """
        )

# ============================
# Core Helpers
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
        with c2: tip("rsi")
        with c3: tip("52w")
        with c4: tip("cross")

        st.line_chart(sig[["Close", "SMA50", "SMA200"]])
        tip("sma")
        st.line_chart(sig[["RSI14"]])
        tip("rsi")
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
DARK_YELLOW = "#f9a825"
DARK_RED = "#b71c1c"

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
# Fundamentals fallbacks & formatting
# ============================
@st.cache_data(ttl=21600)  # 6h
def fundamental_snapshot(ticker: str) -> dict:
    fi = get_fast_info_safe(ticker) or {}
    pe = fi.get("trailing_pe")
    fpe = fi.get("forward_pe")
    div = fi.get("dividend_yield")
    pb  = fi.get("price_to_book")
    last_px = fi.get("last_price")
    mcap = fi.get("market_cap")

    info = {}
    if any(v is None for v in [pe, fpe, div, pb, last_px, mcap]):
        info = get_info_safe(ticker) or {}
        if pe    is None: pe    = info.get("trailingPE")
        if fpe   is None: fpe   = info.get("forwardPE")
        if div   is None: div   = info.get("dividendYield")
        if pb    is None: pb    = info.get("priceToBook")
        if last_px is None: last_px = info.get("currentPrice")
        if mcap  is None: mcap  = info.get("marketCap")

    return {
        "Ticker": ticker,
        "Last Price": last_px,
        "Mkt Cap": mcap,
        "Trailing PE": pe,
        "Forward PE": fpe,
        "Div Yield": div,  # fraction if from info
        "P/B": pb,
    }

def _fmt_num(x, digs=2, comma=True):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        x = float(x)
        return f"{x:,.{digs}f}" if comma else f"{x:.{digs}f}"
    except Exception:
        return "—"

def _fmt_int(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "—"

def _fmt_pct(x, digs=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        return f"{float(x)*100:.{digs}f}%"
    except Exception:
        return "—"

# ============================
# Tabs (4 tabs now)
# ============================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analysis",
    "📊 Fundamentals (light)",
    "🗒️ Watchlist (Factors)",
    "💼 My Portfolio"
])

# ----------------------------
# Tab 1: Analysis
# ----------------------------
with tab1:
    st.subheader("🔍 RSI(14), SMA(50/200), 52-Week Levels")
    tip("rsi"); tip("sma"); tip("52w")
    default = st.session_state.get("selected_tickers", "AAPL, MSFT, TSLA")
    t_input = st.text_input("Enter tickers (comma-separated):", default, help="Enter up to ~10 symbols to minimize rate limits.")
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
    st.subheader("📊 Quick Fundamentals (cached & fallback to get_info for missing fields)")
    tip("fundamentals"); tip("pe"); tip("pb")

    t_input_f = st.text_input("Tickers (comma-separated):", "AAPL, MSFT", key="funds_input", help="Try a few at a time for speed.")
    tickers_f = _clean_tickers(t_input_f, cap=8)

    if st.button("Get Fundamentals"):
        if not tickers_f:
            st.warning("Please enter at least one ticker.")
        else:
            snaps = [fundamental_snapshot(t) for t in tickers_f]
            df = pd.DataFrame(snaps, columns=["Ticker","Last Price","Mkt Cap","Trailing PE","Forward PE","Div Yield","P/B"])

            df["Last Price"] = df["Last Price"].apply(lambda v: _fmt_num(v, 2))
            df["Mkt Cap"]    = df["Mkt Cap"].apply(_fmt_int)
            df["Trailing PE"]= df["Trailing PE"].apply(lambda v: _fmt_num(v, 2, comma=False))
            df["Forward PE"] = df["Forward PE"].apply(lambda v: _fmt_num(v, 2, comma=False))
            df["P/B"]        = df["P/B"].apply(lambda v: _fmt_num(v, 2, comma=False))
            df["Div Yield"]  = df["Div Yield"].apply(lambda v: _fmt_pct(v, 2))

            st.dataframe(df, use_container_width=True)

# ----------------------------
# Tab 3: Watchlist — factor view + FactorBlend
# ----------------------------
with tab3:
    st.subheader("🗒️ Watchlist — Factor View (Value • Growth • Momentum)")
    tip("valuescore"); tip("growthscore"); tip("momentumscore"); tip("blend"); tip("ret"); tip("pe"); tip("peg"); tip("pb"); tip("de"); tip("roic")

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
            wV = st.slider("Value %", 0, 100, 40, help="Weight on ValueScore (P/E, P/B, PEG, D/E).")
        with c2:
            wG = st.slider("Growth %", 0, 100, 20, help="Weight on GrowthScore (ROIC).")
        with c3:
            wM = st.slider("Momentum %", 0, 100, 40, help="Weight on MomentumScore (RSI band + returns).")
        total = wV + wG + wM
        if total == 0:
            st.warning("Weights sum to 0 — defaulting to Balanced.")
            wV, wG, wM = 40, 20, 40

    if st.button("Analyze Watchlist (Factors + Blend)"):
        if not tickers_w:
            st.warning("No tickers detected.")
        else:
            try:
                price_df = fetch_data_batch(tickers_w, period="6mo")
            except Exception as e:
                st.error(f"Download error: {e}")
                price_df = pd.DataFrame()

            rows = []
            for t in tickers_w[:50]:
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

                info = get_info_safe(t)
                pe  = safe_num(info.get("trailingPE"))
                peg = safe_num(info.get("pegRatio"))
                pb  = safe_num(info.get("priceToBook"))
                de  = safe_num(info.get("debtToEquity"))
                roic = safe_num(info.get("returnOnInvestedCapital") or info.get("returnOnCapitalEmployed"))

                value_score    = score_value(pe, pb, peg, de)
                growth_score   = score_growth(roic)
                momentum_score = score_momentum(rsival, ret_1m, ret_3m)

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

# ----------------------------
# Tab 4: My Portfolio (with Sector Breakdown)
# ----------------------------
with tab4:
    st.subheader("💼 My Portfolio — Overview, Signals & Sector Breakdown")

    tmpl = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "NVDA"], "Shares": [10, 5, 3], "CostBasis": [150, 300, 400]})
    st.download_button(
        "Download CSV template",
        data=tmpl.to_csv(index=False),
        file_name="portfolio_template.csv",
        mime="text/csv",
        help="Columns: Symbol, Shares, CostBasis (per share)."
    )

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Paste positions (one per line):** `SYMBOL, shares, cost_basis`")
        pasted = st.text_area(
            "Example:\nAAPL, 10, 150\nMSFT, 5, 300",
            value="AAPL, 10, 150\nMSFT, 5, 300",
            height=120
        )
    with colR:
        uploaded_pf = st.file_uploader("…or upload CSV (Symbol, Shares, CostBasis)", type=["csv"])

    rows_in = []
    if pasted.strip():
        for line in pasted.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue
            sym = parts[0].upper()
            sh = float(parts[1]) if len(parts) > 1 and parts[1] != "" else 0.0
            cb = float(parts[2]) if len(parts) > 2 and parts[2] != "" else None
            if sym.replace(".", "").isalnum():
                rows_in.append({"Symbol": sym, "Shares": sh, "CostBasis": cb})

    if uploaded_pf is not None:
        try:
            df_up = pd.read_csv(uploaded_pf)
            for _, r in df_up.iterrows():
                sym = str(r.get("Symbol", "")).upper().strip()
                if not sym or not sym.replace(".", "").isalnum():
                    continue
                sh = float(r.get("Shares", 0) or 0.0)
                cb = r.get("CostBasis", None)
                cb = float(cb) if cb is not None and cb != "" else None
                rows_in.append({"Symbol": sym, "Shares": sh, "CostBasis": cb})
        except Exception as e:
            st.error(f"CSV parse error: {e}")

    if rows_in:
        dedup = {}
        for r in rows_in:
            dedup[r["Symbol"]] = r
        positions = list(dedup.values())
    else:
        positions = []

    st.markdown("#### Factor Weights")
    preset_pf = st.radio(
        "Choose weights",
        ["Balanced (V40 / G20 / M40)", "Value Tilt (V60 / G20 / M20)", "Momentum Tilt (V20 / G20 / M60)", "Custom"],
        horizontal=True,
        key="pf_weights_choice"
    )
    if preset_pf == "Balanced (V40 / G20 / M40)":
        wV_pf, wG_pf, wM_pf = 40, 20, 40
    elif preset_pf == "Value Tilt (V60 / G20 / M20)":
        wV_pf, wG_pf, wM_pf = 60, 20, 20
    elif preset_pf == "Momentum Tilt (V20 / G20 / M60)":
        wV_pf, wG_pf, wM_pf = 20, 20, 60
    else:
        c1, c2, c3 = st.columns(3)
        with c1: wV_pf = st.slider("Value %", 0, 100, 40, key="pf_wV")
        with c2: wG_pf = st.slider("Growth %", 0, 100, 20, key="pf_wG")
        with c3: wM_pf = st.slider("Momentum %", 0, 100, 40, key="pf_wM")
        if (wV_pf + wG_pf + wM_pf) == 0:
            st.warning("Weights sum to 0 — defaulting to Balanced.")
            wV_pf, wG_pf, wM_pf = 40, 20, 40

    if st.button("Analyze Portfolio", type="primary"):
        if not positions:
            st.warning("Add positions first (paste or upload).")
        else:
            symbols = [p["Symbol"] for p in positions]
            try:
                price_df = fetch_data_batch(symbols, period="6mo")
            except Exception as e:
                st.error(f"Download error: {e}")
                price_df = pd.DataFrame()

            pos_rows = []
            total_value = 0.0
            total_cost = 0.0

            # For sector breakdown
            sector_values = {}

            for p in positions:
                tkr = p["Symbol"]
                shares = float(p.get("Shares", 0) or 0.0)
                cost_basis = p.get("CostBasis", None)
                cost_basis = float(cost_basis) if cost_basis is not None else None

                snap = fundamental_snapshot(tkr)
                last_px = safe_num(snap.get("Last Price"))
                if last_px is None:
                    inf = get_info_safe(tkr)
                    last_px = safe_num(inf.get("currentPrice"))

                # retrieve sector (cached via get_info_safe)
                info_sector = get_info_safe(tkr)
                sector = info_sector.get("sector") or "Unknown"

                rsival = None
                ret_1m = None
                ret_3m = None
                try:
                    if not price_df.empty:
                        dfp = extract_single(price_df, tkr)
                        if not dfp.empty and len(dfp) > 64:
                            close = dfp["Close"]
                            rsival = rsi(close, 14).iloc[-1]
                            ret_1m = float((close.iloc[-1] / close.iloc[-22] - 1.0) * 100.0) if len(close) > 21 else None
                            ret_3m = float((close.iloc[-1] / close.iloc[-64] - 1.0) * 100.0) if len(close) > 63 else None
                except Exception:
                    pass

                pe  = safe_num(info_sector.get("trailingPE"))
                peg = safe_num(info_sector.get("pegRatio"))
                pb  = safe_num(info_sector.get("priceToBook"))
                de  = safe_num(info_sector.get("debtToEquity"))
                roic = safe_num(info_sector.get("returnOnInvestedCapital") or info_sector.get("returnOnCapitalEmployed"))

                val_score  = score_value(pe, pb, peg, de)
                grw_score  = score_growth(roic)
                mom_score  = score_momentum(rsival, ret_1m, ret_3m)

                wsum = wV_pf + wG_pf + wM_pf
                wVn, wGn, wMn = wV_pf/wsum, wG_pf/wsum, wM_pf/wsum
                blend = round(val_score*wVn + grw_score*wGn + mom_score*wMn, 1)

                value = last_px * shares if (last_px is not None and shares) else None
                cost  = (cost_basis * shares) if (cost_basis is not None and shares) else None
                pl    = (value - cost) if (value is not None and cost is not None) else None
                plpct = ((value/cost - 1.0)*100.0) if (value is not None and cost and cost != 0) else None

                if value is not None:
                    total_value += value
                    sector_values[sector] = sector_values.get(sector, 0.0) + value
                if cost  is not None:
                    total_cost  += cost

                pos_rows.append({
                    "Ticker": tkr,
                    "Sector": sector,
                    "Shares": shares if shares else "—",
                    "Price": round(last_px, 2) if last_px is not None else "—",
                    "Value": round(value, 2) if value is not None else "—",
                    "CostBasis": round(cost_basis, 2) if cost_basis is not None else "—",
                    "P/L %": round(plpct, 2) if plpct is not None else "—",
                    "RSI(14)": round(rsival, 2) if rsival is not None else "—",
                    "1M %": round(ret_1m, 2) if ret_1m is not None else "—",
                    "3M %": round(ret_3m, 2) if ret_3m is not None else "—",
                    "P/E": round(pe, 2) if pe is not None else "—",
                    "PEG": round(peg, 2) if peg is not None else "—",
                    "P/B": round(pb, 2) if pb is not None else "—",
                    "D/E": round(de, 2) if de is not None else "—",
                    "ROIC": round(roic, 3) if roic is not None else "—",
                    "ValueScore": val_score,
                    "GrowthScore": grw_score,
                    "MomentumScore": mom_score,
                    "FactorBlend": blend,
                })

            col1, col2, col3, col4 = st.columns(4)
            total_pl = (total_value - total_cost) if (total_value and total_cost) else None
            total_plpct = ((total_value/total_cost - 1.0)*100.0) if (total_value and total_cost and total_cost != 0) else None
            col1.metric("Total Value", f"${total_value:,.2f}" if total_value else "—")
            col2.metric("Total Cost", f"${total_cost:,.2f}" if total_cost else "—")
            col3.metric("Total P/L %", f"{total_plpct:.2f}%" if total_plpct is not None else "—")
            col4.metric("Total P/L $", f"${total_pl:,.2f}" if total_pl is not None else "—")

            dfp = pd.DataFrame(pos_rows)

            # Weighted averages (by Value) for RSI & Blend if value available
            try:
                dfp["_valnum"] = pd.to_numeric(dfp["Value"], errors="coerce")
                dfp["_rsi"]    = pd.to_numeric(dfp["RSI(14)"], errors="coerce")
                dfp["_blend"]  = pd.to_numeric(dfp["FactorBlend"], errors="coerce")
                if dfp["_valnum"].sum() > 0:
                    w_avg_rsi = (dfp["_rsi"] * dfp["_valnum"]).sum() / dfp["_valnum"].sum()
                    w_avg_blend = (dfp["_blend"] * dfp["_valnum"]).sum() / dfp["_valnum"].sum()
                else:
                    w_avg_rsi, w_avg_blend = None, None
            except Exception:
                w_avg_rsi, w_avg_blend = None, None

            cA, cB = st.columns(2)
            cA.metric("Weighted Avg RSI", f"{w_avg_rsi:.2f}" if w_avg_rsi is not None else "—")
            cB.metric("Weighted Avg FactorBlend", f"{w_avg_blend:.2f}" if w_avg_blend is not None else "—")

            # Sort selector
            sort_pf = st.radio(
                "Sort positions by",
                ["Largest Value", "Lowest RSI", "Highest FactorBlend", "Highest P/L %"],
                horizontal=True
            )
            if sort_pf == "Largest Value":
                dfp["_sort"] = pd.to_numeric(dfp["Value"], errors="coerce")
                dfp = dfp.sort_values("_sort", ascending=False).drop(columns=["_sort"])
            elif sort_pf == "Lowest RSI":
                dfp["_sort"] = pd.to_numeric(dfp["RSI(14)"], errors="coerce")
                dfp = dfp.sort_values("_sort", ascending=True).drop(columns=["_sort"])
            elif sort_pf == "Highest FactorBlend":
                dfp = dfp.sort_values("FactorBlend", ascending=False)
            else:
                dfp["_sort"] = pd.to_numeric(dfp["P/L %"], errors="coerce")
                dfp = dfp.sort_values("_sort", ascending=False).drop(columns=["_sort"])

            def styler(row: pd.Series):
                styles = []
                for col, val in row.items():
                    if col == "P/E":              styles.append(color_band_val(val, 15, 25, reverse=True))
                    elif col == "P/B":            styles.append(color_band_val(val, 2, 4, reverse=True))
                    elif col == "PEG":            styles.append(color_band_val(val, 1, 2, reverse=True))
                    elif col == "D/E":            styles.append(color_band_val(val, 0.5, 1, reverse=True))
                    elif col == "ROIC":           styles.append(color_band_val(val, 0.15, 0.08, reverse=False))
                    elif col == "RSI(14)":        styles.append(color_band_rsi(val))
                    elif col in ["1M %", "3M %", "P/L %"]:
                        try:
                            styles.append(cell_css("#1b5e20", "#ffffff") if float(val) > 0 else cell_css("#b71c1c", "#ffffff"))
                        except:
                            styles.append("")
                    elif col in ["ValueScore", "GrowthScore", "MomentumScore", "FactorBlend"]:
                        styles.append(color_band_val(val, 75, 50, reverse=False))
                    else:
                        styles.append("")
                return styles

            st.dataframe(dfp.style.apply(styler, axis=1).format(precision=2), use_container_width=True)

            # Sector breakdown (by current Value)
            st.markdown("### 🧩 Sector Breakdown")
            if total_value > 0 and sector_values:
                sector_df = pd.DataFrame([
                    {"Sector": k, "Value": v, "Weight %": (v / total_value) * 100.0}
                    for k, v in sorted(sector_values.items(), key=lambda kv: kv[1], reverse=True)
                ])
                st.dataframe(sector_df.style.format({"Value": "{:,.2f}", "Weight %": "{:.2f}%"}),
                             use_container_width=True)
                # Simple bar chart of weights
                chart_df = sector_df.set_index("Sector")["Weight %"]
                st.bar_chart(chart_df)
            else:
                st.info("No sector data available yet (check that positions fetched current prices).")

            # Convenience: send top FactorBlend to Analysis
            if st.button("Open top 5 FactorBlend in Analysis"):
                top_syms = dfp.sort_values("FactorBlend", ascending=False)["Ticker"].head(5).tolist()
                st.session_state["selected_tickers"] = ",".join(top_syms)
                st.success(f"Loaded {', '.join(top_syms)} into Analysis tab.")

