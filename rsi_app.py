# rsi_app.py  — FMP-only version
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="RSI & Fundamentals (FMP)", layout="wide")
st.title("📈 Stocks: RSI, Fundamentals & Screeners (FMP)")

# Read API key from Streamlit Secrets (add it in Streamlit Cloud -> Settings -> Secrets)
FMP_API_KEY = st.secrets.get("FMP_API_KEY", "")
if not FMP_API_KEY:
    st.error("FMP_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

BASE = "https://financialmodelingprep.com/api/v3"

# ----------------------------
# Helpers
# ----------------------------
def _req(path: str, params: dict | None = None) -> dict | list | None:
    """GET request helper adding the API key and basic error handling."""
    if params is None:
        params = {}
    params["apikey"] = FMP_API_KEY
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(f"{BASE}{path}", params=params, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error for {path}: {e}")
        return None

@st.cache_data(ttl=1800)
def get_historical_df(ticker: str, days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV history (daily) and return a DataFrame indexed by date."""
    js = _req(f"/historical-price-full/{ticker}", {"timeseries": days})
    if not js or "historical" not in js:
        return pd.DataFrame()
    df = pd.DataFrame(js["historical"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI locally to avoid extra API calls."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def golden_cross(ma50: pd.Series, ma200: pd.Series) -> str:
    if len(ma50.dropna()) < 2 or len(ma200.dropna()) < 2:
        return "⚪ Not enough data"
    if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
        return "🟢 Golden Cross"
    if ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
        return "🔴 Death Cross"
    return "⚪ No recent cross"

@st.cache_data(ttl=1800)
def get_profile(ticker: str) -> dict:
    js = _req(f"/profile/{ticker}")
    if isinstance(js, list) and js:
        return js[0]
    return {}

@st.cache_data(ttl=1800)
def get_key_metrics(ticker: str) -> dict:
    js = _req(f"/key-metrics/{ticker}", {"period": "annual", "limit": 1})
    if isinstance(js, list) and js:
        return js[0]
    return {}

@st.cache_data(ttl=1800)
def get_ratios(ticker: str) -> dict:
    js = _req(f"/ratios/{ticker}", {"period": "annual", "limit": 1})
    if isinstance(js, list) and js:
        return js[0]
    return {}

@st.cache_data(ttl=1800)
def get_statement(ticker: str, which: str) -> pd.DataFrame:
    """which ∈ {'income-statement','balance-sheet-statement','cash-flow-statement'}"""
    js = _req(f"/{which}/{ticker}", {"period": "annual", "limit": 4})
    if not isinstance(js, list) or not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    # Ensure nice date order
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False).set_index("date")
    return df

@st.cache_data(ttl=600)
def get_screener(kind: str = "actives") -> pd.DataFrame:
    """
    kind: 'actives' | 'gainers' | 'losers'
    Returns DataFrame with columns: ticker, companyName, price, changesPercentage, change, volume
    """
    path = {
        "actives": "/stock/actives",
        "gainers": "/stock/gainers",
        "losers": "/stock/losers",
    }.get(kind, "/stock/actives")
    js = _req(path)
    # FMP returns a dict like {'mostActiveStock': [...]} or {'mostGainerStock': [...]} etc.
    key_map = {"actives": "mostActiveStock", "gainers": "mostGainerStock", "losers": "mostLoserStock"}
    block = (js or {}).get(key_map.get(kind, "mostActiveStock"), [])
    df = pd.DataFrame(block)
    # Normalize column names if present
    rename = {
        "ticker": "Ticker",
        "companyName": "Company",
        "price": "Price",
        "changesPercentage": "Change %",
        "change": "Change",
        "volume": "Volume",
    }
    df.rename(columns=rename, inplace=True)
    # Keep a clean subset if possible
    cols = [c for c in ["Ticker", "Company", "Price", "Change %", "Change", "Volume"] if c in df.columns]
    return df[cols] if not df.empty else df

def set_selected_tickers(tickers: list[str]):
    st.session_state["selected_tickers"] = ",".join(tickers)

def get_selected_tickers(default: str) -> str:
    return st.session_state.get("selected_tickers", default)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Fundamentals", "🧮 Screener"])

# ----------------------------
# Tab 3: Screener (first so clicks can drive other tabs)
# ----------------------------
with tab3:
    st.subheader("🧮 Market Movers (FMP)")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Most Active**")
        df_a = get_screener("actives")
        st.dataframe(df_a, use_container_width=True, height=340)
        if not df_a.empty:
            cols = st.columns(min(5, len(df_a)))
            for i, tkr in enumerate(df_a["Ticker"].head(10)):
                with cols[i % len(cols)]:
                    if st.button(f"Analyze {tkr}", key=f"act_{tkr}"):
                        set_selected_tickers([tkr])

    with colB:
        st.markdown("**Top Gainers**")
        df_g = get_screener("gainers")
        st.dataframe(df_g, use_container_width=True, height=340)
        if not df_g.empty:
            cols = st.columns(min(5, len(df_g)))
            for i, tkr in enumerate(df_g["Ticker"].head(10)):
                with cols[i % len(cols)]:
                    if st.button(f"Analyze {tkr}", key=f"gain_{tkr}"):
                        set_selected_tickers([tkr])

    with colC:
        st.markdown("**Top Losers**")
        df_l = get_screener("losers")
        st.dataframe(df_l, use_container_width=True, height=340)
        if not df_l.empty:
            cols = st.columns(min(5, len(df_l)))
            for i, tkr in enumerate(df_l["Ticker"].head(10)):
                with cols[i % len(cols)]:
                    if st.button(f"Analyze {tkr}", key=f"lose_{tkr}"):
                        set_selected_tickers([tkr])

    st.caption("Tip: Click any 'Analyze' button to load that ticker into the Analysis & Fundamentals tabs.")

# ----------------------------
# Tab 1: Analysis (RSI + MAs)
# ----------------------------
with tab1:
    st.subheader("🔍 Price • RSI(14) • SMA(50/200)")
    default_analysis = get_selected_tickers("AAPL, MSFT")
    user_in = st.text_input("Enter tickers (comma-separated):", default_analysis, key="analysis_input")
    tickers = [t.strip().upper() for t in user_in.split(",") if t.strip()]

    if st.button("Run Analysis"):
        for tkr in tickers:
            with st.container():
                prof = get_profile(tkr)
                company = prof.get("companyName") or prof.get("company_name") or prof.get("symbol") or tkr
                st.markdown(f"### {company} ({tkr})")

                df = get_historical_df(tkr, days=365)
                if df.empty:
                    st.warning(f"No historical data for {tkr}.")
                    continue

                df["SMA50"] = sma(df["Close"], 50)
                df["SMA200"] = sma(df["Close"], 200)
                df["RSI14"] = rsi(df["Close"], 14)

                # Candlestick chart with SMA overlays
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="Price"
                ))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50", mode="lines"))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200", mode="lines"))
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # RSI chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI(14)", mode="lines"))
                fig_rsi.add_hline(y=70, line_dash="dot")
                fig_rsi.add_hline(y=30, line_dash="dot")
                fig_rsi.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_rsi, use_container_width=True)

                # Signals & key numbers
                col1, col2, col3, col4 = st.columns(4)
                latest_close = df["Close"].iloc[-1]
                latest_rsi = df["RSI14"].iloc[-1]
                col1.metric("Last Close", f"${latest_close:,.2f}")
                col2.metric("RSI(14)", f"{latest_rsi:,.2f}")
                signal = golden_cross(df["SMA50"], df["SMA200"])
                col3.metric("MA Signal", signal)
                col4.metric("52w High / Low",
                            f"${df['High'].tail(252).max():,.2f} / ${df['Low'].tail(252).min():,.2f}")

# ----------------------------
# Tab 2: Fundamentals (FMP)
# ----------------------------
with tab2:
    st.subheader("📊 Fundamentals (Profile • Key Metrics • Ratios • Statements)")
    default_fund = get_selected_tickers("AAPL, MSFT")
    fund_in = st.text_input("Enter tickers (comma-separated):", default_fund, key="fund_input")
    tickers_f = [t.strip().upper() for t in fund_in.split(",") if t.strip()]

    if st.button("Fetch Fundamentals"):
        for tkr in tickers_f:
            st.markdown(f"### {tkr}")
            prof = get_profile(tkr)
            if not prof:
                st.warning(f"No profile found for {tkr}.")
                continue

            name = prof.get("companyName") or prof.get("company_name") or tkr
            sector = prof.get("sector", "—")
            industry = prof.get("industry", "—")
            website = prof.get("website", "—")
            desc = prof.get("description", "")

            colA, colB, colC = st.columns([2, 1, 1])
            with colA:
                st.markdown(f"**{name}**  \n*{sector} • {industry}*")
                if website and website != "—":
                    st.write(website)
                if desc:
                    st.caption(desc[:400] + ("…" if len(desc) > 400 else ""))

            # Key metrics & ratios
            km = get_key_metrics(tkr) or {}
            rt = get_ratios(tkr) or {}

            def fmt(v):
                if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
                    return "—"
                if isinstance(v, (int, float)):
                    return f"{v:,.2f}"
                return str(v)

            with colB:
                st.markdown("**Key Metrics**")
                km_map = {
                    "Market Cap": km.get("marketCap"),
                    "Enterprise Value": km.get("enterpriseValue"),
                    "PE Ratio": km.get("peRatio"),
                    "Forward PE": km.get("forwardPE"),
                    "EV/EBITDA": km.get("evToEbitda"),
                    "Dividend Yield": km.get("dividendYield"),
                }
                st.table(pd.DataFrame({"Value": [fmt(v) for v in km_map.values()]}, index=km_map.keys()))

            with colC:
                st.markdown("**Ratios**")
                rt_map = {
                    "Profit Margin": rt.get("netProfitMargin"),
                    "ROE": rt.get("returnOnEquity"),
                    "ROA": rt.get("returnOnAssets"),
                    "Current Ratio": rt.get("currentRatio"),
                    "Debt/Equity": rt.get("debtEquityRatio"),
                }
                st.table(pd.DataFrame({"Value": [fmt(v) for v in rt_map.values()]}, index=rt_map.keys()))

            # Financial statements (last 4 FY)
            inc = get_statement(tkr, "income-statement")
            bal = get_statement(tkr, "balance-sheet-statement")
            cfs = get_statement(tkr, "cash-flow-statement")

            st.markdown("#### Financial Statements (Annual)")
            st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            t1, t2, t3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            with t1:
                if not inc.empty:
                    show_cols = [c for c in ["revenue", "grossProfit", "operatingIncome", "netIncome", "eps"] if c in inc.columns]
                    st.dataframe(inc[show_cols].head(4))
                else:
                    st.caption("No income statement data.")
            with t2:
                if not bal.empty:
                    show_cols = [c for c in ["totalAssets", "totalLiabilities", "totalStockholdersEquity", "cashAndCashEquivalents"] if c in bal.columns]
                    st.dataframe(bal[show_cols].head(4))
                else:
                    st.caption("No balance sheet data.")
            with t3:
                if not cfs.empty:
                    show_cols = [c for c in ["netCashProvidedByOperatingActivities", "capitalExpenditure", "freeCashFlow"] if c in cfs.columns]
                    st.dataframe(cfs[show_cols].head(4))
                else:
                    st.caption("No cash flow data.")

# Footer tip
st.caption("Data source: Financial Modeling Prep (FMP). Prices, indicators and fundamentals retrieved via FMP API.")

