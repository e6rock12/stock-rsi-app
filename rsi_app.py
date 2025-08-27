import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from yahooquery import Screener

# ---------- Helpers ----------
def _extract_series(df: pd.DataFrame, col: str, ticker: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    else:
        st.warning(f"{col} not found for {ticker}")
        return pd.Series()

# ---------- Fetch Data ----------
@st.cache_data(ttl=86400)
def get_us_tickers():
    """Fetch most active US tickers (up to 1000)"""
    try:
        s = Screener()
        screen = s.get_screeners('most_actives', count=1000)
        tickers = [item['symbol'] for item in screen['most_actives']['quotes']]
        return tickers
    except Exception as e:
        st.error(f"Error fetching US tickers: {e}")
        return []

def get_company_name(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("shortName", ticker)
    except:
        return ticker

# ---------- RSI Calculation ----------
def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ---------- Moving Averages ----------
def calculate_ma(data: pd.DataFrame, window: int):
    return data["Close"].rolling(window=window).mean()

# ---------- Fundamentals ----------
def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Forward PE": info.get("forwardPE"),
        "EPS Growth (next 5Y)": info.get("earningsQuarterlyGrowth"),
        "Return on Capital": info.get("returnOnEquity"),
    }

# ---------- UI ----------
st.title("📈 Stock RSI & Fundamentals Dashboard")

user_input = st.text_input("Enter stock ticker or company name:", "AAPL").upper()

# try to resolve input
tickers = get_us_tickers()
ticker = None
if user_input in tickers:
    ticker = user_input
else:
    # try to find by name
    matches = [t for t in tickers if user_input.lower() in get_company_name(t).lower()]
    if matches:
        ticker = matches[0]

if ticker:
    company_name = get_company_name(ticker)
    st.subheader(f"{ticker} - {company_name}")

    # Download 1y of data
    data = yf.download(ticker, period="1y")

    # Indicators
    data["RSI"] = calculate_rsi(data)
    data["MA50"] = calculate_ma(data, 50)
    data["MA200"] = calculate_ma(data, 200)

    latest_price = data["Close"].iloc[-1]
    rsi_latest = data["RSI"].iloc[-1]
    ma50_latest = data["MA50"].iloc[-1]
    ma200_latest = data["MA200"].iloc[-1]

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart", "💰 Fundamentals", "🔎 Analysis", "🧮 Screener"])

    # ---- Chart ----
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data["Close"], label="Close Price", alpha=0.8)
        ax.plot(data.index, data["MA50"], label="50-day MA")
        ax.plot(data.index, data["MA200"], label="200-day MA")

        ax.set_title(f"{ticker} Price & Moving Averages")
        ax.legend()
        st.pyplot(fig)

    # ---- Fundamentals ----
    with tab2:
        fundamentals = get_fundamentals(ticker)
        for metric, value in fundamentals.items():
            if value is None:
                st.write(f"{metric}: N/A")
            else:
                if metric == "Forward PE":
                    color = "green" if value < 20 else "red"
                elif metric == "EPS Growth (next 5Y)":
                    color = "green" if value > 0 else "red"
                elif metric == "Return on Capital":
                    color = "green" if value > 0 else "red"
                else:
                    color = "white"
                st.markdown(f"**{metric}:** <span style='color:{color}'>{value:.2f}</span>", unsafe_allow_html=True)

    # ---- Analysis ----
    with tab3:
        st.metric("Current Price", f"${latest_price:.2f}")
        st.metric("RSI (14)", f"{rsi_latest:.2f}")
        st.metric("50-day MA", f"${ma50_latest:.2f}")
        st.metric("200-day MA", f"${ma200_latest:.2f}")

        if ma50_latest > ma200_latest:
            st.success("Golden Cross: 50-day MA is above 200-day MA ✅")
        else:
            st.warning("No Golden Cross: 50-day MA is below 200-day MA ❌")

    # ---- Screener ----
    with tab4:
        st.subheader("📋 Stock Screener (Most Active US Stocks)")

        min_pe = st.number_input("Max Forward PE", value=20.0)
        min_eps = st.number_input("Min EPS Growth", value=0.0)
        min_roc = st.number_input("Min Return on Capital", value=0.0)
        max_rsi = st.number_input("Max RSI", value=70.0)

        results = []
        for t in tickers[:100]:  # limit to top 100 for speed
            try:
                f = get_fundamentals(t)
                d = yf.download(t, period="6mo")
                rsi_val = calculate_rsi(d).iloc[-1]

                if (
                    f["Forward PE"] is not None and f["Forward PE"] <= min_pe
                    and f["EPS Growth (next 5Y)"] is not None and f["EPS Growth (next 5Y)"] >= min_eps
                    and f["Return on Capital"] is not None and f["Return on Capital"] >= min_roc
                    and rsi_val <= max_rsi
                ):
                    results.append([t, get_company_name(t), f["Forward PE"], f["EPS Growth (next 5Y)"], f["Return on Capital"], rsi_val])
            except:
                continue

        if results:
            df_results = pd.DataFrame(results, columns=["Ticker", "Company", "Forward PE", "EPS Growth", "ROC", "RSI"])
            st.dataframe(df_results)
        else:
            st.info("No stocks met the criteria.")
else:
    st.error("Ticker or company not found in most active stocks.")

