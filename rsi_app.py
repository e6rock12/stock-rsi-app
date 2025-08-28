import streamlit as st
import yfinance as yf
import pandas as pd
import requests

st.set_page_config(page_title="RSI & Stock Analyzer", layout="wide")

# ---------------- Helper Functions ---------------- #

def resolve_ticker(user_input: str):
    """Resolve a company name or ticker symbol to a valid ticker."""
    user_input = user_input.strip().upper()

    # If it already looks like a ticker (short and letters only), return directly
    if len(user_input) <= 5 and user_input.isalpha():
        return user_input

    # Otherwise, try searching Yahoo Finance
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={user_input}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        results = r.json().get("quotes", [])
        if results:
            return results[0]["symbol"]
    except Exception as e:
        st.error(f"Lookup failed for {user_input}: {e}")

    return None


def compute_rsi(data, window=14):
    """Compute RSI from closing price data."""
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def golden_cross(ma50, ma200):
    """Detect Golden Cross or Death Cross signals."""
    try:
        if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
            return "🟢 Golden Cross (Bullish)"
        elif ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
            return "🔴 Death Cross (Bearish)"
        else:
            return "⚪ No Cross"
    except Exception:
        return "⚪ Not enough data"


def fetch_stock_data(ticker):
    """Fetch stock price history and fundamentals."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        return stock, hist
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None, None


def color_value(val, good, bad):
    """Color fundamentals: green if good, red if bad."""
    try:
        val = float(val)
        if val >= good:
            return f"🟢 {val}"
        elif val <= bad:
            return f"🔴 {val}"
        else:
            return f"⚪ {val}"
    except:
        return f"⚪ {val}"


# ---------------- Streamlit UI ---------------- #

st.title("📈 RSI & Stock Analyzer with Fundamentals")

tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Screener"])

# --------------- Stock Analysis Tab --------------- #
with tab1:
    st.header("Stock Analysis (RSI & Moving Averages)")
    user_input = st.text_input("Enter stock ticker(s) or company name(s) (comma separated):", "AAPL, MSFT")

    if user_input:
        tickers = [t.strip() for t in user_input.split(",") if t.strip()]

        for entry in tickers:
            ticker = resolve_ticker(entry)
            if not ticker:
                st.error(f"Could not resolve '{entry}' to a valid ticker.")
                continue

            stock, hist = fetch_stock_data(ticker)
            if hist is None or hist.empty:
                st.error(f"No data found for {ticker}")
                continue

            hist["RSI"] = compute_rsi(hist)
            hist["MA50"] = hist["Close"].rolling(window=50).mean()
            hist["MA200"] = hist["Close"].rolling(window=200).mean()

            st.subheader(f"📊 {ticker}")
            st.line_chart(hist[["Close", "MA50", "MA200"]])
            st.line_chart(hist["RSI"])

            signal = golden_cross(hist["MA50"], hist["MA200"])
            st.write(f"**Crossover Signal:** {signal}")


# --------------- Fundamentals Tab --------------- #
with tab2:
    st.header("Company Fundamentals")
    fundamentals_input = st.text_input("Enter stock ticker(s) or company name(s):", "AAPL")

    if fundamentals_input:
        tickers = [t.strip() for t in fundamentals_input.split(",") if t.strip()]

        for entry in tickers:
            ticker = resolve_ticker(entry)
            if not ticker:
                st.error(f"Could not resolve '{entry}' to a valid ticker.")
                continue

            stock, _ = fetch_stock_data(ticker)
            if not stock:
                continue

            info = stock.info
            st.subheader(f"🏢 {ticker}")

            fundamentals = {
                "Market Cap": info.get("marketCap", "N/A"),
                "PE Ratio": info.get("trailingPE", "N/A"),
                "Forward PE": info.get("forwardPE", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A"),
                "Profit Margin": info.get("profitMargins", "N/A"),
                "ROE": info.get("returnOnEquity", "N/A"),
            }

            colored = {
                k: color_value(v, good=15, bad=5) if isinstance(v, (int, float)) else v
                for k, v in fundamentals.items()
            }

            st.table(pd.DataFrame(colored, index=[0]).T)


# --------------- Screener Tab --------------- #
with tab3:
    st.header("Top Movers Screener (Most Active US Stocks)")
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=25"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        results = r.json()["finance"]["result"][0]["quotes"]

        screener_df = pd.DataFrame(results)[
            ["symbol", "shortName", "regularMarketPrice", "regularMarketChangePercent", "regularMarketVolume"]
        ]
        st.dataframe(screener_df)

    except Exception as e:
        st.error(f"Error fetching most active stocks: {e}")

