import streamlit as st
import yfinance as yf
import pandas as pd

# ----------------------------
# Utility functions
# ----------------------------

@st.cache_data(ttl=3600)
def fetch_data(tickers):
    """Fetch 1 year of daily stock data for one or more tickers."""
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(tickers, period="1y", group_by="ticker", auto_adjust=True)
    return data

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def golden_cross(ma50, ma200):
    if len(ma50) == 0 or len(ma200) == 0:
        return "Insufficient data"
    if ma50.iloc[-1] > ma200.iloc[-1]:
        return "Golden Cross (Bullish)"
    else:
        return "Death Cross (Bearish)"

def fetch_most_active():
    """Scrape Yahoo Finance Most Active Stocks page"""
    url = "https://finance.yahoo.com/most-active"
    tables = pd.read_html(url)
    df = tables[0]
    return df

# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Stock RSI App", layout="wide")

st.title("📈 Stock RSI & Fundamentals Explorer")

tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Screener"])

# ----------------------------
# Tab 1: Analysis
# ----------------------------
with tab1:
    st.header("🔍 Stock RSI & Moving Averages")

    tickers_input = st.text_input("Enter one or more stock tickers (comma-separated):", "AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if st.button("Run Analysis"):
        try:
            data = fetch_data(tickers)
            for ticker in tickers:
                st.subheader(f"{ticker} - {yf.Ticker(ticker).info.get('shortName', '')}")
                try:
                    df = data[ticker] if len(tickers) > 1 else data
                    df["RSI"] = calculate_rsi(df["Close"])
                    df["MA50"] = df["Close"].rolling(window=50).mean()
                    df["MA200"] = df["Close"].rolling(window=200).mean()

                    st.line_chart(df[["Close", "MA50", "MA200"]])
                    st.line_chart(df[["RSI"]])

                    cross_signal = golden_cross(df["MA50"].dropna(), df["MA200"].dropna())
                    st.write(f"📊 Signal: **{cross_signal}**")

                except Exception as e:
                    st.error(f"Error analyzing {ticker}: {e}")
        except Exception as e:
            st.error(f"Could not retrieve data: {e}")

# ----------------------------
# Tab 2: Fundamentals
# ----------------------------
with tab2:
    st.header("📊 Company Fundamentals")

    tickers_input = st.text_input("Enter tickers for fundamentals (comma-separated):", "AAPL, MSFT")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if st.button("Fetch Fundamentals"):
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                st.subheader(f"{ticker} - {info.get('shortName', '')}")
                fundamentals = {
                    "Market Cap": info.get("marketCap", "N/A"),
                    "PE Ratio": info.get("trailingPE", "N/A"),
                    "Forward PE": info.get("forwardPE", "N/A"),
                    "EPS": info.get("trailingEps", "N/A"),
                    "Dividend Yield": info.get("dividendYield", "N/A"),
                }
                st.table(pd.DataFrame.from_dict(fundamentals, orient="index", columns=["Value"]))
            except Exception as e:
                st.error(f"Error retrieving {ticker}: {e}")

# ----------------------------
# Tab 3: Screener
# ----------------------------
with tab3:
    st.header("🧮 Most Active Stocks Screener")
    try:
        df = fetch_most_active()
        st.dataframe(df)

        st.markdown("👉 Click a ticker below to analyze:")
        for ticker in df["Symbol"].head(10):  # Show top 10 clickable tickers
            if st.button(f"Analyze {ticker}"):
                st.session_state["selected_ticker"] = ticker
                st.experimental_rerun()

    except Exception as e:
        st.error(f"Error fetching most active stocks: {e}")

