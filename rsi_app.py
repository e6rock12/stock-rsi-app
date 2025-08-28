import streamlit as st
import yfinance as yf
import pandas as pd

# -----------------
# Utility functions
# -----------------

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get("shortName", "Unknown Company")
        return name, stock
    except Exception as e:
        return None, None


def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def golden_cross(ma50, ma200):
    if len(ma50) == 0 or len(ma200) == 0:
        return "Not enough data"
    if ma50.iloc[-1] > ma200.iloc[-1]:
        return "Golden Cross (Bullish)"
    elif ma50.iloc[-1] < ma200.iloc[-1]:
        return "Death Cross (Bearish)"
    else:
        return "Neutral"


# -----------------
# Streamlit UI
# -----------------

st.set_page_config(page_title="Stock RSI App", layout="wide")
st.title("📈 Stock RSI & Fundamentals App")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Screener"])

# -----------------
# Stock Analysis Tab
# -----------------
with tab1:
    tickers_input = st.text_input("Enter ticker symbols (comma separated)", "AAPL, MSFT")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if st.button("Run Analysis"):
        for ticker in tickers:
            name, stock = get_stock_info(ticker)
            if not stock:
                st.error(f"Could not retrieve {ticker}")
                continue

            st.subheader(f"{name} ({ticker})")

            data = stock.history(period="1y")
            if data.empty:
                st.warning(f"No data for {ticker}")
                continue

            # Calculate indicators
            data["RSI"] = calculate_rsi(data)
            ma50 = data["Close"].rolling(window=50).mean()
            ma200 = data["Close"].rolling(window=200).mean()
            cross_signal = golden_cross(ma50, ma200)

            latest_rsi = data["RSI"].iloc[-1]
            st.metric("RSI (14d)", f"{latest_rsi:.2f}")
            st.write(f"**MA Signal**: {cross_signal}")

            st.line_chart(data[["Close", "RSI"]])

# -----------------
# Fundamentals Tab
# -----------------
with tab2:
    tickers_input_fund = st.text_input("Enter ticker symbols (comma separated)", "AAPL, MSFT", key="fund_input")
    tickers_fund = [t.strip().upper() for t in tickers_input_fund.split(",") if t.strip()]

    if st.button("Get Fundamentals"):
        fundamentals_data = []
        for ticker in tickers_fund:
            name, stock = get_stock_info(ticker)
            if not stock:
                st.error(f"Could not retrieve {ticker}")
                continue
            info = stock.info
            fundamentals_data.append({
                "Ticker": ticker,
                "Name": name,
                "Forward P/E": info.get("forwardPE"),
                "EPS Growth": info.get("earningsQuarterlyGrowth"),
                "Return on Capital": info.get("returnOnEquity"),
                "52w High": info.get("fiftyTwoWeekHigh"),
                "52w Low": info.get("fiftyTwoWeekLow")
            })
        df = pd.DataFrame(fundamentals_data)
        st.dataframe(df)

# -----------------
# Screener Tab
# -----------------
with tab3:
    st.write("### Most Active Stocks")

    try:
        active = yf.get_day_most_active()
        if not active.empty:
            active = active.reset_index()

            def show_details(ticker):
                st.session_state["selected_ticker"] = ticker

            for i, row in active.head(10).iterrows():
                ticker = row["Symbol"]
                name = row["Name"]
                st.write(f"**{ticker}** - {name}")
                if st.button(f"Analyze {ticker}", key=f"btn_{ticker}"):
                    st.session_state["selected_ticker"] = ticker

    except Exception as e:
        st.error(f"Error fetching most active stocks: {e}")

# If ticker clicked from Screener, load it into Analysis/Fundamentals
if "selected_ticker" in st.session_state:
    sel = st.session_state["selected_ticker"]
    st.success(f"Loading {sel} into Analysis and Fundamentals tabs...")

