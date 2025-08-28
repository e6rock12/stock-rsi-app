import streamlit as st
import yfinance as yf
import pandas as pd
import requests

# ==========================
# Utility Functions
# ==========================

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def golden_cross(ma50, ma200):
    try:
        if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
            return "Golden Cross"
        elif ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
            return "Death Cross"
        else:
            return "No Signal"
    except Exception:
        return "N/A"


def get_company_info(symbol):
    ticker = yf.Ticker(symbol)
    info = {}
    try:
        info = ticker.get_info()
    except Exception:
        return None

    return {
        "shortName": info.get("shortName", symbol),
        "forwardPE": info.get("forwardPE", None),
        "epsForward": info.get("epsForward", None),
        "returnOnEquity": info.get("returnOnEquity", None),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", None),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", None),
    }


def get_most_active(limit=25):
    url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count={limit}&scrIds=most_actives"
    try:
        data = requests.get(url).json()
        quotes = data['finance']['result'][0]['quotes']
        return [q['symbol'] for q in quotes]
    except Exception as e:
        print("Error fetching most active:", e)
        return []


# ==========================
# Streamlit App
# ==========================

st.title("📈 Stock RSI & Fundamentals App")

tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Stock Screener"])

# --- TAB 1: Stock Analysis ---
with tab1:
    st.header("Stock Analysis")
    tickers = st.text_input("Enter ticker symbols (comma separated)", "AAPL, MSFT")

    if tickers:
        for symbol in [t.strip().upper() for t in tickers.split(",")]:
            st.subheader(symbol)
            try:
                data = yf.download(symbol, period="1y")
                if data.empty:
                    st.error(f"Could not retrieve {symbol}")
                    continue

                data["RSI"] = calculate_rsi(data)
                ma50 = data['Close'].rolling(window=50).mean()
                ma200 = data['Close'].rolling(window=200).mean()

                st.line_chart(data[['Close', 'RSI']])
                st.write("Golden/Death Cross Signal:", golden_cross(ma50, ma200))

            except Exception as e:
                st.error(f"Error retrieving {symbol}: {e}")

# --- TAB 2: Fundamentals ---
with tab2:
    st.header("Fundamentals Lookup")
    tickers = st.text_input("Enter ticker symbols (comma separated)", "GOOGL, AMZN")

    if tickers:
        fundamentals = []
        for symbol in [t.strip().upper() for t in tickers.split(",")]:
            info = get_company_info(symbol)
            if info:
                fundamentals.append(info)
            else:
                st.error(f"Could not retrieve {symbol}")

        if fundamentals:
            df = pd.DataFrame(fundamentals)
            # Color code fundamentals
            def color_cells(val, col):
                if col == "forwardPE":
                    if val and val < 15:
                        return "background-color: lightgreen"
                    elif val and val > 30:
                        return "background-color: lightcoral"
                if col == "returnOnEquity":
                    if val and val > 0.15:
                        return "background-color: lightgreen"
                    elif val and val < 0.05:
                        return "background-color: lightcoral"
                return ""

            styled_df = df.style.apply(
                lambda row: [color_cells(row[col], col) for col in df.columns],
                axis=1
            )
            st.dataframe(styled_df, use_container_width=True)

# --- TAB 3: Stock Screener ---
with tab3:
    st.header("Most Active Stocks")
    most_active = get_most_active(25)
    if most_active:
        results = []
        for symbol in most_active:
            info = get_company_info(symbol)
            if info:
                results.append(info)

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
    else:
        st.error("Error fetching most active stocks.")

