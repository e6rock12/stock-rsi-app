import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt

# ---------- Helper Functions ----------
def get_stock_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period)
        return data
    except Exception as e:
        st.error(f"Error retrieving {ticker}: {e}")
        return None

def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        fundamentals = {
            "Company Name": info.get("longName", "N/A"),
            "EPS Growth": info.get("earningsQuarterlyGrowth", "N/A"),
            "Forward PE": info.get("forwardPE", "N/A"),
            "Return on Capital": info.get("returnOnEquity", "N/A"),
        }
        return fundamentals
    except Exception as e:
        st.error(f"Error fetching fundamentals for {ticker}: {e}")
        return {}

def moving_averages(data):
    ma50 = data["Close"].rolling(window=50).mean()
    ma200 = data["Close"].rolling(window=200).mean()
    return ma50, ma200

def golden_cross(ma50, ma200):
    if ma50.iloc[-1] > ma200.iloc[-1]:
        return "Golden Cross detected ✅"
    else:
        return "No Golden Cross ❌"

# ---------- Streamlit App ----------
st.set_page_config(page_title="Stock RSI & Fundamentals App", layout="wide")

st.title("📈 Stock RSI & Fundamentals App")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Stock Screener"])

# ---------- Tab 1: Stock Analysis ----------
with tab1:
    ticker_input = st.text_input("Enter Ticker Symbol or Company Name", "AAPL").upper()

    if ticker_input:
        data = get_stock_data(ticker_input, period="1y")

        if data is not None and not data.empty:
            latest_price = data["Close"].iloc[-1]
            rsi = calculate_rsi(data)

            ma50, ma200 = moving_averages(data)
            cross_signal = golden_cross(ma50, ma200)

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${latest_price:.2f}")
            col2.metric("52-Week High", f"${data['High'].max():.2f}")
            col3.metric("52-Week Low", f"${data['Low'].min():.2f}")

            col4, col5 = st.columns(2)
            col4.metric("200-Day MA", f"${ma200.iloc[-1]:.2f}")
            col5.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")

            st.info(cross_signal)

            st.line_chart(data["Close"], use_container_width=True)

# ---------- Tab 2: Fundamentals ----------
with tab2:
    if ticker_input:
        fundamentals = get_fundamentals(ticker_input)

        if fundamentals:
            st.subheader(f"Fundamentals for {fundamentals.get('Company Name', ticker_input)}")

            def color_value(val, good, bad):
                try:
                    if val == "N/A":
                        return "⚪ N/A"
                    if val >= good:
                        return f"🟢 {val:.2f}"
                    elif val <= bad:
                        return f"🔴 {val:.2f}"
                    else:
                        return f"🟡 {val:.2f}"
                except:
                    return

