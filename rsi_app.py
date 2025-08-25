import streamlit as st
import yfinance as yf
import pandas as pd

def calculate_rsi(data, window=14):
    if data.empty:
        return None
    
    delta = data['Close'].diff().dropna()
    delta = delta.squeeze()  # ensure 1D

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_rsi_status(ticker, period="6mo", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        rsi = calculate_rsi(data)
        if rsi is None or rsi.empty:
            return f"⚠️ No data found for {ticker}"
        latest_rsi = rsi.iloc[-1]
        if latest_rsi > 70:
            return f"📈 {ticker}: RSI {latest_rsi:.2f} → Overbought"
        elif latest_rsi < 30:
            return f"📉 {ticker}: RSI {latest_rsi:.2f} → Oversold"
        else:
            return f"➖ {ticker}: RSI {latest_rsi:.2f} → Neutral"
    except Exception as e:
        return f"❌ Error retrieving {ticker}: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Stock RSI Checker", page_icon="📊", layout="centered")

st.title("📊 Stock RSI Checker")
st.markdown("Enter one or more stock tickers to check if they are **Overbought, Oversold, or Neutral** based on RSI (14-day).")

# Input box
tickers_input = st.text_input("Enter stock tickers (comma separated)", "AAPL, TSLA")

if st.button("Check RSI"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if not tickers:
        st.warning("⚠️ Please enter at least one ticker.")
    else:
        for t in tickers:
            status = get_rsi_status(t)
            st.write(status)

