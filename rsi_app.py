import yfinance as yf
import pandas as pd
import streamlit as st

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
        if data.empty:
            return f"⚠️ No data found for {ticker}", None, None

        rsi = calculate_rsi(data)
        if rsi is None or rsi.empty:
            return f"⚠️ Could not calculate RSI for {ticker}", data, None

        latest_rsi = rsi.iloc[-1]
        if latest_rsi > 70:
            status = f"{ticker}: RSI {latest_rsi:.2f} → Overbought"
        elif latest_rsi < 30:
            status = f"{ticker}: RSI {latest_rsi:.2f} → Oversold"
        else:
            status = f"{ticker}: RSI {latest_rsi:.2f} → Neutral"

        return status, data, rsi

    except Exception as e:
        return f"❌ Error retrieving {ticker}: {e}", None, None

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Stock RSI Tracker", layout="wide")
st.title("📈 Stock RSI Tracker")

tickers = st.text_input("Enter stock tickers (comma separated):", "AAPL, MSFT, TSLA")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

if tickers:
    for ticker in tickers:
        st.subheader(f"📊 {ticker}")
        status, data, rsi = get_rsi_status(ticker)

        st.write(status)

        if data is not None and not data.empty and rsi is not None:
            # --- Key Stats ---
            latest_price = float(data["Close"].iloc[-1])
            high_52wk = float(data["Close"].max())
            low_52wk = float(data["Close"].min())
            ma_200 = float(data["Close"].rolling(200).mean().iloc[-1])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${latest_price:.2f}")
            col2.metric("52-Week High", f"${high_52wk:.2f}")
            col3.metric("52-Week Low", f"${low_52wk:.2f}")
            col4.metric("200-Day MA", f"${ma_200:.2f}")

            # --- Charts ---
            st.line_chart(data["Close"], height=250, use_container_width=True)
            st.line_chart(rsi, height=150, use_container_width=True)

