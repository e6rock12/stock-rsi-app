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

def get_rsi_status_and_data(ticker, period="6mo", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        rsi = calculate_rsi(data)
        if rsi is None or rsi.empty:
            return f"⚠️ No data found for {ticker}", None
        latest_rsi = rsi.iloc[-1]

        if latest_rsi > 70:
            status = f"📈 {ticker}: RSI {latest_rsi:.2f} → Overbought"
        elif latest_rsi < 30:
            status = f"📉 {ticker}: RSI {latest_rsi:.2f} → Oversold"
        else:
            status = f"➖ {ticker}: RSI {latest_rsi:.2f} → Neutral"
        
        return status, rsi
    except Exception as e:
        return f"❌ Error retrieving {ticker}: {e}", None


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
            status, rsi_data = get_rsi_status_and_data(t)
            st.subheader(status)
            
            if rsi_data is not None:
                st.line_chart(rsi_data, height=200)
                # Add reference lines at RSI 30 and 70
                st.markdown(
                    "<span style='color:orange'>--- RSI 70 (Overbought)</span><br>"
                    "<span style='color:blue'>--- RSI 30 (Oversold)</span>", 
                    unsafe_allow_html=True
                )

