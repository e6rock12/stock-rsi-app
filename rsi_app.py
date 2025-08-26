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

def analyze_stock(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if data.empty:
            return f"⚠️ No data found for {ticker}", None, None

        # RSI
        rsi = calculate_rsi(data)
        if rsi is None or rsi.empty:
            rsi_text = f"⚠️ No RSI data for {ticker}"
        else:
            latest_rsi = rsi.iloc[-1]
            if latest_rsi > 70:
                rsi_text = f"{ticker}: RSI {latest_rsi:.2f} → Overbought"
            elif latest_rsi < 30:
                rsi_text = f"{ticker}: RSI {latest_rsi:.2f} → Oversold"
            else:
                rsi_text = f"{ticker}: RSI {latest_rsi:.2f} → Neutral"

        # Price info
        latest_price = data["Close"].iloc[-1]
        hi_52wk = data["High"].rolling(window=252).max().iloc[-1]
        lo_52wk = data["Low"].rolling(window=252).min().iloc[-1]

        # Moving averages
        ma_50_series = data["Close"].rolling(50).mean()
        ma_200_series = data["Close"].rolling(200).mean()
        ma_50 = ma_50_series.iloc[-1]
        ma_200 = ma_200_series.iloc[-1]

        # Yesterday’s values (for crossover detection)
        ma_50_prev = ma_50_series.iloc[-2]
        ma_200_prev = ma_200_series.iloc[-2]

        # Golden/Death Cross detection
        cross_signal = None
        if not pd.isna(ma_50) and not pd.isna(ma_200):
            if ma_50_prev < ma_200_prev and ma_50 > ma_200:
                cross_signal = "🚀 Golden Cross just happened! (Bullish)"
            elif ma_50_prev > ma_200_prev and ma_50 < ma_200:
                cross_signal = "⚠️ Death Cross just happened! (Bearish)"
            elif ma_50 > ma_200:
                cross_signal = "✅ 50-day above 200-day (Bullish trend)"
            else:
                cross_signal = "❌ 50-day below 200-day (Bearish trend)"

        return rsi_text, latest_price, hi_52wk, lo_52wk, ma_50, ma_200, cross_signal

    except Exception as e:
        return f"❌ Error retrieving {ticker}: {e}", None, None, None, None, None, None


# Streamlit UI
st.title("📈 Stock RSI & Technical Dashboard")

tickers = st.text_input("Enter stock tickers (comma separated):", "AAPL, TSLA, MSFT")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

for t in tickers:
    rsi_text, latest_price, hi_52wk, lo_52wk, ma_50, ma_200, cross_signal = analyze_stock(t)
    
    st.subheader(t)
    st.write(rsi_text)

    if latest_price:
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${latest_price:.2f}")
        col2.metric("52-Week High", f"${hi_52wk:.2f}")
        col3.metric("52-Week Low", f"${lo_52wk:.2f}")

        col4, col5 = st.columns(2)
        col4.metric("50-Day MA", f"${ma_50:.2f}")
        col5.metric("200-Day MA", f"${ma_200:.2f}")

        if cross_signal:
            st.info(cross_signal)

