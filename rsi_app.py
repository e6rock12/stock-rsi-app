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

def analyze_stock(ticker):
    try:
        # Fetch at least 1 year so 200-day MA is valid
        data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)

        if data.empty:
            return f"⚠️ No data for {ticker}", None, None

        # RSI
        rsi = calculate_rsi(data)
        if rsi is None or rsi.empty:
            return f"⚠️ No RSI data for {ticker}", None, None
        latest_rsi = rsi.iloc[-1]

        # Price
        latest_price = data["Close"].iloc[-1]

        # 52-week high and low
        high_52wk = data["Close"].max()
        low_52wk = data["Close"].min()

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
        if not pd.isna(ma_50) and not pd.isna(ma_200) and not pd.isna(ma_50_prev) and not pd.isna(ma_200_prev):
            if ma_50_prev < ma_200_prev and ma_50 > ma_200:
                cross_signal = "🚀 Golden Cross just happened! (Bullish)"
            elif ma_50_prev > ma_200_prev and ma_50 < ma_200:
                cross_signal = "⚠️ Death Cross just happened! (Bearish)"
            elif ma_50 > ma_200:
                cross_signal = "✅ 50-day above 200-day (Bullish trend)"
            else:
                cross_signal = "❌ 50-day below 200-day (Bearish trend)"

        # RSI status
        if latest_rsi > 70:
            rsi_status = f"RSI {latest_rsi:.2f} → Overbought"
        elif latest_rsi < 30:
            rsi_status = f"RSI {latest_rsi:.2f} → Oversold"
        else:
            rsi_status = f"RSI {latest_rsi:.2f} → Neutral"

        return {
            "ticker": ticker,
            "price": latest_price,
            "rsi_status": rsi_status,
            "high_52wk": high_52wk,
            "low_52wk": low_52wk,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "cross_signal": cross_signal
        }

    except Exception as e:
        return f"❌ Error retrieving {ticker}: {e}", None, None

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Stock RSI & Trends", layout="wide")

st.title("📈 Stock RSI and Trend Analyzer")

tickers = st.text_input("Enter stock tickers (comma separated):", "AAPL, TSLA, MSFT")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

if st.button("Analyze"):
    for t in tickers:
        result = analyze_stock(t)
        if isinstance(result, dict):
            st.subheader(f"📊 {result['ticker']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${result['price']:.2f}")
            col2.metric("52-Week High", f"${result['high_52wk']:.2f}")
            col3.metric("52-Week Low", f"${result['low_52wk']:.2f}")

            col4, col5 = st.columns(2)
            col4.metric("50-Day MA", f"${result['ma_50']:.2f}")
            col5.metric("200-Day MA", f"${result['ma_200']:.2f}")

            st.write(result["rsi_status"])
            if result["cross_signal"]:
                st.info(result["cross_signal"])
            st.divider()
        else:
            st.error(result)

