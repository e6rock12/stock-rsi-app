import yfinance as yf
import pandas as pd
import streamlit as st

def calculate_rsi(data, window=14):
    if data.empty:
        return None
    
    delta = data['Close'].diff().dropna()
    delta = delta.squeeze()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_stock_data(ticker, period="1y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    return data

def analyze_stock(ticker):
    try:
        data = get_stock_data(ticker)
        if data.empty:
            return f"⚠️ No data found for {ticker}", None, None

        # RSI
        rsi = calculate_rsi(data)
        latest_rsi = rsi.iloc[-1]

        if latest_rsi > 70:
            rsi_status = f"RSI {latest_rsi:.2f} → Overbought"
        elif latest_rsi < 30:
            rsi_status = f"RSI {latest_rsi:.2f} → Oversold"
        else:
            rsi_status = f"RSI {latest_rsi:.2f} → Neutral"

        # Price metrics
        latest_price = data["Close"].iloc[-1]
        high_52w = data["Close"].max()
        low_52w = data["Close"].min()

        # Moving averages
        ma_50 = data["Close"].rolling(50).mean().iloc[-1]
        ma_200 = data["Close"].rolling(200).mean().iloc[-1]

        # Golden Cross check
        golden_cross = None
        if not pd.isna(ma_50) and not pd.isna(ma_200):
            if ma_50 > ma_200:
                golden_cross = "✅ Golden Cross (50 > 200)"
            else:
                golden_cross = "❌ No Golden Cross (50 < 200)"

        return {
            "ticker": ticker,
            "latest_price": latest_price,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "rsi_status": rsi_status,
            "golden_cross": golden_cross
        }
    except Exception as e:
        return f"❌ Error retrieving {ticker}: {e}", None, None


# --- Streamlit UI ---
st.title("📈 Stock RSI & Moving Average Dashboard")

tickers = st.text_input("Enter stock tickers (comma separated):", "AAPL, TSLA, MSFT").upper().split(",")
tickers = [t.strip() for t in tickers if t.strip()]

for ticker in tickers:
    result = analyze_stock(ticker)

    if isinstance(result, dict):
        st.subheader(result["ticker"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${result['latest_price']:.2f}")
        col2.metric("52W High", f"${result['high_52w']:.2f}")
        col3.metric("52W Low", f"${result['low_52w']:.2f}")

        col4, col5 = st.columns(2)
        col4.metric("50-Day MA", f"${result['ma_50']:.2f}")
        col5.metric("200-Day MA", f"${result['ma_200']:.2f}")

        st.write(result["rsi_status"])
        if result["golden_cross"]:
            st.success(result["golden_cross"]) if "✅" in result["golden_cross"] else st.warning(result["golden_cross"])

    else:
        st.error(result)

