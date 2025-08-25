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


def get_stock_data(ticker, period="1y", interval="1d"):
    """Download stock data and add RSI & 200DMA."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            return None, None, None, None

        # RSI
        rsi = calculate_rsi(data)

        # 200-day moving average
        data["200DMA"] = data["Close"].rolling(window=200).mean()

        # 52-week high/low
        high_52wk = data["Close"].max()
        low_52wk = data["Close"].min()

        return data, rsi, high_52wk, low_52wk
    except Exception as e:
        return None, None, None, None


# --- Streamlit UI ---
st.set_page_config(page_title="Stock RSI Dashboard", page_icon="📊", layout="wide")

st.title("📊 Stock RSI Dashboard")
st.markdown("Check if a stock is **Overbought, Oversold, or Neutral**, plus see **Price, RSI, 52-Week High/Low, and 200DMA**.")

tickers_input = st.text_input("Enter stock tickers (comma separated)", "AAPL, TSLA")

if st.button("Analyze"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if not tickers:
        st.warning("⚠️ Please enter at least one ticker.")
    else:
        for t in tickers:
            st.header(f"📌 {t}")
            data, rsi_data, high_52wk, low_52wk = get_stock_data(t)

            if data is None or rsi_data is None:
                st.error(f"❌ Could not retrieve data for {t}")
                continue

            # --- Key Stats ---
	    latest_price = data["Close"].iloc[-1]
            high_52wk = data["Close"].max()
            low_52wk = data["Close"].min()

            # Convert to floats (avoid Series formatting errors)
            latest_price = float(latest_price)
            high_52wk = float(high_52wk)
            low_52wk = float(low_52wk)

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${latest_price:.2f}")
            col2.metric("52-Week High", f"${high_52wk:.2f}")
            col3.metric("52-Week Low", f"${low_52wk:.2f}")

            # --- Charts ---
            chart_col1, chart_col2 = st.columns(2)

            # Price with 200DMA
            with chart_col1:
                st.subheader("Price & 200-Day MA")
                price_chart = data[["Close", "200DMA"]].dropna()
                st.line_chart(price_chart, height=300)

            # RSI chart
            with chart_col2:
                st.subheader("RSI (14-day)")
                st.line_chart(rsi_data, height=300)
                latest_rsi = rsi_data.iloc[-1]
                if latest_rsi > 70:
                    st.warning(f"RSI {latest_rsi:.2f} → Overbought 📈")
                elif latest_rsi < 30:
                    st.success(f"RSI {latest_rsi:.2f} → Oversold 📉")
                else:
                    st.info(f"RSI {latest_rsi:.2f} → Neutral ➖")

