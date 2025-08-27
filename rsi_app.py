import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- RSI Helper -----------------
def calculate_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ----------------- Screener Builder -----------------
def build_screener(tickers):
    data = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            hist = stock.history(period="6mo")

            if hist.empty:
                continue

            # RSI
            rsi = calculate_rsi(hist["Close"])
            latest_rsi = rsi.iloc[-1]

            # Fundamentals
            eps_growth = info.get("earningsQuarterlyGrowth", None)
            forward_pe = info.get("forwardPE", None)
            roe = info.get("returnOnEquity", None)

            data.append({
                "Ticker": t,
                "RSI": round(latest_rsi, 2) if pd.notna(latest_rsi) else None,
                "EPS Growth (YoY %)": round(eps_growth * 100, 2) if eps_growth else None,
                "Forward P/E": round(forward_pe, 2) if forward_pe else None,
                "ROE %": round(roe * 100, 2) if roe else None
            })
        except Exception as e:
            st.warning(f"⚠️ Could not fetch data for {t}: {e}")

    return pd.DataFrame(data)

# ----------------- App Layout -----------------
st.set_page_config(page_title="Stock RSI & Screener", layout="wide")
st.title("📊 Stock RSI & Fundamental Screener")

# --- Screener Section ---
st.header("🔎 Multi-Ticker Screener")

tickers_input = st.text_area(
    "Enter tickers (comma-separated):", 
    "AAPL, MSFT, CSCO, TSLA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.button("Run Screener"):
    screener_df = build_screener(tickers)
    if not screener_df.empty:
        st.subheader("📋 Screener Results")
        st.dataframe(screener_df, use_container_width=True)

        # Highlight signals
        st.markdown("### 🚦 Trade Signals")
        oversold = screener_df[screener_df["RSI"] < 30]
        overbought = screener_df[screener_df["RSI"] > 70]

        if not oversold.empty:
            st.success(f"✅ Potential oversold: {', '.join(oversold['Ticker'])}")
        if not overbought.empty:
            st.error(f"⚠️ Potential overbought: {', '.join(overbought['Ticker'])}")

        # CSV Download
        st.download_button(
            "Download results as CSV",
            screener_df.to_csv(index=False).encode("utf-8"),
            "screener_results.csv",
            "text/csv"
        )

st.markdown("---")

# --- Detailed Single Ticker Analysis ---
st.header("📈 Detailed Stock Analysis")

ticker = st.text_input("Enter a ticker for detailed view:", "AAPL")

if ticker:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")

        if not data.empty:
            # Moving averages
            data["50MA"] = data["Close"].rolling(window=50).mean()
            data["200MA"] = data["Close"].rolling(window=200).mean()

            # RSI
            data["RSI"] = calculate_rsi(data["Close"])

            latest_price = data["Close"].iloc[-1]
            high_52w = data["Close"].max()
            low_52w = data["Close"].min()
            ma50 = data["50MA"].iloc[-1]
            ma200 = data["200MA"].iloc[-1]
            rsi_latest = data["RSI"].iloc[-1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${latest_price:.2f}")
            col2.metric("52W High", f"${high_52w:.2f}")
            col3.metric("52W Low", f"${low_52w:.2f}")

            col4, col5 = st.columns(2)
            col4.metric("50-Day MA", f"${ma50:.2f}")
            col5.metric("200-Day MA", f"${ma200:.2f}")

            st.metric("RSI (14)", f"{rsi_latest:.2f}")

            # Golden cross detection
            if ma50 > ma200:
                st.success("🌟 Golden Cross detected (50MA above 200MA)")
            elif ma50 < ma200:
                st.warning("❌ Death Cross (50MA below 200MA)")

            # Plot chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, data["Close"], label="Close Price")
            ax.plot(data.index, data["50MA"], label="50-Day MA", linestyle="--")
            ax.plot(data.index, data["200MA"], label="200-Day MA", linestyle="--")
            ax.set_title(f"{ticker} Price with Moving Averages")
            ax.legend()
            st.pyplot(fig)

        else:
            st.error("No historical data found for this ticker.")

    except Exception as e:
        st.error(f"Error retrieving {ticker}: {e}")

