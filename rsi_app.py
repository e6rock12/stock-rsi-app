import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def calculate_rsi(data, window=14):
    if data.empty:
        return None
    delta = data["Close"].diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_stock_data(ticker, period="1y", interval="1d"):
    return yf.download(ticker, period=period, interval=interval, progress=False)

def get_metrics(ticker, data):
    info = yf.Ticker(ticker).info
    metrics = {}
    try:
        metrics["Latest Price"] = data["Close"].iloc[-1]
        metrics["52W High"] = data["Close"].max()
        metrics["52W Low"] = data["Close"].min()
        metrics["200d MA"] = data["Close"].rolling(200).mean().iloc[-1]
        metrics["50d MA"] = data["Close"].rolling(50).mean().iloc[-1]
        metrics["Golden Cross"] = (
            metrics["50d MA"] > metrics["200d MA"]
        )
        metrics["RSI"] = calculate_rsi(data).iloc[-1]

        # Fundamental metrics
        metrics["Forward PE"] = info.get("forwardPE", None)
        metrics["EPS Growth"] = info.get("earningsQuarterlyGrowth", None)
        metrics["Return on Capital"] = info.get("returnOnEquity", None)

    except Exception as e:
        st.error(f"Error getting metrics for {ticker}: {e}")
    return metrics

def plot_chart(ticker, data):
    plt.figure(figsize=(8, 4))
    plt.plot(data.index, data["Close"], label="Close", color="blue")
    plt.plot(data.index, data["Close"].rolling(50).mean(), label="50d MA", color="orange")
    plt.plot(data.index, data["Close"].rolling(200).mean(), label="200d MA", color="red")
    plt.title(f"{ticker} Price & Moving Averages")
    plt.legend()
    st.pyplot(plt)

# ---------- Streamlit App ----------
st.title("📈 RockStock RSI App")

tickers_input = st.text_input("Enter stock tickers (comma separated)", "AAPL, MSFT, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

tab1, tab2 = st.tabs(["📊 Screener", "🧠 Analysis"])

# ---- Tab 1: Screener ----
with tab1:
    st.subheader("Market Screener")
    for t in tickers:
        st.write(f"### {t}")
        data = get_stock_data(t)
        if data.empty:
            st.warning(f"No data for {t}")
            continue
        metrics = get_metrics(t, data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${metrics['Latest Price']:.2f}")
        col2.metric("52W High", f"${metrics['52W High']:.2f}")
        col3.metric("52W Low", f"${metrics['52W Low']:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("200d MA", f"${metrics['200d MA']:.2f}")
        col5.metric("50d MA", f"${metrics['50d MA']:.2f}")
        col6.metric("RSI (14d)", f"{metrics['RSI']:.2f}")

        # Golden Cross
        if metrics["Golden Cross"]:
            st.success("✨ Golden Cross detected (50d > 200d)")
        else:
            st.info("No Golden Cross yet (50d <= 200d)")

        # Fundamentals
        st.write("**Fundamentals:**")
        st.write(f"- Forward PE: {metrics['Forward PE']}")
        st.write(f"- EPS Growth: {metrics['EPS Growth']}")
        st.write(f"- Return on Capital: {metrics['Return on Capital']}")

        st.divider()

# ---- Tab 2: Analysis ----
with tab2:
    st.subheader("Stock Analysis & Charts")
    for t in tickers:
        st.write(f"### {t}")
        data = get_stock_data(t)
        if data.empty:
            st.warning(f"No data for {t}")
            continue
        metrics = get_metrics(t, data)
        plot_chart(t, data)

        # Narrative analysis
        analysis_text = f"""
        - **RSI**: {metrics['RSI']:.2f} → {"Overbought (>70)" if metrics['RSI'] > 70 else "Oversold (<30)" if metrics['RSI'] < 30 else "Neutral"}
        - **Golden Cross**: {"Yes ✅" if metrics['Golden Cross'] else "No ❌"}
        - **Forward P/E**: {metrics['Forward PE']}
        - **EPS Growth**: {metrics['EPS Growth']}
        - **Return on Capital**: {metrics['Return on Capital']}
        """
        st.markdown(analysis_text)
        st.divider()

