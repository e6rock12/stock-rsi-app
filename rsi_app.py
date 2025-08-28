import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def lookup_ticker(query: str) -> str:
    """
    Use Yahoo Finance search API to resolve company name to ticker symbol.
    If lookup fails, fallback to raw uppercase input.
    """
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "quotes" in data and len(data["quotes"]) > 0:
                return data["quotes"][0]["symbol"]  # take first match
    except Exception as e:
        st.warning(f"Lookup error: {e}")
    return query.upper().strip()  # fallback

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def golden_cross(ma50: pd.Series, ma200: pd.Series) -> str:
    """Check for Golden Cross or Death Cross"""
    if len(ma50) > 0 and len(ma200) > 0:
        if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
            return "Golden Cross ⚡ (Bullish)"
        elif ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
            return "Death Cross 💀 (Bearish)"
    return "No crossover"

def safe_get(info_dict, key, default="N/A"):
    try:
        val = info_dict.get(key, default)
        if isinstance(val, (int, float)):
            return round(val, 2)
        return val
    except Exception:
        return default

def colorize(val, good_high=True):
    """Color code fundamentals (green good, red bad)."""
    if isinstance(val, (int, float)):
        if good_high:
            return f"✅ {val}"
        else:
            return f"⚠️ {val}"
    return val


# ---------- Streamlit App ----------
st.set_page_config(page_title="Stock RSI & Fundamentals App", layout="wide")
st.title("📈 Stock RSI & Fundamentals App")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Screener"])

# -------- Tab 1: Stock Analysis --------
with tab1:
    user_input = st.text_input("Enter a stock ticker *or* company name:", "AAPL")
    ticker = lookup_ticker(user_input)
    st.write(f"Looking up: **{ticker}**")

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")

        if data.empty:
            st.error(f"No data found for {ticker}")
        else:
            st.subheader(f"{ticker} Price Chart with RSI")

            # RSI and Moving Averages
            data["RSI"] = calculate_rsi(data["Close"])
            data["MA50"] = data["Close"].rolling(window=50).mean()
            data["MA200"] = data["Close"].rolling(window=200).mean()

            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(data.index, data["Close"], label="Close Price", color="blue")
            ax1.plot(data.index, data["MA50"], label="50-day MA", color="orange")
            ax1.plot(data.index, data["MA200"], label="200-day MA", color="red")
            ax1.set_ylabel("Price")
            ax1.legend()

            ax2.plot(data.index, data["RSI"], label="RSI", color="purple")
            ax2.axhline(70, color="red", linestyle="--")
            ax2.axhline(30, color="green", linestyle="--")
            ax2.set_ylabel("RSI")
            ax2.legend()

            st.pyplot(fig)

            # Golden Cross Check
            signal = golden_cross(data["MA50"], data["MA200"])
            st.info(f"📊 Moving Average Signal: **{signal}**")

    except Exception as e:
        st.error(f"Error retrieving {ticker}: {e}")

# -------- Tab 2: Fundamentals --------
with tab2:
    ticker = lookup_ticker(user_input)
    stock = yf.Ticker(ticker)
    info = stock.info

    st.subheader(f"Fundamentals for {ticker} - {safe_get(info, 'longName')}")
    fundamentals = {
        "Forward P/E": colorize(safe_get(info, "forwardPE"), good_high=False),
        "EPS Growth (5y)": colorize(safe_get(info, "earningsQuarterlyGrowth")),
        "Return on Capital": colorize(safe_get(info, "returnOnEquity")),
        "52 Week High": safe_get(info, "fiftyTwoWeekHigh"),
        "52 Week Low": safe_get(info, "fiftyTwoWeekLow"),
        "Market Cap": safe_get(info, "marketCap"),
    }

    st.table(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]))

# -------- Tab 3: Screener --------
with tab3:
    st.subheader("🧮 Stock Screener (Most Active US Stocks)")

    try:
        # Get Yahoo "Most Active" stocks
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"
        resp = requests.get(url, timeout=5)
        data = resp.json()

        quotes = data.get("finance", {}).get("result", [])[0].get("quotes", [])
        tickers = [q["symbol"] for q in quotes[:25]]  # top 25 most active

        st.write("Analyzing top 25 most active stocks...")

        results = []
        for t in tickers:
            try:
                s = yf.Ticker(t)
                info = s.info
                hist = s.history(period="6mo")
                if hist.empty:
                    continue

                rsi = calculate_rsi(hist["Close"]).iloc[-1]

                results.append({
                    "Ticker": t,
                    "Name": safe_get(info, "shortName"),
                    "RSI": round(rsi, 2) if not pd.isna(rsi) else "N/A",
                    "Forward P/E": safe_get(info, "forwardPE"),
                    "EPS Growth": safe_get(info, "earningsQuarterlyGrowth"),
                    "Return on Capital": safe_get(info, "returnOnEquity"),
                })
            except Exception:
                continue

        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            st.warning("No screener results available.")

    except Exception as e:
        st.error(f"Error fetching US tickers: {e}")

