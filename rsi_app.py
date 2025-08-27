import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from yahooquery import Screener

# ---------- Helpers ----------
def _extract_series(df: pd.DataFrame, col: str, ticker: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    else:
        st.warning(f"{col} not found for {ticker}")
        return pd.Series(dtype="float64")

def get_stock_data(ticker: str, period="1y"):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error retrieving {ticker}: {e}")
        return None

def calculate_rsi(data: pd.DataFrame, window: int = 14):
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_company_info(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "forwardPE": info.get("forwardPE", None),
            "epsGrowth": info.get("earningsGrowth", None),
            "roc": info.get("returnOnEquity", None),
        }
    except Exception as e:
        st.warning(f"Could not fetch fundamentals for {ticker}: {e}")
        return {"name": ticker, "forwardPE": None, "epsGrowth": None, "roc": None}

def get_us_tickers():
    """Fetch most active US tickers from Yahoo Screener"""
    try:
        s = Screener()
        screen = s.get_screeners("most_actives", count=100)
        quotes = screen.get("most_actives", {}).get("quotes", [])
        tickers = [q["symbol"] for q in quotes if "symbol" in q]
        return tickers
    except Exception as e:
        st.error(f"Error fetching US tickers: {e}")
        return []

# ---------- App ----------
st.set_page_config(page_title="RockStock RSI App", layout="wide")

st.title("📈 RockStock RSI App")

tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Sto]()

