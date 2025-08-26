import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def _extract_series(df: pd.DataFrame, col: str, ticker: str) -> pd.Series:
    s = df[col]
    if isinstance(s, pd.DataFrame):
        if ticker in s.columns:
            s = s[ticker]
        else:
            s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

def calculate_rsi_from_close(close: pd.Series, window: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window, min_periods=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_stock(ticker: str):
    try:
        # Fetch at least 1 year of data
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return f"⚠️ No price data for {ticker}", None

        # Extract OHLC series
        close = _extract_series(df, "Close", ticker)
        high = _extract_series(df, "High", ticker)
        low  = _extract_series(df, "Low", ticker)

        if close.dropna().empty:
            return f"⚠️ No close prices for {ticker}", None

        # RSI
        rsi = calculate_rsi_from_close(close)
        latest_rsi = float(rsi.iloc[-1]) if not rsi.empty else None
        if latest_rsi is not None:
            if latest_rsi > 70:
                rsi_text = f"RSI {latest_rsi:.2f} → Overbought"
            elif latest_rsi < 30:
                rsi_text = f"RSI {latest_rsi:.2f} → Oversold"
            else:
                rsi_text = f"RSI {latest_rsi:.2f} → Neutral"
        else:
            rsi_text = "⚠️ No RSI data"

        # Price & 52-week range
        latest_price = float(close.iloc[-1])
        hi_52wk = float(high.tail(252).max())
        lo_52wk = float(low.tail(252).min())

        # Moving averages
        ma50_series = close.rolling(50).mean()
        ma200_series = close.rolling(200).mean()
        ma_50 = float(ma50_series.iloc[-1]) if pd.notna(ma50_series.iloc[-1]) else float("nan")
        ma_200 = float(ma200_series.iloc[-1]) if pd.notna(ma200_series.iloc[-1]) else float("nan")

        # Crossovers
        cross_signal = None
        if len(ma50_series.dropna()) >= 2 and len(ma200_series.dropna()) >= 2:
            ma_50_prev = float(ma50_series.iloc[-2])
            ma_200_prev = float(ma200_series.iloc[-2])
            if pd.notna(ma_50_prev) and pd.notna(ma_200_prev) and pd.notna(ma_50) and pd.notna(ma_200):
                if ma_50_prev < ma_200_prev and ma_50 > ma_200:
                    cross_signal = "🚀 Golden Cross just happened! (Bullish)"
                elif ma_50_prev > ma_200_prev and ma_50 < ma_200:
                    cross_signal = "⚠️ Death Cross just happened! (Bearish)"
                elif ma_50 > ma_200:
                    cross_signal = "✅ 50-day above 200-day (Bullish trend)"
                else:
                    cross_signal = "❌ 50-day below 200-day (Bearish trend)"

        # Fundamental metrics
        tkr = yf.Ticker(ticker)
        info = tkr.info
        eps_growth = info.get("earningsQuarterlyGrowth", None)
        forward_pe = info.get("forwardPE", None)
        roc = info.get("returnOnCapital", None)

        return {
            "ticker": ticker,
            "price": latest_price,
            "rsi_text": rsi_text,
            "hi_52wk": hi_52wk,
            "lo_52wk": lo_52wk,
            "ma50_series": ma50_series,
            "ma200_series": ma200_series,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "cross_signal": cross_signal,
            "eps_growth": eps_growth,
            "forward_pe": forward_pe,
            "roc": roc,
            "close_series": close
        }

    except Exception as e:
        return

