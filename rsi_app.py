import streamlit as st
import yfinance as yf
import pandas as pd

# ---------- Helpers ----------
def _extract_series(df: pd.DataFrame, col: str, ticker: str) -> pd.Series:
    """
    Return a 1D float Series for the given column.
    Handles cases where yfinance returns a DataFrame (e.g., multi-ticker shape).
    Priority: column for ticker -> first column -> squeeze.
    """
    s = df[col]
    if isinstance(s, pd.DataFrame):
        if ticker in s.columns:
            s = s[ticker]
        else:
            # fall back to the first column
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
        # Ensure we have enough data for 200DMA
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return f"⚠️ No data for {ticker}", None

        # Extract clean 1D series
        close = _extract_series(df, "Close", ticker)
        high = _extract_series(df, "High", ticker)
        low  = _extract_series(df, "Low", ticker)

        if close.dropna().empty:
            return f"⚠️ No close prices for {ticker}", None

        # RSI (14)
        rsi = calculate_rsi_from_close(close)
        if rsi.dropna().empty:
            rsi_text = f"⚠️ No RSI data for {ticker}"
        else:
            latest_rsi = float(rsi.iloc[-1])
            if latest_rsi > 70:
                rsi_text = f"RSI {latest_rsi:.2f} → Overbought"
            elif latest_rsi < 30:
                rsi_text = f"RSI {latest_rsi:.2f} → Oversold"
            else:
                rsi_text = f"RSI {latest_rsi:.2f} → Neutral"

        # Price & 52-week range (use last 252 trading days)
        latest_price = float(close.iloc[-1])
        hi_52wk = float(high.tail(252).max())
        lo_52wk = float(low.tail(252).min())

        # Moving averages (use close)
        ma50_series = close.rolling(50).mean()
        ma200_series = close.rolling(200).mean()

        ma_50 = float(ma50_series.iloc[-1]) if pd.notna(ma50_series.iloc[-1]) else float("nan")
        ma_200 = float(ma200_series.iloc[-1]) if pd.notna(ma200_series.iloc[-1]) else float("nan")

        # Need previous values to detect crossover
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

        return {
            "ticker": ticker,
            "price": latest_price,
            "rsi_text": rsi_text,
            "hi_52wk": hi_52wk,
            "lo_52wk": lo_52wk,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "cross_signal": cross_signal,
        }

    except Exception as e:
        # Surface the exact error in Streamlit
        return f"❌ Error retrieving {ticker}: {e}", None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Stock RSI & Trends", layout="wide")
st.title("📈 Stock RSI & Trend Analyzer")

tickers_raw = st.text_input("Enter stock tickers (comma separated):", "AAPL, TSLA, MSFT")
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

if st.button("Analyze"):
    for t in tickers:
        result = analyze_stock(t)
        if isinstance(result, dict):
            st.subheader(f"📊 {result['ticker']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${result['price']:.2f}")
            c2.metric("52-Week High", f"${result['hi_52wk']:.2f}")
            c3.metric("52-Week Low", f"${result['lo_52wk']:.2f}")

            c4, c5 = st.columns(2)
            c4.metric("50-Day MA", f"${result['ma_50']:.2f}" if pd.notna(result['ma_50']) else "N/A")
            c5.metric("200-Day MA", f"${result['ma_200']:.2f}" if pd.notna(result['ma_200']) else "N/A")

            st.write(result["rsi_text"])
            if result["cross_signal"]:
                st.info(result["cross_signal"])
            st.divider()
        else:
            st.error(result)

