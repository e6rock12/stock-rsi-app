import streamlit as st
import yfinance as yf
import pandas as pd

# Try importing matplotlib for plotting, but don’t fail if unavailable
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
        # Fetch price data
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return f"⚠️ No price data for {ticker}", None

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

        latest_price = float(close.iloc[-1])
        hi_52wk = float(high.tail(252).max())
        lo_52wk = float(low.tail(252).min())

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

        # Fundamentals with safe get
        try:
            info = yf.Ticker(ticker).info
            eps_growth = info.get("earningsQuarterlyGrowth")
            forward_pe = info.get("forwardPE")
            roc = info.get("returnOnCapital")
        except Exception:
            eps_growth = forward_pe = roc = None

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
        return f"❌ Error retrieving {ticker}: {e}", None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Stock RSI & Fundamentals", layout="wide")
st.title("📈 Stock RSI, MAs & Fundamentals")

tickers_raw = st.text_input("Enter stock tickers (comma separated):", "AAPL, TSLA, MSFT")
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

if st.button("Analyze"):
    for t in tickers:
        result = analyze_stock(t)
        if isinstance(result, dict):
            st.subheader(f"📊 {result['ticker']}")

            # Price & 52-week
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${result['price']:.2f}")
            c2.metric("52-Week High", f"${result['hi_52wk']:.2f}")
            c3.metric("52-Week Low", f"${result['lo_52wk']:.2f}")

            # Moving averages
            c4, c5 = st.columns(2)
            c4.metric("50-Day MA", f"${result['ma_50']:.2f}" if pd.notna(result['ma_50']) else "N/A")
            c5.metric("200-Day MA", f"${result['ma_200']:.2f}" if pd.notna(result['ma_200']) else "N/A")

            # RSI
            st.write(result["rsi_text"])

            # Crossovers
            if result["cross_signal"]:
                st.info(result["cross_signal"])

            # Fundamentals
            c6, c7, c8 = st.columns(3)
            c6.metric("EPS Growth (QoQ)", f"{result['eps_growth']:.2f}" if result['eps_growth'] else "N/A")
            c7.metric("Forward PE", f"{result['forward_pe']:.2f}" if result['forward_pe'] else "N/A")
            c8.metric("Return on Capital", f"{result['roc']:.2f}" if result['roc'] else "N/A")

            # Plot chart if matplotlib available
            if MATPLOTLIB_AVAILABLE:
                try:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    close_series = result["close_series"]
                    ma50_series = result["ma50_series"]
                    ma200_series = result["ma200_series"]

                    ax.plot(close_series.index, close_series, label="Close Price", color="blue")
                    ax.plot(ma50_series.index, ma50_series, label="50-Day MA", color="orange")
                    ax.plot(ma200_series.index, ma200_series, label="200-Day MA", color="green")

                    golden_crosses = (ma50_series.shift(1) < ma200_series.shift(1)) & (ma50_series > ma200_series)
                    death_crosses = (ma50_series.shift(1) > ma200_series.shift(1)) & (ma50_series < ma200_series)
                    ax.scatter(close_series.index[golden_crosses], close_series[golden_crosses], marker="^", color="gold", s=100, label="Golden Cross")
                    ax.scatter(close_series.index[death_crosses], close_series[death_crosses], marker="v", color="red", s=100, label="Death Cross")

                    ax.set_title(f"{t} Price and Moving Averages")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price ($)")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Chart error for {t}: {e}")
            else:
                st.warning("Matplotlib not installed; charts unavailable")

            st.divider()
        else:
            st.error(result)

