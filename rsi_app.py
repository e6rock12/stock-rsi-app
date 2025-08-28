import streamlit as st
import yfinance as yf
import pandas as pd

# ---------- Helpers ----------
def _extract_series(df: pd.DataFrame, col: str, ticker: str) -> pd.Series:
    try:
        return df[col]
    except KeyError:
        st.warning(f"{ticker}: Could not find {col}")
        return pd.Series(dtype=float)

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return None, None
        return stock, hist
    except Exception as e:
        st.error(f"Error retrieving {ticker}: {e}")
        return None, None

def compute_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def golden_cross(ma50, ma200):
    try:
        if pd.isna(ma50.iloc[-1]) or pd.isna(ma200.iloc[-1]):
            return "Not enough data for Golden Cross ❌"
        if float(ma50.iloc[-1]) > float(ma200.iloc[-1]):
            return "Golden Cross detected ✅"
        else:
            return "No Golden Cross ❌"
    except Exception as e:
        return f"Error calculating Golden Cross: {e}"

# ---------- Streamlit Layout ----------
st.title("📈 Stock RSI & Fundamentals Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Stock Screener"]
)

# ---------------- Tab 1: Stock Analysis ----------------
with tab1:
    st.header("🔍 Stock Analysis")

    user_input = st.text_input("Enter stock ticker or company name (e.g., AAPL or Apple):", "AAPL")

    if user_input:
        ticker = user_input.upper().strip()
        stock, hist = get_stock_data(ticker)

        if stock and hist is not None and not hist.empty:
            company_name = stock.info.get("longName", "Unknown Company")
            st.subheader(f"{company_name} ({ticker})")

            hist["RSI"] = compute_rsi(hist["Close"])
            ma50 = hist["Close"].rolling(50).mean()
            ma200 = hist["Close"].rolling(200).mean()
            cross_signal = golden_cross(ma50, ma200)

            st.line_chart(hist[["Close", "RSI"]])
            st.write(f"📊 **Golden Cross Signal:** {cross_signal}")

# ---------------- Tab 2: Fundamentals ----------------
with tab2:
    st.header("📊 Fundamentals")

    if user_input:
        ticker = user_input.upper().strip()
        stock, hist = get_stock_data(ticker)

        if stock:
            info = stock.info
            company_name = info.get("longName", "Unknown Company")
            st.subheader(f"{company_name} ({ticker})")

            fundamentals = {
                "Forward P/E": info.get("forwardPE"),
                "EPS Growth (YoY)": info.get("earningsQuarterlyGrowth"),
                "Return on Capital": info.get("returnOnEquity")
            }

            for metric, value in fundamentals.items():
                if value is None:
                    st.write(f"{metric}: Data not available")
                else:
                    if metric == "Forward P/E":
                        color = "🟢" if value < 20 else "🟡" if value < 35 else "🔴"
                    elif metric == "EPS Growth (YoY)":
                        color = "🟢" if value > 0.1 else "🟡" if value > 0 else "🔴"
                    elif metric == "Return on Capital":
                        color = "🟢" if value > 0.15 else "🟡" if value > 0.05 else "🔴"
                    else:
                        color = "⚪"

                    st.write(f"{metric}: {value:.2f} {color}")

# ---------------- Tab 3: Stock Screener ----------------
with tab3:
    st.header("🧮 Stock Screener (Most Active US Stocks)")

    # Fetch most active stocks
    try:
        tickers = yf.Tickers(" ".join([
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
            "NVDA", "AMD", "NFLX", "CSCO", "INTC", "BAC",
            "JPM", "XOM", "PFE", "KO", "PEP", "WMT", "DIS"
        ]))
    except Exception as e:
        st.error(f"Error fetching US tickers: {e}")
        tickers = None

    if tickers:
        pe_max = st.number_input("Max Forward P/E", value=25.0)
        eps_min = st.number_input("Min EPS Growth (YoY)", value=0.05)
        roc_min = st.number_input("Min Return on Capital", value=0.1)

        results = []
        for t in tickers.tickers.keys():
            try:
                info = tickers.tickers[t].info
                pe = info.get("forwardPE")
                eps = info.get("earningsQuarterlyGrowth")
                roc = info.get("returnOnEquity")
                name = info.get("shortName", t)

                if pe and eps and roc:
                    if pe < pe_max and eps > eps_min and roc > roc_min:
                        results.append([t, name, pe, eps, roc])
            except Exception:
                continue

        if results:
            df = pd.DataFrame(results, columns=["Ticker", "Name", "Forward P/E", "EPS Growth", "Return on Capital"])
            st.dataframe(df)
        else:
            st.write("No stocks matched your criteria.")

