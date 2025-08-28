import streamlit as st
import yfinance as yf
import pandas as pd

# ---------- Helpers ----------
def get_stock_data(ticker: str, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None, None
        info = stock.info
        return df, info
    except Exception as e:
        st.error(f"Error retrieving {ticker}: {e}")
        return None, None

def calculate_rsi(data: pd.DataFrame, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def golden_cross(ma50, ma200):
    if ma50.empty or ma200.empty:
        return None
    try:
        if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
            return "Golden Cross"
        elif ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
            return "Death Cross"
        else:
            return None
    except Exception:
        return None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Stock RSI & Fundamentals", layout="wide")

st.title("📈 Stock RSI, Moving Averages & Fundamentals")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Fundamentals", "🧮 Stock Screener"])

with tab1:
    st.header("🔍 Stock Analysis")

    user_input = st.text_input("Enter a stock ticker or company name:", "AAPL")

    if user_input:
        try:
            ticker = yf.Ticker(user_input).ticker
        except:
            ticker = user_input.upper()
        df, info = get_stock_data(ticker)

        if df is not None and info is not None:
            st.subheader(f"{info.get('longName', ticker)} ({ticker})")

            # Moving averages
            ma50 = df["Close"].rolling(window=50).mean()
            ma200 = df["Close"].rolling(window=200).mean()
            cross_signal = golden_cross(ma50, ma200)

            # RSI
            rsi = calculate_rsi(df)

            # Chart
            chart_data = pd.DataFrame({
                "Close": df["Close"],
                "50 MA": ma50,
                "200 MA": ma200,
                "RSI": rsi
            })
            st.line_chart(chart_data[["Close", "50 MA", "200 MA"]])

            if cross_signal:
                st.success(f"Signal: **{cross_signal}**")
            else:
                st.info("No Golden/Death Cross detected.")

            st.metric("Latest RSI", f"{rsi.iloc[-1]:.2f}")

with tab2:
    st.header("📊 Fundamentals")

    if user_input:
        try:
            ticker = yf.Ticker(user_input).ticker
        except:
            ticker = user_input.upper()
        df, info = get_stock_data(ticker)

        if info:
            forward_pe = info.get("forwardPE", "N/A")
            eps_growth = info.get("earningsQuarterlyGrowth", "N/A")
            roe = info.get("returnOnEquity", "N/A")

            fundamentals = {
                "Forward P/E": forward_pe,
                "EPS Growth (YoY)": eps_growth,
                "Return on Equity": roe
            }
            st.write(pd.DataFrame.from_dict(fundamentals, orient="index", columns=["Value"]))

with tab3:
    st.header("🧮 Stock Screener")

    st.write("Filter **Most Active US Stocks** based on your criteria.")

    # Screener inputs
    pe_max = st.number_input("Max Forward P/E", min_value=0.0, value=25.0)
    eps_min = st.number_input("Min EPS Growth (YoY)", min_value=-1.0, value=0.0)
    roe_min = st.number_input("Min Return on Equity", min_value=-1.0, value=0.1)
    rsi_min = st.slider("RSI Min", 0, 100, 30)
    rsi_max = st.slider("RSI Max", 0, 100, 70)

    if st.button("Run Screener 🚀"):
        st.info("Fetching most active US stocks...")
        try:
            tickers = yf.Tickers("AAPL MSFT AMZN TSLA NVDA META NFLX GOOGL INTC AMD JPM BAC XOM CVX WMT T")
            results = []

            for t in tickers.tickers:
                df, info = get_stock_data(t, period="6mo")
                if df is None or info is None:
                    continue

                rsi = calculate_rsi(df)
                latest_rsi = rsi.iloc[-1] if not rsi.empty else None
                fpe = info.get("forwardPE")
                eps_growth = info.get("earningsQuarterlyGrowth")
                roe = info.get("returnOnEquity")

                if not all([fpe, eps_growth, roe, latest_rsi]):
                    continue

                if fpe <= pe_max and eps_growth >= eps_min and roe >= roe_min and rsi_min <= latest_rsi <= rsi_max:
                    results.append({
                        "Ticker": t,
                        "Company": info.get("longName", t),
                        "Forward P/E": fpe,
                        "EPS Growth": eps_growth,
                        "ROE": roe,
                        "RSI": latest_rsi
                    })

            if results:
                df_results = pd.DataFrame(results)
                st.success(f"✅ Found {len(df_results)} matching stocks.")
                st.dataframe(df_results)
            else:
                st.warning("No stocks matched your filters.")
        except Exception as e:
            st.error(f"Error fetching screener data: {e}")

