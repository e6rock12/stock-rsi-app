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

def resolve_ticker(user_input):
    """Try to resolve user input (ticker or company name)"""
    try:
        ticker = yf.Ticker(user_input)
        info = ticker.info
        if "shortName" in info:  # valid ticker
            return user_input.upper(), info.get("shortName", "")
    except:
        pass

    # Fallback: try search suggestions
    try:
        df = yf.search(user_input)
        if not df.empty:
            symbol = df.iloc[0]["symbol"]
            name = df.iloc[0]["shortname"]
            return symbol.upper(), name
    except:
        pass

    return user_input.upper(), ""  # return raw if not resolvable

def get_stock_data(ticker, period="1y", interval="1d"):
    return yf.download(ticker, period=period, interval=interval, progress=False)

def get_metrics(ticker, data):
    info = yf.Ticker(ticker).info
    metrics = {}
    try:
        metrics["Name"] = info.get("shortName", "")
        metrics["Latest Price"] = float(data["Close"].iloc[-1])
        metrics["52W High"] = float(data["Close"].max())
        metrics["52W Low"] = float(data["Close"].min())
        metrics["200d MA"] = float(data["Close"].rolling(200).mean().iloc[-1])
        metrics["50d MA"] = float(data["Close"].rolling(50).mean().iloc[-1])
        metrics["Golden Cross"] = metrics["50d MA"] > metrics["200d MA"]

        rsi_series = calculate_rsi(data)
        metrics["RSI"] = float(rsi_series.iloc[-1]) if rsi_series is not None else None

        # Fundamentals
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

def color_metric(label, value, good_condition, bad_condition, suffix=""):
    """Helper to return color-coded metrics"""
    if value is None:
        return f"- {label}: N/A"
    try:
        if good_condition(value):
            return f"- {label}: <span style='color:green;font-weight:bold'>{value:.2f}{suffix}</span>"
        elif bad_condition(value):
            return f"- {label}: <span style='color:red;font-weight:bold'>{value:.2f}{suffix}</span>"
        else:
            return f"- {label}: {value:.2f}{suffix}"
    except Exception:
        return f"- {label}: {value}"

def color_rsi(value):
    """Specialized RSI coloring"""
    if value is None:
        return "- RSI: N/A"
    if value > 70:
        return f"- RSI: <span style='color:red;font-weight:bold'>{value:.2f} (Overbought)</span>"
    elif value < 30:
        return f"- RSI: <span style='color:green;font-weight:bold'>{value:.2f} (Oversold)</span>"
    else:
        return f"- RSI: {value:.2f} (Neutral)"

# ---------- Streamlit App ----------
st.title("📈 RockStock RSI App")

tickers_input = st.text_input("Enter stock tickers or company names (comma separated)", "Apple, MSFT, Tesla")
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

tab1, tab2, tab3 = st.tabs(["📊 Screener", "🧠 Analysis", "🔍 Screener by Criteria"])

# ---- Tab 1: Screener ----
with tab1:
    st.subheader("Market Screener")
    for t in tickers:
        ticker, cname = resolve_ticker(t)
        data = get_stock_data(ticker)
        if data.empty:
            st.warning(f"No data for {t}")
            continue
        metrics = get_metrics(ticker, data)
        display_name = f"{ticker} ({metrics['Name']})" if metrics["Name"] else ticker
        st.write(f"### {display_name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${metrics['Latest Price']:.2f}")
        col2.metric("52W High", f"${metrics['52W High']:.2f}")
        col3.metric("52W Low", f"${metrics['52W Low']:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("200d MA", f"${metrics['200d MA']:.2f}")
        col5.metric("50d MA", f"${metrics['50d MA']:.2f}")
        if metrics["RSI"] is not None:
            col6.metric("RSI (14d)", f"{metrics['RSI']:.2f}")
        else:
            col6.metric("RSI (14d)", "N/A")

        if metrics["Golden Cross"]:
            st.success("✨ Golden Cross detected (50d > 200d)")
        else:
            st.info("No Golden Cross yet (50d <= 200d)")

        st.markdown("**Fundamentals:**", unsafe_allow_html=True)
        st.markdown(color_rsi(metrics["RSI"]), unsafe_allow_html=True)
        st.markdown(color_metric("Forward PE", metrics["Forward PE"], lambda v: v < 20, lambda v: v > 40), unsafe_allow_html=True)
        st.markdown(color_metric("EPS Growth", metrics["EPS Growth"], lambda v: v > 0.1, lambda v: v < 0), unsafe_allow_html=True)
        st.markdown(color_metric("Return on Capital", metrics["Return on Capital"], lambda v: v > 0.15, lambda v: v < 0.05), unsafe_allow_html=True)

        st.divider()

# ---- Tab 2: Analysis ----
with tab2:
    st.subheader("Stock Analysis & Charts")
    for t in tickers:
        ticker, cname = resolve_ticker(t)
        data = get_stock_data(ticker)
        if data.empty:
            st.warning(f"No data for {t}")
            continue
        metrics = get_metrics(ticker, data)
        display_name = f"{ticker} ({metrics['Name']})" if metrics["Name"] else ticker
        st.write(f"### {display_name}")

        plot_chart(ticker, data)

        st.markdown("**Analysis:**", unsafe_allow_html=True)
        st.markdown(color_rsi(metrics["RSI"]), unsafe_allow_html=True)
        st.markdown(f"- Golden Cross: {'✅ Yes' if metrics['Golden Cross'] else '❌ No'}", unsafe_allow_html=True)

        st.markdown("**Fundamentals:**", unsafe_allow_html=True)
        st.markdown(color_metric("Forward PE", metrics["Forward PE"], lambda v: v < 20, lambda v: v > 40), unsafe_allow_html=True)
        st.markdown(color_metric("EPS Growth", metrics["EPS Growth"], lambda v: v > 0.1, lambda v: v < 0), unsafe_allow_html=True)
        st.markdown(color_metric("Return on Capital", metrics["Return on Capital"], lambda v: v > 0.15, lambda v: v < 0.05), unsafe_allow_html=True)

        st.divider()

# ---- Tab 3: Screener by Criteria ----
with tab3:
    st.subheader("Custom Screener")

    # Criteria sliders
    min_pe = st.slider("Max Forward P/E", 0, 60, 25)
    min_eps = st.slider("Min EPS Growth", -0.5, 0.5, 0.1)
    min_roc = st.slider("Min Return on Capital", 0.0, 0.5, 0.1)
    min_rsi, max_rsi = st.slider("RSI Range", 0, 100, (30, 70))
    
    st.write("🔄 Running Screener on S&P 500 (may take 1–2 minutes)...")

    # Load S&P 500 tickers
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    symbols = sp500["Symbol"].tolist()

    results = []
    for symbol in symbols[:100]:  # limit for demo, can expand
        data = get_stock_data(symbol)
        if data.empty:
            continue
        m = get_metrics(symbol, data)
        if m["Forward PE"] and m["Forward PE"] < min_pe \
           and m["EPS Growth"] and m["EPS Growth"] > min_eps \
           and m["Return on Capital"] and m["Return on Capital"] > min_roc \
           and m["RSI"] and min_rsi <= m["RSI"] <= max_rsi:
            results.append([symbol, m["Name"], m["Forward PE"], m["EPS Growth"], m["Return on Capital"], m["RSI"]])

    if results:
        df = pd.DataFrame(results, columns=["Ticker", "Company", "Fwd P/E", "EPS Growth", "ROC", "RSI"])
        st.dataframe(df)
    else:
        st.warning("No stocks matched your criteria.")

