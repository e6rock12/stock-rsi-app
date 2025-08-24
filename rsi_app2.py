import yfinance as yf
import pandas as pd
import ta

def get_stock_rsi(ticker, period="6mo", interval="1d"):
    try:
        # Download historical stock data
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            return f"⚠️ No data found for {ticker}"
        
        # Calculate RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        latest_rsi = data['RSI'].iloc[-1]
        
        # Interpret RSI
        if latest_rsi > 70:
            status = "📈 Overbought"
        elif latest_rsi < 30:
            status = "📉 Oversold"
        else:
            status = "➖ Neutral"
        
        return f"{ticker.upper()} → RSI: {latest_rsi:.2f} | Status: {status}"
    
    except Exception as e:
        return f"❌ Error retrieving {ticker}: {e}"

def main():
    print("=== Stock RSI Checker ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip()
        if ticker.lower() == "exit":
            print("Goodbye 👋")
            break
        if ticker:
            print(get_stock_rsi(ticker))
            print()

if __name__ == "__main__":
    main()

