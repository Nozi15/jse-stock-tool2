import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="JSE Stock Dashboard", layout="wide")
st.title("📈 JSE Stock Investment Dashboard")

st.markdown("""
This live dashboard allows you to screen, track, and analyze JSE-listed stocks.
If live data fails to load, you can upload a CSV as a fallback.
""")

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=3600)
def get_data(tickers):
    data = {}
    rsi_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1mo")
            if hist.empty or 'Close' not in hist:
                raise ValueError("Empty or invalid historical data")
            rsi_series = calculate_rsi(hist['Close'])
            latest_rsi = rsi_series.dropna().iloc[-1] if not rsi_series.dropna().empty else np.nan
            data[ticker] = {
                'Price': info.get('currentPrice'),
                'PE_Ratio': info.get('trailingPE'),
                'PB_Ratio': info.get('priceToBook'),
                'ROE': info.get('returnOnEquity'),
                'EPS_Growth': info.get('earningsGrowth'),
                'Dividend_Yield': info.get('dividendYield'),
                '1Y_Return_%': ((hist['Close'][-1] / hist['Close'][0]) - 1) * 100,
                'Volatility': info.get('beta'),
                'RSI': latest_rsi
            }
            rsi_data[ticker] = rsi_series
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch data for {ticker}: {e}")
    return pd.DataFrame(data).T.reset_index().rename(columns={'index': 'Ticker'}), rsi_data

# Input options
st.subheader("📥 Data Input")
option = st.radio("Select Data Source", ["Live from Yahoo Finance", "Upload CSV"])
rsi_history = {}

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("CSV data loaded successfully!")
    else:
        df = pd.DataFrame()
else:
    def_tickers = ['SOL', 'SHP', 'CPI', 'MTN', 'NPN']
    tickers = st.text_input("Enter tickers separated by commas:", value=','.join(def_tickers))
    tickers_list = [t.strip().upper() for t in tickers.split(',')]
    df, rsi_history = get_data(tickers_list)

# Signal logic
def signal_logic(row):
    buy = (row.get('PE_Ratio', 0) < 15) and (row.get('EPS_Growth', 0) > 0.1) and (row.get('Dividend_Yield', 0) > 0.02)
    sell = (row.get('RSI', 0) > 75) or (row.get('1Y_Return_%', 0) < -10)
    return pd.Series({'Buy_Signal': buy, 'Sell_Signal': sell})

if not df.empty:
    signals = df.apply(signal_logic, axis=1)
    df = pd.concat([df, signals], axis=1)

    st.subheader("📊 Stock Metrics")
    st.dataframe(df, use_container_width=True)

    st.subheader("🛠 Buy/Sell Recommendations")
    expected_cols = ['Ticker', 'Price', 'RSI', 'Buy_Signal', 'Sell_Signal']
    existing_cols = [col for col in expected_cols if col in df.columns]
    if existing_cols:
        st.dataframe(df[existing_cols], use_container_width=True)
    else:
        st.warning("No Buy/Sell data available.")

    st.subheader("📉 RSI Over Time")
    if rsi_history:
        selected_rsi_stock = st.selectbox("Select stock to view RSI", options=list(rsi_history.keys()))
        if selected_rsi_stock in rsi_history:
            rsi_series = rsi_history[selected_rsi_stock]
            fig, ax = plt.subplots()
            ax.plot(rsi_series.index, rsi_series.values, label=f"RSI ({selected_rsi_stock})")
            ax.axhline(70, color='red', linestyle='--', label='Overbought')
            ax.axhline(30, color='green', linestyle='--', label='Oversold')
            ax.set_title(f"RSI Trend for {selected_rsi_stock}")
            ax.set_ylabel("RSI")
            ax.legend()
            st.pyplot(fig)

    st.subheader("💼 Portfolio Tracker")
    portfolio = st.data_editor(
        pd.DataFrame({
            'Ticker': df['Ticker'] if 'Ticker' in df else [],
            'Quantity': [0] * len(df),
            'Buy_Price': [0.0] * len(df)
        }),
        num_rows="dynamic"
    )
    merged = pd.merge(portfolio, df[['Ticker', 'Price']], on='Ticker', how='left')
    merged['Market_Value'] = merged['Quantity'] * merged['Price']
    merged['Unrealized_PnL'] = (merged['Price'] - merged['Buy_Price']) * merged['Quantity']
    st.dataframe(merged[['Ticker', 'Quantity', 'Buy_Price', 'Price', 'Market_Value', 'Unrealized_PnL']], use_container_width=True)

    st.download_button("Download Stock Data", df.to_csv(index=False), file_name="jse_stock_data.csv", mime="text/csv")
else:
    st.warning("No data available. Please upload a CSV or check your ticker input.")