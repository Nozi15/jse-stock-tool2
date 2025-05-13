import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="JSE Stock Dashboard", layout="wide")
st.title("📈 JSE Stock Investment Dashboard")

st.markdown("""
This live dashboard allows you to screen, track, and analyze JSE-listed stocks.
Use the default list or enter your own stock tickers. Live price and financial updates are pulled from Yahoo Finance.
""")

def_tickers = ['SOL.JO', 'SHP.JO', 'CPI.JO', 'MTN.JO', 'NPN.JO']
tickers = st.text_input("Enter JSE tickers separated by commas:", value=','.join(def_tickers))
tickers_list = [t.strip().upper() for t in tickers.split(',')]

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
            rsi_series = calculate_rsi(hist['Close']) if not hist.empty else pd.Series(dtype=float)
            latest_rsi = rsi_series.dropna().iloc[-1] if not rsi_series.dropna().empty else np.nan
            data[ticker] = {
                'Price': info.get('currentPrice'),
                'PE_Ratio': info.get('trailingPE'),
                'PB_Ratio': info.get('priceToBook'),
                'ROE': info.get('returnOnEquity'),
                'EPS_Growth': info.get('earningsGrowth'),
                'Dividend_Yield': info.get('dividendYield'),
                '1Y_Return_%': ((hist['Close'][-1] / hist['Close'][0]) - 1) * 100 if len(hist) >= 2 else None,
                'Volatility': info.get('beta'),
                'RSI': latest_rsi
            }
            rsi_data[ticker] = rsi_series
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {e}")
    return pd.DataFrame(data).T.reset_index().rename(columns={'index': 'Ticker'}), rsi_data

st.subheader("📊 Stock Metrics")
df, rsi_history = get_data(tickers_list)
st.dataframe(df, use_container_width=True)

def signal_logic(row):
    buy = (row['PE_Ratio'] < 15 if pd.notna(row['PE_Ratio']) else False) and           (row['EPS_Growth'] > 0.1 if pd.notna(row['EPS_Growth']) else False) and           (row['Dividend_Yield'] and row['Dividend_Yield'] > 0.02)
    sell = (row['RSI'] > 75 if pd.notna(row['RSI']) else False) or            (row['1Y_Return_%'] < -10 if pd.notna(row['1Y_Return_%']) else False)
    return pd.Series({'Buy_Signal': buy, 'Sell_Signal': sell})

if not df.empty:
    signals = df.apply(signal_logic, axis=1)
    df = pd.concat([df, signals], axis=1)

st.subheader("🛠 Buy/Sell Recommendations")
expected_cols = ['Ticker', 'Price', 'RSI', 'Buy_Signal', 'Sell_Signal']
existing_cols = [col for col in expected_cols if col in df.columns]

if existing_cols:
    st.dataframe(df[existing_cols], use_container_width=True)
else:
    st.warning("No Buy/Sell data available. Please check tickers or data source.")

metric = st.selectbox("Select metric for comparison", ['Price', 'PE_Ratio', 'PB_Ratio', 'EPS_Growth', 'Dividend_Yield', '1Y_Return_%', 'RSI'])
fig, ax = plt.subplots()
ax.bar(df['Ticker'], df[metric], color='steelblue')
ax.set_ylabel(metric)
ax.set_title(f"{metric} Comparison")
st.pyplot(fig)

st.subheader("📉 RSI Over Time")
selected_rsi_stock = st.selectbox("Select a stock to view RSI history", options=list(rsi_history.keys()))
if selected_rsi_stock in rsi_history:
    rsi_series = rsi_history[selected_rsi_stock]
    fig_rsi, ax_rsi = plt.subplots()
    ax_rsi.plot(rsi_series.index, rsi_series.values, label=f"RSI ({selected_rsi_stock})")
    ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought')
    ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold')
    ax_rsi.set_title(f"RSI Trend for {selected_rsi_stock}")
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend()
    st.pyplot(fig_rsi)

st.subheader("💼 Portfolio Tracker")
portfolio = st.data_editor(
    pd.DataFrame({
        'Ticker': tickers_list,
        'Quantity': [0] * len(tickers_list),
        'Buy_Price': [0.0] * len(tickers_list)
    }),
    num_rows="dynamic"
)

merged = pd.merge(portfolio, df[['Ticker', 'Price']], on='Ticker', how='left')
merged['Market_Value'] = merged['Quantity'] * merged['Price']
merged['Unrealized_PnL'] = (merged['Price'] - merged['Buy_Price']) * merged['Quantity']
st.dataframe(merged[['Ticker', 'Quantity', 'Buy_Price', 'Price', 'Market_Value', 'Unrealized_PnL']], use_container_width=True)

st.download_button("Download Stock Data", df.to_csv(index=False), file_name="jse_live_stock_data.csv", mime="text/csv")