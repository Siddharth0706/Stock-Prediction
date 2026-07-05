import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import seaborn as sns
import datetime as dt
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
from ta import momentum

# Set date range
start = '2020-01-01'
stop = dt.date.today().strftime("%Y-%m-%d")

# Streamlit title
st.title('Stock Mark')

# Button for navigation
selected_page = st.radio("Select a Page", ["Home", "Educational Resources"])

# Page content
if selected_page == "Home":
    st.title("Welcome to the Stock Market Web Page!")
elif selected_page == "Educational Resources":
    st.title("Educational Resources")
    st.write("Expand your knowledge about investing, trading strategies, and market analysis with these resources:")

    st.subheader("Articles")
    st.markdown("- [Introduction to Stock Market](https://www.example.com/article1)")
    st.markdown("- [Technical Analysis Basics](https://www.example.com/article2)")
    st.markdown("- [Fundamental Analysis Guide](https://www.example.com/article3)")

    st.subheader("Tutorials")
    st.markdown("- [Stock Market Investing 101](https://www.example.com/tutorial1)")
    st.markdown("- [Options Trading Strategies](https://www.example.com/tutorial2)")
    st.markdown("- [Introduction to Forex Trading](https://www.example.com/tutorial3)")

    st.subheader("Videos")
    st.markdown("- [Candlestick Charting Explained](https://www.example.com/video1)")
    st.markdown("- [Risk Management Techniques](https://www.example.com/video2)")
    st.markdown("- [Value Investing Principles](https://www.example.com/video3)")

# Input for stock ticker
company = st.text_input('Enter Stock', 'META')

# Download stock data
data = yf.download(company, start=start, end=stop)

# Reset the index to convert the DateTime index to a column
data.reset_index(inplace=True)

# Convert the 'Date' column to date without timezone
data['Date'] = data['Date'].dt.tz_localize(None).dt.date
reversed_df = data.iloc[::-1].reset_index(drop=True)
st.subheader('Date from 2024 - Present')
st.write(reversed_df.head(120))

# Stock Trends Plot
st.subheader('Stock Trends')
figure = plt.figure(figsize=(10, 5))
plt.plot(data['Close'])
st.pyplot(figure)

# Calculate correlation matrix across selected columns
corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()

# Plot correlation matrix as a heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)


# Flatten the multi-level columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


# Define required columns for the candlestick chart
required_columns = ['Open', 'High', 'Low', 'Close']

# Ensure required columns are present
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
else:
    # Drop rows with missing values in the required columns
    data_clean = data.dropna(subset=required_columns)

    # Create candlestick chart
    fig = go.Figure(data=go.Candlestick(
        x=data_clean.index,
        open=data_clean['Open'],
        high=data_clean['High'],
        low=data_clean['Low'],
        close=data_clean['Close']
    ))

    # Update layout for the candlestick chart
    fig.update_layout(
        title=f'Stock Price Candlestick Chart for {company}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    # Display the candlestick chart in Streamlit
    st.plotly_chart(fig)


# Calculate moving average
data['MA'] = data['Close'].rolling(window=20).mean()

# Plot moving average
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(data.index, data['MA'], label='Moving Average', color='orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price with Moving Average')
plt.legend()
st.pyplot(plt)


# Assuming data is already loaded and contains a 'Close' column
n_days = st.slider('Days of prediction:', 7,60)
period = n_days

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_days} days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


# User input for multiple stock symbols
symbols = st.text_input('Enter Stock Symbols (comma-separated)', 'AAPL,GOOGL,MSFT')

# Split symbols and retrieve data for each stock
stock_symbols = [symbol.strip() for symbol in symbols.split(',')]
stock_data = yf.download(stock_symbols, start=start, end=stop)
stock_data.reset_index(inplace=True)

# Convert the 'Date' column to date without timezone
stock_data['Date'] = stock_data['Date'].dt.tz_localize(None).dt.date

# Plot stock prices for multiple stocks
for symbol in stock_symbols:
    plt.plot(stock_data.index, stock_data['Close'][symbol], label=symbol)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices Comparison')
plt.legend()
st.pyplot(plt)

# Streamlit app setup
st.title("Personalized Watchlists")

# Watchlist management
watchlist = []

# Add stock to watchlist
st.header("Add Stock to Watchlist")

stock_symbol = st.text_input("Stock Symbol", value="AAPL")

if st.button("Add to Watchlist"):
    watchlist.append(stock_symbol)
    st.success(f"Added {stock_symbol} to your watchlist!")

# Display watchlist
st.header("Your Watchlist")

if len(watchlist) > 0:
    for symbol in watchlist:
        st.write(symbol)
else:
    st.info("Your watchlist is empty.")

# Historical Returns Calculation
returns = data['Close'].pct_change()
returns_cumulative = (returns + 1).cumprod() - 1

# Historical Returns Plot
if st.button('Historical Returns'):
    plt.plot(returns_cumulative)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Historical Cumulative Returns')
    st.pyplot()

# Calculate RSI
data['RSI'] = momentum.RSIIndicator(data['Close']).rsi()

# Plot RSI
# Relative Strength Index (RSI)
delta = data['Close'].diff()
gain = delta.mask(delta < 0, 0)
loss = -delta.mask(delta > 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
rsi = 100 - (100 / (1 + rs))

if st.button('Relative Strength Index (RSI)'):
    fig, ax = plt.subplots()
    ax.plot(data.index, rsi)
    ax.axhline(30, color='red', linestyle='--')
    ax.axhline(70, color='red', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.set_title('Relative Strength Index (RSI)')
    st.pyplot(fig)

# Bollinger Bands

rolling_mean = data['Close'].rolling(window=20).mean()
rolling_std = data['Close'].rolling(window=20).std()
upper_band = rolling_mean + (2 * rolling_std)
lower_band = rolling_mean - (2 * rolling_std)
if st.button('Bollinger Bands'):
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, upper_band, label='Upper Band')
    ax.plot(data.index, lower_band, label='Lower Band')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Bollinger Bands')
    ax.legend()
    st.pyplot(fig)

stock_data = pd.DataFrame({
    'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'],
    'Name': ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Facebook Inc.'],
    'MarketCap': [2265, 1869, 1779, 1697, 1048],
    'Industry': ['Technology', 'Technology', 'Technology', 'Retail', 'Technology'],
    'PE_Ratio': [31.98, 29.32, 38.78, 66.45, 24.59]
})


st.title("Advanced Stock Screener")

industry = st.selectbox("Industry", sorted(stock_data['Industry'].unique()))
min_market_cap = st.number_input("Minimum Market Cap", min_value=0)
max_pe_ratio = st.number_input("Maximum P/E Ratio", min_value=0)

filtered_data = stock_data[
    (stock_data['Industry'] == industry) &
    (stock_data['MarketCap'] >= min_market_cap) &
    (stock_data['PE_Ratio'] <= max_pe_ratio)
]

if len(filtered_data) > 0:
    st.dataframe(filtered_data)
else:
    st.info("No stocks match the selected criteria.")
    