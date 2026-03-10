import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("AI Stock Price Predictor")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

st.subheader("Stock Data")
st.write(data.tail())

data = data[['Close']]
data['Prediction'] = data['Close'].shift(-10)

X = data[['Close']][:-10]
y = data['Prediction'][:-10]

model = LinearRegression()
model.fit(X, y)

last_close = data[['Close']].tail(1)
future_price = model.predict(last_close)

st.subheader("Predicted Price")
st.write(f"Next predicted price: {future_price[0]}")

st.subheader("Stock Price Chart")

fig, ax = plt.subplots()
ax.plot(data['Close'])
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.set_title(f"{ticker} Stock Price")

st.pyplot(fig)