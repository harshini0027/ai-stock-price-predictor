import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ticker = "AAPL"

data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

data = data[['Close']]
data['Prediction'] = data['Close'].shift(-10)

X = data[['Close']][:-10]
y = data['Prediction'][:-10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

last_close = data[['Close']].tail(1)
future_price = model.predict(last_close)

print("Predicted next price:", future_price[0])


plt.plot(data['Close'])
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()