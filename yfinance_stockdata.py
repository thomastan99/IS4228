import yfinance as yf
from sklearn.linear_model import LinearRegression

# def get_stock_data(ticker, start_date="2015-01-01", end_date="2022-03-21"):
#     """
#     Retrieves historical stock prices for the given ticker symbol and date range.
#     Returns a Pandas dataframe with the stock prices.
#     """
#     stock_data = yf.download(ticker, start=start_date, end=end_date)
#     return stock_data

# # Define the stock ticker symbol and date range
# ticker = "AAPL" # Apple Inc. stock symbol
# start_date = "2015-01-01"
# end_date = "2022-03-21"

# # Use the yfinance library to retrieve the historical stock data
# stock_data = yf.download(ticker, start=start_date, end=end_date)

# # Print the resulting dataframe
# print(stock_data)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def get_beta(ticker, start_date="2015-01-01", end_date="2022-03-21"):
    """
    Retrieves historical stock prices for the given ticker symbol and date range,
    as well as the historical prices of the S&P 500 index.
    Calculates the beta of the stock using a linear regression model.
    Returns the beta value.
    """
    # Get historical prices for the stock and the S&P 500 index
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    index_data = yf.download("^GSPC", start=start_date, end=end_date)

    # Calculate daily returns for the stock and the index
    stock_returns = stock_data['Adj Close'].pct_change()
    index_returns = index_data['Adj Close'].pct_change()

    returns_data = pd.concat([stock_returns, index_returns], axis=1)
    returns_data.columns = [ticker, "^GSPC"]
    returns_data = returns_data.dropna()
    X = np.array(returns_data["^GSPC"]).reshape(-1, 1)
    y = np.array(returns_data[ticker]).reshape(-1, 1)
    model = LinearRegression().fit(X, y)

    # Return the slope coefficient of the regression model as the beta value
    beta = model.coef_[0][0]
    print(f"The beta of symbol {ticker} is {beta}" )
    return beta

while True:
    get_beta(input())
