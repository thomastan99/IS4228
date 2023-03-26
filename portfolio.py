import yfinance as yf
import datetime
import pandas as pd
import numpy as np

'''
Sample portfolio 
portfolio = ['AAPL', 'TSLA']
noOfShares = [200, 100]
'''

def get_value(ticker, start_date='2023-01-01', end_date=datetime.date.today()):
  #Get the current value of a stock
  data = yf.download(ticker, start=start_date, end=end_date)
  value = data['Adj Close'][0]
  return round(value,2)

def get_totalValue(stock_list, noOfShare):
  #Get total value of the portfolio
  total_value = 0
  for i in range(len(stock_list)):
    total_value += get_value(stock_list[i]) * noOfShare[i]
  return total_value

def get_weight(stock_list, noOfShare):
  #Get individual weight of each stock
  weights = {}
  total_value = get_totalValue(stock_list, noOfShare)
  for i in range(len(stock_list)):
    weights[stock_list[i]] = (get_value(stock_list[i]) * noOfShare[i])/total_value
  return weights

def build_portfolio(stock_list, noOfShares):
  #Creates a dictionary with all the stocks in the portfolio and their individual value and weight
  portfolio = {}
  for i in range(len(stock_list)):
    sub_dict = {'noOfShares': noOfShares[i], 'value': get_value(stock_list[i]), 'weight': get_weight(stock_list, noOfShares)[stock_list[i]]}
    portfolio[stock_list[i]] = sub_dict   
  return portfolio

def get_stockReturn(ticker, start_date="2015-01-01", end_date="2022-03-21"):
  #Get daily returns of stock
  data = yf.download(ticker, start=start_date, end=end_date)
  returns = data["Adj Close"].pct_change()
  return returns

def get_expectedStockReturn(ticker):
  #get the expected return of a stock
  return get_stockReturn(ticker).mean()

def get_expectedPortfolioReturn(portfolio):
  #Get the expected return of the portfolio
  tickers = list(portfolio.keys())
  weights = []
  for ticker in portfolio.keys():
    weights.append(portfolio[ticker]['weight'])
  expected_return = 0
  for i in range(len(tickers)):
    expected_return += get_expectedStockReturn(tickers[i]) * weights[i]
  return expected_return

def get_portfolioVariance(portfolio):
  #Get the variance of the portfolio
  weights = []
  for ticker in portfolio.keys():
    weights.append(portfolio[ticker]['weight'])
  returns = pd.concat([get_stockReturn(ticker) for ticker in portfolio.keys()], axis=1)
  cov_matrix = returns.cov()
  variance = np.dot(weights, np.dot(cov_matrix, weights))
  return variance

def get_portfolioVolatility(portfolio):
  #Get the volatility of the portfolio
  volatility = get_portfolioVariance(portfolio) ** 0.5
  return volatility

def get_sharpe_ratio(portfolio):
  #Assume risk-free rate is equals to current yield of the 10-year Treasury note as it is a govt bond, hence a risk-free asset
  rf = get_value('^TNX') 
  sharpe_ratio = (get_expectedPortfolioReturn(portfolio) - rf)/get_portfolioVolatility(portfolio)
  return sharpe_ratio