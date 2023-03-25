import yfinance as yf
import datetime

'''
Sample portfolio 
portfolio = ['AAPL', 'TSLA']
noOfShares = [200, 100]
'''

def get_value(ticker, start_date=datetime.date.today(), end_date=datetime.date.today()):
  #Get the current value of a stock
  data = yf.download(ticker, start=start_date, end=end_date)
  value = data['Adj Close'][0]
  return value

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
