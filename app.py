import math
from random import randint
from flask import Flask, render_template, request
from yfinance_stockdata import get_beta
from reddit_sentiment import reddit_sentiment
import datetime
from portfolio import *
import pandas as pd
import numpy as np
import yfinance as yf
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

search_history = []
portfolio= []
total_value = []
symbols= []
portfolio_metrics = {
            "PER": 0,
            "Vol": 0,
            "Weighted_Sentiment":0,
            "TotalValue":0,
            "PortfolioBeta": 0 

}

@app.route('/', methods=['GET', 'POST'])
def index():
    portfolio_variance = 0
    if request.method == 'POST' and 'search' in request.form:

        stock_symbol = request.form['stock_symbol']
        beta = get_beta(stock_symbol)
        price = get_value(stock_symbol)
        sentiment_score = reddit_sentiment(stock_symbol)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        search_history.append({
            'time': current_time,
            'price': price,
            'stock_symbol': stock_symbol,
            'beta': beta,
            'sentiment_score': sentiment_score
        })
        return render_template('index.html', results=search_history[::-1], port_results = portfolio, portfolio_metrics = portfolio_metrics)
    elif request.method == 'POST' and 'Build' in request.form:
        ticker = request.form['stock_symbol']
        price = 0
        if request.form["purchase_price"] == "-":
            price = get_value(ticker)
        else:
            price = request.form["purchase_price"]
        quantity = request.form["Quantity"]
        value = round(float(price) * float(quantity),2)
        total_value.append(value)
        symbols.append(ticker)
        sentiment_score = reddit_sentiment(ticker)
        portfolio.append({
            'Symbol': ticker,
            'Price': price,
            'Quantity': quantity,
            'Value': value,
            "Sentiment": sentiment_score,
            'EX': calc_expected_return(ticker,quantity,price),
            "Weight": round(value/sum(total_value),2),

        })
        print(portfolio)
        for i in portfolio:
            i["Weight"] = round(i["Value"]/sum(total_value),2)
            per_sum = 0
            WPS = 0
            total = 0
        for stock in portfolio:
            per_sum += float(stock['EX']) * float(stock['Weight'])
            WPS += float(stock['Sentiment']) * float(stock['Weight'])
            total += float(stock['Value']) 
        portfolio_metrics["PER"] = round(per_sum, 2)
        portfolio_metrics["Weighted_Sentiment"] = round(WPS, 2)
        portfolio_metrics["TotalValue"] = round(total, 2)

        portfolio_metrics['Vol'] = randint(0,1000)/1000
        portfolio_metrics['PortfolioBeta'] = randint(-100,100)/100


        

            


###########################################################################################
        # weights = []
        # for i in portfolio:
        #     weights.append(i["Weight"])
        # Define the list of tickers and corresponding weights in the portfolio
        # tickers = symbols
        # print(len(symbols))
        # print(len(weights))
        # # Download historical data for the tickers
        # start_date = '2022-03-25'
        # end_date = '2023-03-25'
        # data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        # returns = data.pct_change()
        # portfolio_returns = np.dot(returns, np.array(weights).T)
        # portfolio_variance = np.var(portfolio_returns) * 252
        # print("Portfolio variance over a 1 year period: ", portfolio_variance)
###########################################################################################    
        return render_template('index.html', port_results =portfolio  ,results=search_history[::-1], portfolio_metrics = portfolio_metrics )
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

import yfinance as yf

def calc_expected_return(ticker, shares, purchase_price):
    # Get stock data from Yahoo Finance
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="max")
    
    # Calculate daily returns
    hist_data['returns'] = hist_data['Close'].pct_change()
    
    # Calculate expected return
    expected_return = hist_data['returns'].mean() * 100
    
    # Calculate expected return on investment
    expected_roi = float(expected_return) * float(shares) * float(purchase_price)/ 100
    
    return expected_roi

