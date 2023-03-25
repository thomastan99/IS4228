from yfinance_stockdata import get_beta 
from reddit_sentiment import reddit_sentiment 

while True:
    test = input()
    beta =  get_beta(test)
    reddit = reddit_sentiment(test)
    print("Beta: " + str(beta))
    print("Reddit Sentiment Score: " + str(reddit))

