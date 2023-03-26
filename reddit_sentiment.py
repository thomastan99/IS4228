from statistics import mean
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer




def reddit_sentiment(stock_symbol):
    client_id = 'uwjwEuIgxWDsluEe4bgAew' 
    client_secret = 'x9j8g9e5MGGKvxJqnCmfQUioxS3pKA' 
    user_agent = 'crypto3107' 
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    subreddit = reddit.subreddit('wallstreetbets')  # Change subreddit as needed
    search_results = subreddit.search(f'{stock_symbol} OR {stock_symbol.upper()}')
    sentiment_arr = []
    for post in search_results:
        sentiment_arr.append(get_sentiment_score(post.title))
    average_score = mean(sentiment_arr)
    return round(average_score,2)

def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

# stock_symbol = input("Enter a stock symbol: ")
# print(reddit_sentiment(stock_symbol))
