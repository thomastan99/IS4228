from datetime import datetime, timedelta  
import datetime as dt
from dateutil import tz
import praw
import pandas as pd
import os
from io import StringIO
from yahoofinancials import YahooFinancials
from statistics import mean
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import yfinance as yf
from yahooquery import Ticker
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import talib 
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler,MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor

def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']


def reddit_sentiment(stock_symbol, start_date="2015-01-01", end_date="2022-03-21"):
    client_id = 'uwjwEuIgxWDsluEe4bgAew' 
    client_secret = 'x9j8g9e5MGGKvxJqnCmfQUioxS3pKA' 
    user_agent = 'crypto3107' 
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    # Convert the datetime objects to Unix timestamps
    timestamp_start = int(start_date.timestamp())
    timestamp_end = int(end_date.timestamp())

    # Define the search query
    search_query = f'{stock_symbol} OR {stock_symbol.upper()}'

    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    search_results = reddit.subreddit('wallstreetbets').search(search_query, time_filter='all')
    search_results_filtered = [post for post in search_results if timestamp_start <= post.created_utc <= timestamp_end]



    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('Singapore')
    sentiment_arr = []
    date_arr = []
    for post in search_results_filtered:
        sentiment_arr.append(get_sentiment_score(post.selftext))
        post_time = dt.datetime.fromtimestamp(post.created_utc)
        new_datetime = str(post_time)
        utc = datetime.strptime(new_datetime, '%Y-%m-%d %H:%M:%S')
        utc = utc.replace(tzinfo=from_zone)
        local = utc.astimezone(to_zone)
        local_dt = local.strftime('%Y-%m-%d')
        date_arr.append(local_dt)
    data_dic = {'Date': date_arr, 'sentiment':sentiment_arr}
    df = pd.DataFrame(data_dic)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.date
    grouped = df.groupby('Date').mean()
    return grouped.reset_index()






def get_stock(ticker, start_date="2015-01-01", end_date="2022-03-21"):
    """
    Retrieves historical stock prices for the given ticker symbol and date range,
    as well as the historical prices of the S&P 500 index.
    Calculates the beta of the stock using a linear regression model.
    Returns the beta value.
    """
    # Get historical prices for the stock and the S&P 500 index
    stock = yf.Ticker(ticker)
    historical_prices = stock.history(start=start_date, end=end_date).reset_index()
    historical_prices['Date'] = pd.to_datetime(historical_prices['Date']).dt.tz_localize(None).dt.date
    historical_prices= historical_prices.reset_index()
#     tickers = Ticker(ticker)
#     types = [
#     'MarketCap'
# ]
#     fundamental = tickers.aapl.balance_sheet(frequency='q', trailing) #.reset_index().drop(['symbol'], axis=1)

#     print(fundamental)

#     # fundamental.asOfDate = pd.to_datetime(fundamental.asOfDate,unit='D')
#     # fundamental = fundamental.set_index('asOfDate')
#     # fundamental = fundamental.resample('D').last().ffill().reset_index().rename({'asOfDate':'Date'}, axis=1)
#     # fundamental['Date'] = pd.to_datetime(fundamental['Date']).dt.tz_localize(None).dt.date
#     # df = pd.merge(historical_prices, fundamental, on='Date', how='left')
    
#     # df.drop(['asOfDate'], axis=1, inplace=True)

#     # # Drop rows with missing values (if any)
#     # df = df.dropna()
#     # print(df)
    return historical_prices


def prepare_dataset(data1, data2):
    df = pd.merge(data1, data2, on='Date', how='left')
    df['next_day_closing_price'] = df['Close'].shift(-1)
    df.replace(to_replace='null', value=np.nan,inplace=True)
    df.drop(df.tail(1).index,inplace=True)
    return df



def feature_engineering(df):
    # https://github.com/rohansawant7978/bitcoin-price-forecasting/blob/main/03_Feature_Engineering.ipynb
    feature_list = [i for i in list(df.columns) if i not in ['Date','next_day_closing_price']]
    
    def feature_smoothening(df,feature_name,smoothening_type,smoothening_range=[3,7,30]):
        if smoothening_type == 'sma':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.sma(df[feature_name],j) 

        elif smoothening_type == 'stdev':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.stdev(df[feature_name],j)
        
        elif smoothening_type == 'ema':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.ema(df[feature_name],j)

        elif smoothening_type == 'wma':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.wma(df[feature_name],j)

        elif smoothening_type == 'rsi':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.rsi(df[feature_name],j)

        elif smoothening_type == 'dema':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.dema(df[feature_name],j) 

        elif smoothening_type == 'roc':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.roc(df[feature_name],j)  

        elif smoothening_type == 'tema':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.tema(df[feature_name],j) 
        elif smoothening_type == 'bband_lower':
            for j in smoothening_range:
                bband_df = ta.bbands(df[feature_name],j)
                df[f'{smoothening_type}{j} {feature_name}'] = bband_df[f'BBL_{j}_2.0']

        elif smoothening_type == 'bband_upper':
            for j in smoothening_range:
                bband_df = ta.bbands(df[feature_name],j)
                df[f'{smoothening_type}{j} {feature_name}'] = bband_df[f'BBU_{j}_2.0']
            

        elif smoothening_type == 'macd':
            macd_df = ta.macd(df[feature_name])
            df[f'{smoothening_type} hist {feature_name}'] = macd_df['MACDh_12_26_9']
            df[f'{smoothening_type} signal {feature_name}'] = macd_df['MACDs_12_26_9']
            df[f'{smoothening_type} {feature_name}'] = macd_df['MACD_12_26_9']
        elif smoothening_type == 'var':
            for j in smoothening_range:
                df[f'{smoothening_type}{j} {feature_name}'] = ta.variance(df[feature_name],j)
            
        return df


    for feature in feature_list:
        df = feature_smoothening(df,feature,'sma')
        df =feature_smoothening(df,feature,'wma')
        df = feature_smoothening(df,feature,'ema')
        df =feature_smoothening(df,feature,'dema')
        df =feature_smoothening(df,feature,'tema')
        df =feature_smoothening(df,feature,'stdev')
        df =feature_smoothening(df,feature,'rsi')
        df =feature_smoothening(df,feature,'macd')
        df =feature_smoothening(df,feature,'roc')
        df =feature_smoothening(df,feature,'bband_lower')
        df =feature_smoothening(df,feature,'bband_upper')
        df =feature_smoothening(df,feature,'var')
  
    
    df[df.columns.difference(['sentiment'])] = df[df.columns.difference(['sentiment'])].fillna(method='bfill')
    df.loc[:, 'sentiment'] = df['sentiment'].fillna(df['sentiment'].mode(),inplace=False, downcast='infer')
    
    nan_percentages = df.isna().sum() / len(df)

    # If more than 50% of the rows are NaN, drop the column
    for column in df.columns:
        if nan_percentages[column] > 0.5:
            df = df.drop(column, axis=1)

    # Otherwise, drop the rows with NaN values
    df = df.dropna(axis=0)

    return df




def feature_selection(df):
    X = df.drop(['Date','next_day_closing_price'],axis=1)
    scaler = RobustScaler()
    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.fit_transform(X[X.columns])

    scaler = MinMaxScaler()
    X_scaled[X.columns] =  scaler.fit_transform(X_scaled[X.columns])
    X_scaled
    y = df['next_day_closing_price']
    rf = RandomForestRegressor(n_estimators=80,n_jobs=-1,bootstrap=True,
                            verbose=5,random_state=1)
    rf.fit(X_scaled,y)
    
    def feature_imp(df,feat_importance,n_select_features):
        feat_imp_df = pd.DataFrame(data = {"Feature Name": df.columns,"Feature Importance":feat_importance})
        feat_imp_df = feat_imp_df.sort_values("Feature Importance",ascending=False)[:n_select_features]
        return feat_imp_df,df[feat_imp_df['Feature Name']]
    
    feat_range = list(range(3,10,1))
    for i in feat_range:
        feat_imp_df,df_next_day = feature_imp(X_scaled,rf.feature_importances_,i)
        df_next_day.insert(loc=0, column='Date', value=df['Date'])
        df_next_day['next_day_closing_price'] = df['next_day_closing_price']
        
    # # check correlation matrix
    # corr_matrix = df_next_day.drop(['next_day_closing_price'], axis=1).corr()

    # # create a list of variables to drop
    # to_drop = []

    # # iterate through the correlation matrix
    # for i in range(len(corr_matrix.columns)):
    #     for j in range(i):
    #         # if the correlation is 1, add to the list of variables to drop
    #         if abs(corr_matrix.iloc[i, j]) >=0.99:
    #             to_drop.append(corr_matrix.columns[i])
    # if 'Close' in to_drop:
    #     to_drop.remove('Close')
    # # drop the variables with perfect correlation
    # df_next_day = df_next_day.drop(columns=to_drop)

    return df_next_day



stock = 'GME'
start = datetime(2019,1, 1, 0, 0, 0, 0)  
end = datetime(2021, 5, 1, 0, 0, 0, 0)  
sentiment_output = reddit_sentiment(stock, start, end)
price_output = price_output = get_stock(stock, start,end)
df= prepare_dataset(price_output, sentiment_output)
# df.to_csv('GME_data.csv',index=False)
df = pd.read_csv('GME_data.csv')
df = feature_engineering(df)
final_df = feature_selection(df)
# final_df.to_csv('final_GME_data.csv', index=False)

