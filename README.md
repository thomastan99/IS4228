# IS4228
Our group took on a slightly more practical approach to our research where we attemmpted to create our own trading dashboard using certain formulas taught in IS4228 as well 
as sentiment analysis by scraping reddit data

# Dashboard - Stock Search Usage
Time Searched - Time that the stock was searched at (Indicates the time that the price is valid at)
Price - The price of the stock
Stock Symbol -  The stock's ticker
Beta - The stock Beta, with respect to S&P500
Reddit Sentiment Score - The sentiment score based on scraping reddit threads in the Wall Street Bets page, using NLP to obtain the aggregated sentiment score

# Dashboard - Portfolio Builder Usage
In the Enter Purchase Price field, input "-" to use the current market price from yfinance
Stock Symbol -  The stock's ticker
Price - The price of the stock
Quanity - Quantity of Stock
Value - Price * Quantity
Weight in portfolio - Value of the particular stock / Total Value
Expected Returns - Expected Returns


# Dashboard - Portfolio Metrics
Volatility - Portfolio Volatility [To be Implemented]
Expected Return - Portfolio Expected Return
Weighted Portfolio Sentiment - Based on the sentiment score of each stock and weighted by the stock weights
Total Portfolio Value - Sum of the total Value
Portfolio Beta - Portfolio Beta with respect to S&P500 [To be Implemented]

# Dashboard - Price Prediction Model
Refer to the model.ipynb for the model details

# Comments / Future Improvements
In light of the time constraint, there we several features such as the Volatility and Portfolio Beta which were not implemented yet
Additionally, other improvements couold also be increasing the number of sources that the sentiment score is scraped from and weighed, the Portfolio Bulder component could also include connection to a sqllite DB to retain the data



