from flask import Flask, render_template, request
from yfinance_stockdata import get_beta
from reddit_sentiment import reddit_sentiment
from datetime import datetime

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

search_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        beta = get_beta(stock_symbol)
        sentiment_score = reddit_sentiment(stock_symbol)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        search_history.append({
            'time': current_time,
            'stock_symbol': stock_symbol,
            'beta': beta,
            'sentiment_score': sentiment_score
        })
        return render_template('index.html', results=search_history[::-1])
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
