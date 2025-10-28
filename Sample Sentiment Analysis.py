import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

def fetch_news(ticker):
    """\
    
    Fetch recent news headlines for a given ticker symbol.
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch news for ticker: {ticker}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all('h3', class_='Mb(5px)')
    if not headlines:
        print(f"No headlines found for ticker: {ticker}")
        return []
    
    return [headline.text for headline in headlines]

def analyze_sentiment(headlines):
    """
    Analyze sentiment of news headlines using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    return sentiments

def sentiment_analysis(tickers):
    """
    Perform sentiment analysis on a list of ticker symbols.
    """
    sentiment_scores = {}
    for ticker in tickers:
        headlines = fetch_news(ticker)
        if not headlines:
            print(f"No headlines found for ticker: {ticker}")
            continue
        sentiments = analyze_sentiment(headlines)
        sentiment_scores[ticker] = {
            'headlines': headlines,
            'sentiment_scores': sentiments,
            'average_sentiment': sum(sentiments) / len(sentiments) if sentiments else 0
        }
    return sentiment_scores

# Example usage
if __name__ == "__main__":
    # List of portfolio assets (ticker symbols)
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Perform sentiment analysis
    sentiment_scores = sentiment_analysis(tickers)

    # Display the results
    for ticker, data in sentiment_scores.items():
        print(f"Ticker: {ticker}")
        print(f"Average Sentiment: {data['average_sentiment']}")
        print("Headlines and Sentiment Scores:")
        for headline, score in zip(data['headlines'], data['sentiment_scores']):
            print(f"  {headline} - Sentiment Score: {score}")
        print()
