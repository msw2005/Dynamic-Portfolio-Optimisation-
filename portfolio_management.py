import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.news import News
import requests

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to fetch stock data from Alpha Vantage API
def fetch_stock_data(api_key, ticker, start_date, end_date):
    ts = TimeSeries(key=api_key, output_format='pandas')
    stock_data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
    stock_data = stock_data[start_date:end_date]
    return stock_data

# Function to fetch news data from Alpha Vantage API
def fetch_news_data(api_key, ticker):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    articles = news_data.get('feed', [])
    return pd.DataFrame(articles)

# Function to analyze sentiment
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score['compound']

# Function to preprocess data
def preprocess_data(stock_data, news_data):
    # Example preprocessing steps
    stock_data['Returns'] = stock_data['5. adjusted close'].pct_change()
    news_data['Sentiment'] = news_data['summary'].apply(analyze_sentiment)
    
    # Reset indices to ensure compatibility for merging
    stock_data.reset_index(inplace=True)
    news_data.reset_index(inplace=True)
    
    return stock_data, news_data

# Function to train models
def train_models(stock_data, news_data):
    # Combine stock and sentiment data
    combined_data = pd.merge(stock_data, news_data, left_index=True, right_index=True)
    X = combined_data[['Sentiment', '6. volume']]
    y = combined_data['Returns'] > 0

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM model
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    # Train Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    return svm_model, rf_model, svm_accuracy, rf_accuracy

# Function to validate models
def validate_models(models, X, y):
    for model in models:
        scores = cross_val_score(model, X, y, cv=5)
        print(f"Model: {model.__class__.__name__}, Accuracy: {scores.mean()}")

# Function to integrate models into a dashboard (placeholder)
def integrate_dashboard(models):
    print("Integrating models into the dashboard...")

# Main workflow
def main():
    api_key = '21LQWMXRT8MY0356'
    ticker = 'AAPL'
    
    # Data collection
    stock_data = fetch_stock_data(api_key, ticker, '2020-01-01', '2021-01-01')
    news_data = fetch_news_data(api_key, ticker)

    # Data preprocessing
    stock_data, news_data = preprocess_data(stock_data, news_data)

    # Model training
    svm_model, rf_model, svm_accuracy, rf_accuracy = train_models(stock_data, news_data)
    print(f"SVM Accuracy: {svm_accuracy}, Random Forest Accuracy: {rf_accuracy}")

    # Model validation
    X = stock_data[['Sentiment', '6. volume']]
    y = stock_data['Returns'] > 0
    validate_models([svm_model, rf_model], X, y)

    # Integration
    integrate_dashboard([svm_model, rf_model])

if __name__ == "__main__":
    main()