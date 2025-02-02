import pandas as pd
from datetime import datetime, timedelta
from data.alpha_vantage_loader import load_data
from strategies.moving_average_crossover import execute_moving_average_crossover
from strategies.rsi_strategy import execute_rsi_strategy
from utils.logger import setup_logger
from utils.visualizer import plot_signals

def main():
    logger = setup_logger()
    tickers = ["AAPL", "MSFT", "GOOGL"]
    today = datetime.today().strftime('%Y-%m-%d')
    one_month_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    start_date = one_month_ago
    end_date = today

    # Load data
    prices = load_data(tickers, start_date, end_date, interval='1min', outputsize='compact', api_key='P5OLYM2NB92FRIG')
    
    # Ensure the data structure is correct
    for ticker in tickers:
        if 'Close' not in prices[ticker].columns:
            raise KeyError(f"'Close' column not found in data for {ticker}")

    # Execute algorithmic strategies
    for ticker in tickers:
        ticker_prices = prices[ticker]
        
        # Moving Average Crossover Strategy
        buy_signals_mac, sell_signals_mac = execute_moving_average_crossover(ticker_prices)
        logger.info(f"Buy signals for {ticker} (MAC): {buy_signals_mac.index}")
        logger.info(f"Sell signals for {ticker} (MAC): {sell_signals_mac.index}")
        plot_signals(ticker_prices, buy_signals_mac, sell_signals_mac, title=f'{ticker} - Moving Average Crossover')

        # RSI Strategy
        buy_signals_rsi, sell_signals_rsi = execute_rsi_strategy(ticker_prices)
        logger.info(f"Buy signals for {ticker} (RSI): {buy_signals_rsi.index}")
        logger.info(f"Sell signals for {ticker} (RSI): {sell_signals_rsi.index}")
        plot_signals(ticker_prices, buy_signals_rsi, sell_signals_rsi, title=f'{ticker} - RSI Strategy')

if __name__ == "__main__":
    main()