import numpy as np
import pandas as pd

def moving_average_crossover(prices, short_window=40, long_window=100):
    """
    Moving Average Crossover Strategy.
    prices: DataFrame containing stock prices with a 'Close' column
    short_window: Window for the short moving average
    long_window: Window for the long moving average
    """
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0

    # Create short simple moving average
    signals['short_mavg'] = prices['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average
    signals['long_mavg'] = prices['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals.loc[signals.index[short_window:], 'signal'] = np.where(
        signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0
    )

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

def rsi(prices, window=14):
    """
    Relative Strength Index (RSI) Strategy.
    prices: DataFrame containing stock prices with a 'Close' column
    window: Window for calculating RSI
    """
    delta = prices['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def execute_strategy(prices):
    """
    Execute the algorithmic trading strategy.
    prices: DataFrame containing stock prices with a 'Close' column
    """
    signals = moving_average_crossover(prices)
    rsi_values = rsi(prices)

    # Example strategy: Buy when RSI < 30 and short moving average crosses above long moving average
    buy_signals = (rsi_values < 30) & (signals['positions'] == 1.0)
    sell_signals = (rsi_values > 70) & (signals['positions'] == -1.0)

    # Logging for debugging
    print("RSI Values:")
    print(rsi_values)
    print("Signals:")
    print(signals)
    print("Buy Signals:")
    print(buy_signals[buy_signals].index)
    print("Sell Signals:")
    print(sell_signals[sell_signals].index)

    return buy_signals, sell_signals