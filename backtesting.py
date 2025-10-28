import logging
import numpy as np
import os
import sys
from scipy.optimize import minimize  # Import the minimize function

# Add the current directory to the Python path,
sys.path.append(os.getcwd())

from copilot_code import PortfolioRiskAssessment

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def backtest_portfolio(tickers, train_start, train_end, test_start, test_end, risk_free_rate=0.01):
    """
    Perform backtesting on the portfolio.

    Parameters:
    tickers (list): List of stock tickers.
    train_start (str): Start date for training data.
    train_end (str): End date for training data.
    test_start (str): Start date for testing data.
    test_end (str): End date for testing data.
    risk_free_rate (float): Risk-free rate for calculating Sharpe and Sortino ratios.

    Returns:
    tuple: Test portfolio return, variance, standard deviation, Sharpe ratio, and Sortino ratio.
    """
    # Initialize the PortfolioRiskAssessment class with training data
    portfolio = PortfolioRiskAssessment(tickers, train_start, train_end, risk_free_rate)
    
    # Predict future returns using a linear regression model
    predicted_returns = portfolio.predict_returns()
    
    # Define the objective function for optimization
    def objective_function(weights):
        return portfolio.portfolio_risk(weights, portfolio.cov_matrix)
    
    # Optimize the portfolio based on predicted returns
    num_assets = len(predicted_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Ensure the sum of weights is 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # Weights should be between 0 and 1
    result = minimize(objective_function, num_assets*[1./num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    
    # Load testing data
    test_prices = portfolio.load_data(tickers, test_start, test_end)
    test_returns = portfolio.calculate_returns(test_prices)
    
    # Apply optimized weights to testing data
    test_portfolio_return = np.dot(optimal_weights, test_returns.mean())
    test_portfolio_variance = np.dot(optimal_weights.T, np.dot(test_returns.cov(), optimal_weights))
    test_portfolio_std_dev = np.sqrt(test_portfolio_variance)
    test_sharpe_ratio = (test_portfolio_return - risk_free_rate) / test_portfolio_std_dev
    test_sortino_ratio = (test_portfolio_return - risk_free_rate) / portfolio.downside_risk(optimal_weights)
    
    logging.info("Backtesting completed successfully.")
    return test_portfolio_return, test_portfolio_variance, test_portfolio_std_dev, test_sharpe_ratio, test_sortino_ratio

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]  # List of stock tickers
    train_start = "2020-01-01"            # Start date for training data
    train_end = "2022-01-01"              # End date for training data
    test_start = "2022-01-02"             # Start date for testing data
    test_end = "2023-01-01"               # End date for testing data

    # Perform backtesting
    test_portfolio_return, test_portfolio_variance, test_portfolio_std_dev, test_sharpe_ratio, test_sortino_ratio = backtest_portfolio(
        tickers, train_start, train_end, test_start, test_end)

    # Print backtesting results
    print("Backtesting Results:")
    print("Test Portfolio Return:", test_portfolio_return)
    print("Test Portfolio Variance:", test_portfolio_variance)
    print("Test Portfolio Standard Deviation:", test_portfolio_std_dev)
    print("Test Sharpe Ratio:", test_sharpe_ratio)
    print("Test Sortino Ratio:", test_sortino_ratio)
