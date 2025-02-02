import numpy as np
from scipy.optimize import minimize
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from config.settings import DATA_PATH, RISK_FREE_RATE
from data.data_loader import load_data
from models.portfolio import PortfolioRiskAssessment
from utils.logger import setup_logger
from data.twitter_sentiment import get_user_profile
from financial_metrics import calculate_revenue_and_profit_margins, calculate_roi, calculate_eps

# Set up logging
setup_logger()

def main():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    today = datetime.today().strftime('%Y-%m-%d')
    six_months_ago = (datetime.today() - timedelta(days=6*30)).strftime('%Y-%m-%d')
    train_start = six_months_ago
    train_end = today
    test_start = six_months_ago
    test_end = today

    # Load data
    prices = load_data(tickers, train_start, train_end)
    
    # Initialize PortfolioRiskAssessment
    portfolio = PortfolioRiskAssessment(prices, RISK_FREE_RATE)
    
    # Predict returns
    predicted_returns = portfolio.predict_returns()
    
    # Adjust returns with volatility and VIX
    adjusted_returns = portfolio.adjust_returns_with_volatility_and_vix(predicted_returns)
    
    # Fetch user profile information (skip if not accessible)
    try:
        user_profile = get_user_profile()
        print("User Profile:", user_profile)
    except Exception as e:
        logging.error(f"Error fetching user profile: {e}")
        user_profile = None
        print("User Profile: None")
    
    # Optimize portfolio based on adjusted returns
    num_assets = len(adjusted_returns)
    initial_weights = np.ones(num_assets) / num_assets  # Ensure initial weights have the correct shape
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(portfolio.portfolio_risk, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    
    # Calculate Sharpe Ratio and Sortino Ratio
    sharpe_ratio = portfolio.calculate_sharpe_ratio(optimal_weights)
    sortino_ratio = portfolio.calculate_sortino_ratio(optimal_weights)
    
    # Print results
    print("Optimal Weights:", optimal_weights)
    print("Adjusted Returns:", adjusted_returns)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Sortino Ratio:", sortino_ratio)
    
    # Visualize the results
    visualize_results(tickers, optimal_weights, adjusted_returns)
    
    # Black-Scholes option pricing
    S = prices.iloc[-1]  # Current stock prices
    K = S * 1.05  # Strike price 5% above current price
    T = 1  # 1 year to maturity
    sigma = portfolio.volatility
    r = RISK_FREE_RATE
    call_prices = [portfolio.black_scholes(S[ticker], K[ticker], T, r, sigma[ticker], 'call') for ticker in tickers]
    put_prices = [portfolio.black_scholes(S[ticker], K[ticker], T, r, sigma[ticker], 'put') for ticker in tickers]
    print("Call Option Prices:", call_prices)
    print("Put Option Prices:", put_prices)
    
    # Monte Carlo simulations
    num_simulations = 1000
    num_steps = 252
    for ticker in tickers:
        price_paths = portfolio.monte_carlo_simulation(S[ticker], T, r, sigma[ticker], num_simulations, num_steps)
        visualize_monte_carlo_simulation(ticker, price_paths)
    
    # Financial metrics
    financial_data = pd.DataFrame({
        'Revenue': [100000, 150000, 200000],
        'Profit': [10000, 20000, 30000],
        'Net Income': [5000, 10000, 15000],
        'Shares Outstanding': [1000, 1000, 1000]
    })
    investment_data = pd.DataFrame({
        'Investment': [50000, 60000, 70000],
        'Return': [10000, 12000, 14000]
    })
    
    revenue, profit_margin = calculate_revenue_and_profit_margins(financial_data)
    roi = calculate_roi(investment_data)
    eps = calculate_eps(financial_data)
    
    print("Revenue:", revenue)
    print("Profit Margin:", profit_margin)
    print("ROI:", roi)
    print("EPS:", eps)

def visualize_results(tickers, optimal_weights, adjusted_returns):
    # Plot optimal weights
    plt.figure(figsize=(10, 5))
    plt.bar(tickers, optimal_weights, color='blue')
    plt.xlabel('Tickers')
    plt.ylabel('Optimal Weights')
    plt.title('Optimal Weights for Each Ticker')
    plt.show()

    # Plot adjusted returns
    plt.figure(figsize=(10, 5))
    plt.bar(tickers, adjusted_returns, color='green')
    plt.xlabel('Tickers')
    plt.ylabel('Adjusted Returns')
    plt.title('Adjusted Returns for Each Ticker')
    plt.show()

def visualize_monte_carlo_simulation(ticker, price_paths):
    plt.figure(figsize=(10, 5))
    plt.plot(price_paths)
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.title(f'Monte Carlo Simulation for {ticker}')
    plt.show()

if __name__ == "__main__":
    main()