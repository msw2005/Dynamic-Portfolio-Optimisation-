import numpy as np
import pandas as pd
import logging
import yfinance as yf
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PortfolioRiskAssessment:
    """
    
    A class to perform portfolio risk assessment and optimization.

    Attributes:
        prices (pd.DataFrame): Adjusted closing prices.
        returns (pd.DataFrame): Daily returns.
        cov_matrix (pd.DataFrame): Covariance matrix.
        risk_free_rate (float): Risk-free rate.

    Methods:
        __init__(tickers, start_date, end_date, risk_free_rate=0.01): Initializes the class.
        load_data(tickers, start_date, end_date): Loads adjusted closing prices.
        calculate_returns(prices): Calculates daily returns.
        calculate_covariance_matrix(returns): Calculates covariance matrix.
        portfolio_risk(weights, cov_matrix): Calculates portfolio risk.
        optimize_portfolio(): Optimizes the portfolio.
        assess_portfolio(weights): Assesses portfolio performance.
        downside_risk(weights): Calculates downside risk.
        monte_carlo_simulation(num_simulations=10000): Performs Monte Carlo simulation.
        predict_returns(): Predicts future returns using a linear regression model.
    """
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.01):
        self.prices = self.load_data(tickers, start_date, end_date)
        self.returns = self.calculate_returns(self.prices)
        self.cov_matrix = self.calculate_covariance_matrix(self.returns)
        self.risk_free_rate = risk_free_rate

    def load_data(self, tickers, start_date, end_date):
        try:
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            logging.info("Data loaded successfully from Yahoo Finance.")
            return data
        except Exception as e:
            logging.error(f"Error loading data from Yahoo Finance: {e}")
            raise

    def calculate_returns(self, prices):
        returns = prices.pct_change().dropna()
        logging.info("Returns calculated successfully.")
        return returns

    def calculate_covariance_matrix(self, returns):
        cov_matrix = returns.cov()
        logging.info("Covariance matrix calculated successfully.")
        return cov_matrix

    def portfolio_risk(self, weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def optimize_portfolio(self):
        num_assets = len(self.cov_matrix)
        args = (self.cov_matrix,)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        result = minimize(self.portfolio_risk, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def assess_portfolio(self, weights):
        mean_returns = self.returns.mean()
        i0 = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_std_dev = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std_dev
        sortino_ratio = (portfolio_return - self.risk_free_rate) / self.downside_risk(weights)
        logging.info("Portfolio assessed successfully.")
        return portfolio_return, portfolio_variance, portfolio_std_dev, sharpe_ratio, sortino_ratio

    def downside_risk(self, weights):
        portfolio_return = np.dot(weights, self.returns.mean())
        downside_returns = self.returns[self.returns < 0]
        downside_std_dev = np.sqrt(np.dot(weights.T, np.dot(downside_returns.cov(), weights)))
        return downside_std_dev

    def monte_carlo_simulation(self, num_simulations=10000):
        num_assets = len(self.cov_matrix)
        results = np.zeros((4, num_simulations))
        for i in range(num_simulations):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            portfolio_return, portfolio_variance, portfolio_std_dev, sharpe_ratio, sortino_ratio = self.assess_portfolio(weights)
            results[0,i] = portfolio_return
            results[1,i] = portfolio_std_dev
            results[2,i] = sharpe_ratio
            results[3,i] = sortino_ratio
        return results

    def predict_returns(self):
        """
        Predict future returns using a linear regression model.
        """
        model = LinearRegression()
        X = np.arange(len(self.returns)).reshape(-1, 1)
        predicted_returns = {}
        for ticker in self.returns.columns:
            y = self.returns[ticker].values
            model.fit(X, y)
            predicted_returns[ticker] = model.predict(X[-1].reshape(1, -1))[0]
        return pd.Series(predicted_returns)
