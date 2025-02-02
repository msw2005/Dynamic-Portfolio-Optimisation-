import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes formula for option pricing.
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility of the underlying asset
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return option_price

def monte_carlo_simulation(S, T, r, sigma, num_simulations=1000, num_steps=252):
    """
    Monte Carlo simulation for future price paths.
    S: Current stock price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility of the underlying asset
    num_simulations: Number of simulations
    num_steps: Number of time steps
    """
    dt = T / num_steps
    price_paths = np.zeros((num_steps, num_simulations))
    price_paths[0] = S
    for t in range(1, num_steps):
        z = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return price_paths