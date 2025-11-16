import pandas as pd

def calculate_revenue_and_profit_margins(financial_data):
    """
    Calculate revenue and profit margins.;x
    financial_data: DataFrame containing financial data with columns 'Revenue' and 'Profit'
    """
    revenue = financial_data['Revenue']
    profit = financial_data['Profit']
    profit_margin = profit / revenue
    return revenue, profit_margin

def calculate_roi(investment_data):
    """
    Calculate return on investment (ROI).
    investment_data: DataFrame containing investment data with columns 'Investment' and 'Return'
    """
    investment = investment_data['Investment']
    return_on_investment = investment_data['Return']
    roi = return_on_investment / investment
    return roi

def calculate_eps(financial_data):
    """
    Calculate earnings per share (EPS).
    financial_data: DataFrame containing financial data with columns 'Net Income' and 'Shares Outstanding'
    """
    net_income = financial_data['Net Income']
    shares_outstanding = financial_data['Shares Outstanding']
    eps = net_income / shares_outstanding
    return eps
