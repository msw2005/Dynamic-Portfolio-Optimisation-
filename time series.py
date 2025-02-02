import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker symbols for FTSE 100 and S&P 500
ftse_ticker = "^FTSE"
sp500_ticker = "^GSPC"

# Define the date range for minute-level data
start_date = "2024-12-03"
end_date = "2024-12-04"

# Fetch the minute-level data
ftse100_data = yf.download(ftse_ticker, start=start_date, end=end_date, interval="1m")
sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date, interval="1m")

# Plot the data
plt.figure(figsize=(12, 8))

# Plot FTSE 100
plt.plot(ftse100_data['Close'], label='FTSE 100')

# Plot S&P 500
plt.plot(sp500_data['Close'], label='S&P 500')

plt.title('FTSE 100 and S&P 500 Minute-by-Minute Price')
plt.xlabel('DateTime')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()