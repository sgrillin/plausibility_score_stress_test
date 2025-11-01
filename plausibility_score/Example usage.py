# Example usage
import yfinance as yf
import pandas as pd
import numpy as np
import score_class as sc

# Download historical data for a stock (e.g., Apple Inc.)
ticker = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(ticker,
            start='2020-01-01',
            end='2025-10-31',
            progress=False,
            auto_adjust=False)['Adj Close']

# Compute log returns
returns = np.log(data / data.shift(1)).dropna()

# Fit the stress testing model
Sigma = returns.cov().values  # shape (3, 3)

# 4) Portfolio vector P must match the number/order of tickers
P = np.array([2.0, -5.0, 10.0])  # shape (3,)

# 5) Instantiate the model and compute the optimal scenario
ps = sc.PlausibilityStress(Sigma, dist="gaussian")  # Gaussian case
S_star, q = ps.optimal_scenario(P, alpha=0.999)

# 6) Present results
print("Tickers:", ticker)
print("Alpha:", 0.999)
print("Loss quantile (q):", q)
print("\nOptimal scenario S* (percent moves):")
print(pd.Series(100.0 * S_star, index=ticker))