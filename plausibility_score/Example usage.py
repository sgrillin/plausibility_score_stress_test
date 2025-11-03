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
# Student-t elliptical model
ps_t = sc.PlausibilityStress(Sigma, dist="student", df=5)
S_star_t, q_t = ps_t.optimal_scenario(P, alpha=0.999)

print("Tickers:", ticker)
print("Alpha:", 0.999)
print("Loss quantile q_t:", q_t)
print("Optimal scenario (Student-t df=5):")
print(pd.Series(100 * S_star_t, index=ticker))

# Fixed shock vector (DECIMALS)
s = np.array([-0.10, -0.08, -0.12])

# Implied alphas
alpha_port = ps_t.implied_alpha_portfolio(s, P)   # 1-D portfolio quantile
alpha_joint = ps_t.implied_alpha_joint(s)         # n-D (elliptical) quantile

# Context
q_obs   = - float(P @ s)                           # positive loss
scale   = np.sqrt(float(P @ ps_t.Sigma @ P))       # <-- ps_t.Sigma (fix)
z_ratio = q_obs / scale

print("Shock vector s (in %):", 100*s)
print("Portfolio loss produced by s (q_obs):", q_obs)
print("Portfolio vol (sqrt(P'ΣP)):", scale)
print("Standardized portfolio loss (q_obs / sqrt(P'ΣP)):", z_ratio)
print("Implied portfolio alpha :", alpha_port)
print("Implied joint alpha     :", alpha_joint)
