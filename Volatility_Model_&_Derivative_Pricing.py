
### As model uses recent option data, errors can occur when running the model outside of trading hours ###

import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

#  Black-Scholes Model and Vega Calculation


def black_scholes(S, K, T, r, sigma, flag="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif flag == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

#  Newton-Raphson Method for Implied Volatility


def implied_vol_newton(option_price, S, K, T, r, flag='call', tolerance=1e-5, max_iter=200):
    sigma = 0.2
    for k in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, flag)
        v = vega(S, K, T, r, sigma)
        if v < 1e-8:
            break
        increment = (price - option_price) / v
        sigma -= increment
        sigma = max(sigma, 1e-6)  # enforce non-negative sigma
        if abs(increment) < tolerance:
            return sigma
    return np.nan  # return NaN if not converged

#  Download Stock and Option Data


stock = yf.Ticker("AAPL")
S = stock.history(period="1d")['Close'].iloc[-1]
r = 0.01  # assumed constant risk-free rate
expiration_dates = stock.options[:50]  # limit to first 50 expiries for speed

#  Obtaining Implied Volatility Data


all_strikes = []
all_expirations = []
all_vols = []

for expiration_date in expiration_dates:
    try:
        chain = stock.option_chain(expiration_date).calls
        T = (pd.to_datetime(expiration_date) - pd.to_datetime('today')).days / 365.0
        if T <= 0:
            continue

        # Use mid-price and clean data
        chain = chain[(chain['bid'] > 0) & (chain['ask'] > 0)]
        chain['mid'] = (chain['bid'] + chain['ask']) / 2
        chain = chain[(chain['mid'] > 0) & (chain['strike'] > 0)]

        for _, row in chain.head(25).iterrows():  # up to 25 strikes per expiry
            K = row['strike']
            price = row['mid']
            intrinsic = max(S - K, 0)
            if price <= intrinsic:
                continue  # skip over invalid inputs

            iv = implied_vol_newton(price, S, K, T, r, flag='call')
            if np.isnan(iv) or iv > 5:
                continue  # filter out unreasonable values values

            all_strikes.append(K)
            all_expirations.append(T)
            all_vols.append(iv)

    except Exception as e:
        print(f"Error processing {expiration_date}: {e}")

#  Building Implied Volatility Surface
if len(all_vols) == 0:
    raise ValueError("No valid implied volatilities found.")

strike_grid = np.linspace(min(all_strikes), max(all_strikes), 30)
exp_grid = np.linspace(min(all_expirations), max(all_expirations), 30)
X, Y = np.meshgrid(strike_grid, exp_grid)
Z = griddata((all_strikes, all_expirations), all_vols, (X, Y), method='linear')

#  Plot Surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y * 365, Z, cmap='viridis')  # Convert T back to days for plot
ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Expiration (Days)')
ax.set_zlabel('Implied Volatility')
ax.set_title('AAPL Call Option Implied Volatility Surface')
plt.tight_layout()
plt.show()

