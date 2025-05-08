import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

#  Getting stock data and option data

stock = yf.Ticker("AAPL")
expiration_dates = stock.options

# Black-Scholes formula for option pricing


def black_scholes(S, K, T, r, sigma, flag="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif flag == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Vega - derivative of option price w.r.t volatility


def vega(S, K, T, r, sigma, flag="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega_value = S * norm.pdf(d1) * np.sqrt(T)
    print(vega_value)
    return vega_value


# Newton-Raphson to find implied vol


def implied_vol_newton(option_price, S, K, T, r, flag='call', tolerance=0.00001, max_iter=200):
    sigma_old = 0.2  # initial guess
    for k in range(max_iter):
        price = black_scholes(S, K, T, r, sigma_old, flag)
        vega_value = vega(S, K, T, r, sigma_old, flag)

        if vega_value ==0:
            break

        # applying newton-raphson
        sigma_new = sigma_old - (price - option_price) / vega_value

        # if change is small enough then we stop
        if abs(sigma_new - sigma_old) < tolerance:
            return sigma_new

        sigma_old = sigma_new

    return sigma_old


# Acquire Stock price and risk-free rate
S = stock.history(period="1d")['Close'][0]  # last closing price
r = 0.01  # assuming 1% risk free rate for simplicity

# Get multiple expiration dates (e.g., first 100)
expiration_dates = expiration_dates[:100]
all_strikes = []
all_expirations = []
all_vols = []

for expiration_date in expiration_dates:
    try:
        calls = stock.option_chain(expiration_date).calls
        expiration_time = (pd.to_datetime(expiration_date) - pd.to_datetime('today')).days

        # Filter out options with missing or zero last price
        calls = calls[(calls['lastPrice'] > 0) & (calls['strike'] > 0)]
        strikes = calls['strike'][:10].values  # up to 10 strikes per expiry
        for strike in strikes:
            option_price_call = calls[calls['strike'] == strike]['lastPrice'].values[0]
            iv_call = implied_vol_newton(option_price_call, S, strike, expiration_time, r, flag='call')
            all_strikes.append(strike)
            all_expirations.append(expiration_time)
            all_vols.append(iv_call)
    except Exception as e:
        print(f"Error with expiry {expiration_date}: {e}")

# Convert lists to 2D grid for surface plot

strike_grid = np.linspace(min(all_strikes), max(all_strikes), 30)
exp_grid = np.linspace(min(all_expirations), max(all_expirations), 30)
X, Y = np.meshgrid(strike_grid, exp_grid)
Z = griddata((all_strikes, all_expirations), all_vols, (X, Y), method='linear')

# Plotting the implied volatility surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Expiration (Days)')
ax.set_zlabel('Implied Volatility')
ax.set_title('Call Option Implied Volatility Surface')
plt.show()
