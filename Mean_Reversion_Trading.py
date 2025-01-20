import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# getting up to date financial data
gld = yf.download("GLD", period='5y')

# finding a moving average
ma = 21
gld['returns'] = np.log(gld["Close"]).diff()
gld['ma'] = gld['Close'].rolling(ma).mean()
gld['ratio'] = gld['Close'] / gld['ma']

print(gld['ratio'].describe())

# finding the percentiles of ratios between prices
percentiles = [5, 10, 50, 90, 95]
p = np.percentile(gld['ratio'].dropna(), percentiles)

# plots the ratio between prices compared to the percentile ratios
gld['ratio'].dropna().plot(legend=True)
plt.axhline(p[0], c=(.5, .5, .5), ls='--')
plt.axhline(p[2], c=(.5, .5, .5), ls='--')
plt.axhline(p[-1], c=(.5, .5, .5), ls='--')
plt.show()

short = p[- 1]
long = p[0]
print(short, long)
gld['position'] = np.where(gld.ratio > short, -1, np.nan)
gld['position'] = np.where(gld.ratio < long, 1, gld['position'])
gld['position'] = gld['position'].ffill()

gld.position.dropna().plot()
plt.show()

gld['strat_returns'] = gld['returns'] * gld['position'].shift()

plt.plot(np.exp(gld['returns'].dropna()).cumprod(), label='Buy/Hold')
plt.plot(np.exp(gld['strat_returns'].dropna()).cumprod(), label='Strategy')
plt.legend()
plt.show()

print(np.exp(gld['returns'].dropna()).cumprod()[-1] - 1)
print(np.exp(gld['strat_returns'].dropna()).cumprod()[-1] - 1)