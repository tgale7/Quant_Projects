import numpy as np
import matplotlib.pyplot as plt

#  set up parameters for Brownian Motion
s0 = 100
sigma = 0.351
mu = 0.15

#  set up parameters for simulation
paths = 10000
delta = 1 / 252
time = 5 * 252

# setting up Brownian motion for prediction
def wiener_process(delta, sigma, time, paths):
    #  return array of samples from normal distribution
    return sigma * np.random.normal(loc=0, scale=np.sqrt(delta), size=(time, paths))

# defining geometric Brownian motion for returns
def gbm_returns(delta, sigma, time, mu, paths):

    process = wiener_process(delta, sigma, time, paths)
    return np.exp(process + (mu - sigma**2 / 2) * delta)

# producing price paths by multiplying s0 by the cumulative product of GBM returns
def gbm_levels(s0, delta, sigma, time, mu, paths):

    returns = gbm_returns(delta, sigma, time, mu, paths)

    stacked = np.vstack([np.ones(paths), returns])
    return s0 * stacked.cumprod(axis=0)

# plotting results

price_paths = gbm_levels(s0, delta, sigma, time, mu, paths)
plt.plot(price_paths, linewidth=0.25)
plt.show()

# returning how many paths were higher than initial price
print(len(price_paths[-1, price_paths[-1, :] > s0]))