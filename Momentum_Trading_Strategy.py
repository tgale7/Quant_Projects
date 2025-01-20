import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# getting up-to-date financial data
gld = yf.download("GLD", period='5y')

# assigning days
day = np.arange(1, len(gld) + 1)
gld['day'] = day

# removing irrelevant columns
gld.drop(columns=['Adj Close', 'Volume'], inplace=True)

# creating moving average for 9 and 21 days as a measure of movement
gld['9-day'] = gld['Close'].rolling(9).mean().shift()  # shift used to ensure we are looking at the previous close, not the current day
gld['21-day'] = gld['Close'].rolling(21).mean().shift()

# conditions for being long or short, creates a signal to indicate long or short
gld['signal'] = np.where(gld['9-day'] > gld['21-day'], 1, 0)
gld['signal'] = np.where(gld['9-day'] < gld['21-day'], -1, gld['signal'])
gld.dropna(inplace=True)

# calculating returns by holding gold, and through the trading system
gld['Return'] = np.log(gld['Close']).diff()
gld['system_return'] = gld['signal'] * gld['Return']
gld['entry'] = gld.signal.diff()


# plotting model
plt.rcParams['figure.figsize'] = 12, 6
plt.plot(gld.iloc[-252:]['Close'], label='GLD')     # price of gold
plt.plot(gld.iloc[-252:]['9-day'], label='9-day')   # 9-day average
plt.plot(gld.iloc[-252:]['21-day'], label='21-day')     # 21-day average
plt.plot(gld[-252:].loc[gld.entry == 2].index, gld[-252:]['9-day'][gld.entry == 2], '^', color='g', markersize=12)  # indicates a buy
plt.plot(gld[-252:].loc[gld.entry == 2].index, gld[-252:]['9-day'][gld.entry == 2], 'v', color='r', markersize=12)  # indicates a sell
plt.legend(loc=2)
plt.show()

# plotting performance of buying and holding vs trading system
gld['system_return'] = gld.signal * gld.Return
plt.plot(np.exp(gld.Return).cumprod(), label='Buy/Hold')
plt.plot(np.exp(gld.system_return).cumprod(), label='System')    # cumprod tracks the cumulative product
plt.legend(loc=2)
plt.show()

print(gld)

# returning the returns for holding and trading separately as % to compare
print(np.exp(gld['Return']).cumprod()[-1] - 1)
print(np.exp(gld['system_return']).cumprod()[-1] - 1)


