import pandas as pd
from math import comb
import numpy as np


df = pd.read_csv("/Users/tomgale/Documents/S&P500_YTD.csv")
df = df.reset_index()

# approximating up and down factor
u_counter = 0
u_amount = 0

for index, row in df.iterrows():
    if row["Close/Last"] > row["Open"]:
        u_counter = u_counter + 1
        u_amount = u_amount + row["Close/Last"]/row["Open"]

u = u_amount/u_counter

# for simplicity of binomial symmetry
d = 1/u


# Gathering inputs
option_call = int(input("How many call options? "))
option_put = int(input("How many put options? "))
r = float(input("What is the current interest rate? "))
T = int(input("What is the time until maturity (in days) for the option? "))
K = float(input("What is the strike price of the option? "))
S0 = float(input("What is the stock price today? "))


# risk neutral measure

rnm = (1+r-d)/(u-d)
print(rnm)
print(u)
while rnm >= 1:
    print("This value of r is not arbitrage-free")
    r = float(input("What is the current interest rate? "))
    rnm = (1+r-d)/(u-d)

# create array to represent each up and down step


# finding price for each event

payoff = 0
expectation = 0

if option_call > 0:
    for j in range(1, option_call+1):
        for i in range(0, T+1):
            ST = S0 * (u ** i) * (d ** (T-i))
            payoff = comb(T, i) * max(ST-K, 0)
            prob = (rnm ** i) * ((1 - rnm) ** (T - i))
            expectation = expectation + (prob * payoff)
if option_put > 0:
    for k in range(1, option_put+1):
        for i in range(0, T + 1):
            ST = S0 * (u ** i) * (d ** (T - i))
            payoff = comb(T, i) * max(K-ST, 0)
            prob = (rnm ** i) * ((1 - rnm) ** (T - i))
            expectation = expectation + prob*payoff

option_price = expectation / ((1 + r) ** T)
print("This option combination is valued at: ", option_price)








