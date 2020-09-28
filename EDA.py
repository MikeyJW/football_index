# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:13:43 2020

@author: MikeyJW
"""

''' EDA '''

#TODO: Remove short smaples from the mkt beta calculations (or do the top 100 format)
#TODO: Variance, box plots, on returns variable, the classics [make]
#TODO: Compare returns to SPY
#TODO: Think of more classic stock market stuff
#TODO: Liquidity (only consider the top 100 players)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import and organise dataframe
df = pd.read_csv('Q32019_Q22020.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(['Date', 'PlayerName'], inplace=True)
df.sort_index(level=0, inplace=True)

# Extract market returns
mkt_returns = df.groupby(level=0)['daily_mkt_returns'].head(1).values
# Extract index price
index_price = df.groupby(level=0)['ave_mkt_price'].head(1).values


# PLOT: Market aggregates
# TODO: put epl start end dates in for reference.
plt.plot(mkt_returns)
plt.plot(index_price) # Compare this to the money in circulation stats from FIE
sns.kdeplot(mkt_returns) # Comp to norm dist/SPY
plt.boxplot(mkt_returns.dropna())


# Gen cov matrix and slice off the useful values
cov = df.groupby(level=1)[['daily_returns', 'daily_mkt_returns']].cov()
cov = cov.loc[(slice(None), 'daily_mkt_returns'), 'daily_returns']

cov.index = cov.index.droplevel(1)
mkt_beta = cov / (df.groupby(level=1)['daily_returns'].std() ** 2)

# PLOT: Market beta dispersion
sns.kdeplot(mkt_beta.dropna())


# Begin scouting basic quant strategy: 
# Buy on certain business days, times of the month. 
# Mean reversions
# Passing meving averages
# Receiving div's. (Any type of drift?)


# Business day effects
df['dayofweek'] = df.index.get_level_values(0)
df['dayofweek'] = df['dayofweek'].dt.weekday

# Extract day of week from index
mkt_returns = df.groupby(level=0)[['daily_mkt_returns', 'dayofweek']].head(1)
mkt_returns.index = mkt_returns.index.droplevel(level=1)

# PLOT: Day of week trends
mkt_returns.groupby('dayofweek').mean()
sns.boxplot(x="dayofweek", y='daily_mkt_returns', data=mkt_returns)
sns.violinplot(x="dayofweek", y='daily_mkt_returns', data=mkt_returns)

# STRATEGY 1: Momentum

# Buy winners over different time periods, compare to market returns.


3# STRATEGY 2: Mean reversion

# STRATEGY 3: Post div. drift

# STARTEGY 4: Post div. reversion