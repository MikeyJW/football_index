# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:13:43 2020

@author: MikeyJW
"""

''' EDA '''

#TODO: Remove short smaples from the mkt beta calculations (or do the top 100 format)
#TODO: Compare returns to SPY
#TODO: Think of more classic stock market stuff

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import yfinance as yf [to get spy numbers]

# Import and organise dataframe
df = pd.read_csv('Q32019_Q22020.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(['Date', 'PlayerName'], inplace=True)
df.sort_index(level=0, inplace=True)

df['daily_log_returns'] = df['EndofDayPrice'].apply(np.log).groupby(level=1).diff(1)
df['daily_mkt_log_returns'] = df['ave_mkt_price'].apply(np.log).groupby(level=1).diff(1) 

first_day = df.loc[('2020-06-06', slice(None)), :]
top200_mask = first_day.nlargest(200, 'EndofDayPrice').index.droplevel(0).values
top200 = df.loc[(slice(None), top200_mask), 'EndofDayPrice']


# Extract market returns
mkt_returns = df.groupby(level=0)['daily_mkt_returns'].head(1).values
# Extract index price
index_price = df.groupby(level=0)['ave_mkt_price'].head(1).values
# Benchmark (holding roughly the same amount of all players TODO: THIS NEEDS REFINING [if we can't hold equal values of all players???])
compounding_index_return = df['daily_mkt_log_returns'].groupby(level=0).head(1).droplevel(1)
compounding_index_return.cumsum().apply(np.exp).plot(title='Benchmark')

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
# groupby playername, get return on end of day price for each 
df['prev_week_return'] = df['EndofDayPrice'].groupby(level=1).pct_change(14)

# Backtesting

technicals_df = df['prev_week_return'].unstack().fillna(-1)

def momentum_strat(technicals_df):
    'Find top n momentum players from lookback period'
    # Find 10th place cutoff
    transposed = technicals_df.transpose()
    cutoff = transposed.sort_values(by=transposed.columns[0]).iloc[-15].values[0]
    # Generate portfolio dataframe
    portfolio_df = transposed
    portfolio_df[portfolio_df.values >= cutoff] = 1
    portfolio_df[portfolio_df < cutoff] = 0
    return portfolio_df.transpose()

portfolio_df = technicals_df.groupby(level=0).apply(momentum_strat)
# Shift portfolio positions to avoid look-ahead bias
portfolio_df = portfolio_df.shift(1)

# Fix optimal portfolio positions for n days
def custom_resampler(resampled_df):
    return resampled_df.head(1)

fixed_portfolio = portfolio_df.resample('14D').apply(custom_resampler)
fixed_portfolio = fixed_portfolio.resample('1D').ffill()

# Find strategy returns
daily_log_returns = df['daily_log_returns'].unstack()
strat_returns = fixed_portfolio * daily_log_returns[:'2020-06-15'] #Chopping off end as upsample wont fill up the end of the dataframe
strat_returns['total_returns'] = strat_returns.mean(axis=1)

# WARN: Remove front 14 days where we have no info
print(strat_returns.loc['2019-07-14':, 'total_returns'].cumsum().apply(np.exp)[-1])

# for starters only consider top-150, then gridsearch???

# Gridsearch holding period with spread + commission averages

# STRATEGY 2: Mean reversion

# STRATEGY 3: Post div. drift

# STRATEGY 4: Post div. reversion

# STRATEGY 5: SMAC