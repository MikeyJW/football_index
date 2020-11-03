# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:13:43 2020

@author: MikeyJW
"""

''' EDA '''

#TODO: Remove short samples from the mkt beta calculations (or do the top 100 format)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from strategies_and_optimisation import gen_daily_sharpe, custom_grid_search, momentum_strat, mean_reversion, post_div_drift, SMAC, EMAC


# Import and organise dataframe
df = pd.read_csv('Q32019_Q22020.csv')
df['Date'] = pd.to_datetime(df['Date'])

start_date = df['Date'].min()
end_date = df['Date'].max()

df.set_index(['Date', 'PlayerName'], inplace=True)
df.sort_index(level=0, inplace=True)

df['daily_log_returns'] = df['EndofDayPrice'].apply(np.log).groupby(level=1).diff(1)
df['daily_mkt_log_returns'] = df['ave_mkt_price'].apply(np.log).groupby(level=1).diff(1) 

first_day = df.loc[(start_date, slice(None)), :]
top200_mask = first_day.nlargest(200, 'EndofDayPrice').index.droplevel(0).values
top200 = df.loc[(slice(None), top200_mask), :]


# Gen informative plots

# Extract index price
index_price = df.groupby(level=0)['ave_mkt_price'].head(1).values
mkt_log_returns = df['daily_mkt_log_returns'].groupby(level=0).head(1).droplevel(1)
mkt_log_returns.cumsum().apply(np.exp).plot(title='Benchmark')

# Comparing to SP500
SPY = yf.Ticker('SPY')
#SPY_history = SPY.history(start=start_date, end=end_date)['Close'] # Care dont overuse, works on yahoo scraping.
SPY_log_returns = SPY_history.apply(np.log).diff(1)
SPY_log_returns.cumsum().apply(np.exp).plot(title='SPY returns')


# Benchmark Sharpe ratio
benchmark_sharpe = gen_daily_sharpe(mkt_returns)
print('Benchmark Sharpe ratio: ' + str(benchmark_sharpe))
# NOTE/ YOU X THESE BY ROOT(TRADING DAYS), 
# SO JUST ALL DAYS SINCE FI IS A 365-DAY MARKET. You're counting returns across all those days. 
# If you're not mean'ing across sat-sun in SP500 calcs, then dont over multiply when you annualise the sharpes.
# It would be a fairer comparison if you had a year's data for both.

SPY_returns.plot(title='Daily SPY returns')
SPY_sharpe = gen_daily_sharpe(SPY_returns)
print('SPY Sharpe ratio: ' + str(SPY_sharpe))


# KDE plots
sns.kdeplot(mkt_returns) # Comp to norm dist/SPY
sns.kdeplot(SPY_returns)
#sns.kdeplot(np.random.randn(mkt_returns.variance, mean=0)) #etc... norm dist

plt.boxplot(mkt_returns.dropna())

# Gen cov matrix and slice off the useful values
cov = df.groupby(level=1)[['daily_returns', 'daily_mkt_returns']].cov()
cov = cov.loc[(slice(None), 'daily_mkt_returns'), 'daily_returns']

cov.index = cov.index.droplevel(1)
mkt_beta = cov / (df.groupby(level=1)['daily_returns'].std() ** 2)

# PLOT: Market beta dispersion
sns.kdeplot(mkt_beta.dropna())


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




####################################################################





# Begin scouting basic quant strategy:

# STRATEGY 1: Price momentum
data = top200
strategy = momentum_strat
param_grid = {'lookback_window': [3, 7, 14, 21],
              'holding_period': [3, 7, 14, 21]}

results = custom_grid_search(data, strategy, param_grid)
print(results)

optimal_params =  {'holding_period': 7,
                   'lookback_window': 7}
# TODO: Get grid_search returning optimal params to chuck in. (then only run optimals in the final thing.)
results = momentum_strat(data, param_dict=optimal_params)
results.plot(title='Momentum strategy cumulative returns')

daily_pf_returns = results.pct_change(1) + 1
strategy_sharpe = gen_daily_sharpe(daily_pf_returns)
# TODO: Do a nice kde plot with normalized (- mean???) returns showing var???


# STRATEGY 2: Mean reversion
# Find biggest losers in past x period, and long them. # This is a bad strat but holding periods tell us something???
data = top200
strategy = mean_reversion
param_grid = {'lookback_window': [3, 7, 14, 21],
              'holding_period': [3, 7, 14, 21]}

results = custom_grid_search(data, strategy, param_grid)
print(results)

optimal_params =  {'holding_period': 7,
                   'lookback_window': 21}

results = mean_reversion(data, param_dict=optimal_params)
results.plot(title='Mean reversion cumulative returns')



# STRATEGY 3: Post div. drift
data = top200
strategy = post_div_drift
param_grid = {'holding_period': [3, 6, 13, 20]}

results = custom_grid_search(data, strategy, param_grid)
print(results)

optimal_params =  {'holding_period': 20}

results = post_div_drift(data, param_dict=optimal_params)
results.plot(title='Mean reversion cumulative returns')



# STRATEGY 5.1: SMAC (single player)
data = top200.loc[(slice(None),'Mohamed Salah'), 'EndofDayPrice'].unstack()
strategy = SMAC
param_grid = {'duration_MA1': [5, 22],
              'duration_MA2': [5, 22]}

results = custom_grid_search(data, strategy, param_grid)
print(results)

optimal_params =  {'duration_MA1': 5, 'duration_MA2': 22}

results = SMAC(data, param_dict=optimal_params)
results.plot(title='Mean reversion cumulative returns')



# STRATEGY 5.2: EMAC
data = top200.loc[(slice(None),'Mohamed Salah'), 'EndofDayPrice'].unstack()
strategy = EMAC
param_grid = {'duration_EMA1': [5, 22],
              'duration_EMA2': [5, 22]}

results = custom_grid_search(data, strategy, param_grid)
print(results)

optimal_params =  {'duration_EMA1': 5, 'duration_EMA2': 22}

results = EMAC(data, param_dict=optimal_params)
results.plot(title='Mean reversion cumulative returns')

# Shows potential given Salah's dropped in value, and it didn't lose as much relative to benchmark, couldve gained some
# See if we can optimise.