# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:13:43 2020

@author: MikeyJW
"""

''' EDA '''

#TODO: Remove short smaples from the mkt beta calculations (or do the top 100 format)
#TODO: Compare returns to SPY
#TODO: Think of more classic stock market stuff
#TODO: Make top 200 mask dynamic

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

first_day = df.loc[('2019-07-01', slice(None)), :]
top200_mask = first_day.nlargest(200, 'EndofDayPrice').index.droplevel(0).values
top200 = df.loc[(slice(None), top200_mask), :]


# Gen informative plots


# Extract index price
index_price = df.groupby(level=0)['ave_mkt_price'].head(1).values
# Benchmark (holding roughly the same amount of all players TODO: THIS NEEDS REFINING [if we can't hold equal values of all players???])
mkt_log_returns = df['daily_mkt_log_returns'].groupby(level=0).head(1).droplevel(1)
mkt_log_returns.cumsum().apply(np.exp).plot(title='Benchmark')

# PLOT: Market aggregates
# TODO: put epl start/end dates in for reference.
mkt_returns = mkt_log_returns.apply(np.exp)
mkt_returns.plot(title='Daily market returns')
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

def custom_resampler(resampled_df):
    return resampled_df.head(1)

def find_cutoff(data):
    return data.sort_values()[-15]

def momentum_strat(data, lookback_window=14, holding_period=14, param_dict=None):
    'Buy top n momentum players from lookback period and hold for duration of holding period'
    # TODO: Val this actually works
    # TODO: FIX THE CUTOFF SITCH (we hold over 10 positions some days. Tie-breaker, highest price???)
    # If dictionary passed, use those parameters instead
    if param_dict:
        lookback_window = param_dict['lookback_window']
        holding_period = param_dict['holding_period']
    
    technicals_df = data['EndofDayPrice'].groupby(level=1).pct_change(lookback_window) # Make momentum factor TODO: USE LOG RETURNS FOR THIS AS WELL. (diff???)
    
    # Find 15th place cutoff
    cutoff = technicals_df.groupby(level=0).apply(find_cutoff)
    
    # Generate portfolio dataframe
    portfolio_df = technicals_df.unstack()
    portfolio_df[portfolio_df.values >= cutoff.values.reshape(len(cutoff),1)] = 1
    portfolio_df[portfolio_df.values < cutoff.values.reshape(len(cutoff),1)] = 0
    portfolio_df = portfolio_df[lookback_window+1:] # Slice off the 'warm-up' period
    
    # Fix positions for holding period
    fixed_portfolio = portfolio_df.resample(str(holding_period) + 'D').apply(custom_resampler)
    fixed_portfolio = fixed_portfolio.resample('1D').ffill()
    #fixed_portfolio.iloc[:lookback_window, :] = 0 # Wait for duration of lookback window WAIT EARLIER

    # Shift portfolio positions 1 day forward to avoid look-ahead bias
    fixed_portfolio = fixed_portfolio.shift(1)

    daily_log_returns = data['daily_log_returns'].unstack()
    strat_returns = fixed_portfolio * daily_log_returns[fixed_portfolio.index.min():fixed_portfolio.index.max()]
    strat_returns['cumulative_pf_returns'] = strat_returns.mean(axis=1).cumsum().apply(np.exp)
    
    return strat_returns['cumulative_pf_returns']

from sklearn.model_selection import ParameterGrid

def custom_grid_search(data, strategy, param_grid):
    'Grid search function for quant strategy optimisation'
    # Init objects to record results
    results = {} 
    results_list = []
    i = 0
    
    # Try all combinations of parameters from param_grid
    param_combos = ParameterGrid(param_grid)
    
    for param_dict in param_combos:
        # Extract and input params
        strat_returns = strategy(data, param_dict=param_dict)
        
        # Record results
        result = strat_returns[-1] # something like that???
        
        results['Combination ' + str(i)] = param_dict
        results['Combination ' + str(i)]['test_period_return'] = result        
        i += 1
        
        results_list.append(result)

    # Identify top performing set    
    optimal_index = results_list.index(max(results_list))
    optimal = results['Combination ' + str(optimal_index)]
    
    print('Optimal: \n' + str(optimal))
    return results





# STRATEGY 1: Price momentum
data = top200
strategy = momentum_strat
param_grid = {'lookback_window': [3, 7, 14, 21],
              'holding_period': [3, 7, 14, 21]}

results = custom_grid_search(data, strategy, param_grid)
print(results)

optimal_params =  {'holding_period': 7,
                   'lookback_window': 7}

results = momentum_strat(data, param_dict=optimal_params)
results.plot(title='Momentum strategy cumulative returns')
# Check Sharpe ratio?

# STRATEGY 2: Mean reversion
# Find biggest losers in past x period, and long them. 


# STRATEGY 3: Post div. drift

# STRATEGY 4: Post div. reversion

# STRATEGY 5: SMAC

# STRATEGY 6: EMAC