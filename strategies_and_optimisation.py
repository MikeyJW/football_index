# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:27:21 2020

@author: MikeyJW
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

def gen_daily_sharpe(pf_returns, r_f=1.0):
    'Finds sharpe ratio from pandas series of returns'
    sigma = pf_returns.std()
    r_pf = pf_returns.mean()
    daily_sharpe = (r_pf - r_f) / sigma
    return daily_sharpe
    
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
        result = strat_returns[-1]
        
        results['Combination ' + str(i)] = param_dict
        results['Combination ' + str(i)]['test_period_return'] = result        
        i += 1
        
        results_list.append(result)

    # Identify top performing set    
    optimal_index = results_list.index(max(results_list))
    optimal = results['Combination ' + str(optimal_index)]
    
    print('Optimal: \n' + str(optimal))
    return results, optimal

def custom_resampler(resampled_df):
    return resampled_df.head(1)

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
    def find_cutoff(data):
        return data.sort_values()[-15]
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

def mean_reversion(data, lookback_window=14, holding_period=14, param_dict=None):
    'Buy worst performing players from lookback period and hold for duration of holding period'
    # TODO: Val this actually works
    # TODO: FIX THE CUTOFF SITCH (we hold over 10 positions some days. Tie-breaker, highest price???)
    # If dictionary passed, use those parameters instead
    if param_dict:
        lookback_window = param_dict['lookback_window']
        holding_period = param_dict['holding_period']
    
    technicals_df = data['EndofDayPrice'].groupby(level=1).pct_change(lookback_window) # Make momentum factor TODO: USE LOG RETURNS FOR THIS AS WELL. (diff???)
    
    # Find biggest losers 15th place cutoff
    def find_cutoff(data):
        return data.sort_values()[15]
    cutoff = technicals_df.groupby(level=0).apply(find_cutoff)
    
    # Generate portfolio dataframe
    portfolio_df = technicals_df.unstack()
    portfolio_df[portfolio_df.values > cutoff.values.reshape(len(cutoff),1)] = 0
    portfolio_df[portfolio_df.values <= cutoff.values.reshape(len(cutoff),1)] = 1
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


def post_div_drift(data, holding_period=14, param_dict=None):
    'Buy players who have just had performance/media dividends issued'
    # If dictionary passed, use those parameters instead
    if param_dict:
        holding_period = param_dict['holding_period']
    
    technicals_df = data['MatchdayDividends']
    
    # Either find cutoff
    '''
    def find_cutoff(data):
        return data.sort_values()[-15]
    cutoff = technicals_df.groupby(level=0).apply(find_cutoff)
    '''
    # Or just buy div recievers
    technicals_df.loc[technicals_df > 0,:] = 1
    
    portfolio_df = technicals_df.unstack()
    
    def gen_holding_period(player, holding_period):
        'Add holding period to portfolio dataframe'
        div_dates = player[player > 0].index
        
        for date in div_dates:
            date_plus_hold = date + pd.Timedelta(str(holding_period) + 'D')
            player.loc[date:date_plus_hold] = 1
        return player
    
    fixed_portfolio = pd.DataFrame(index=portfolio_df.index)
    # Fix positions for holding period
    for col in portfolio_df:
        player = portfolio_df[col]
        fixed_pf_player = gen_holding_period(player, holding_period)
        fixed_portfolio[col] = fixed_pf_player
    
    # Shift portfolio positions 1 day forward to avoid look-ahead bias
    fixed_portfolio = fixed_portfolio.shift(1)

    daily_log_returns = data['daily_log_returns'].unstack()
    strat_returns = fixed_portfolio * daily_log_returns[fixed_portfolio.index.min():fixed_portfolio.index.max()]
    strat_returns['cumulative_pf_returns'] = strat_returns.mean(axis=1).cumsum().apply(np.exp)
    
    return strat_returns['cumulative_pf_returns']


def SMAC(single_player_data, duration_MA1=5, duration_MA2=22, param_dict=None):
    'Long only simple moving average cross strategy for a single player'
    
    if param_dict:
        duration_MA1 = param_dict['duration_MA1']
        duration_MA2 = param_dict['duration_MA2']
    prices = single_player_data
    
    ma_x = prices.rolling(duration_MA1).mean() - prices.rolling(duration_MA2).mean()
    # Set trading positions (long only)
    ma_x[ma_x > 0] = 1
    ma_x[ma_x <=0] = 0
    ma_x.fillna(0, inplace=True) # Set holdings to 0 during warm-up period
    
    portfolio = ma_x
    
    # Shift portfolio positions 1 day forward to avoid look-ahead bias
    portfolio = portfolio.shift(1)
    
    # Gen daily log'ed returns for player
    daily_log_returns = single_player_data.apply(np.log).diff(1)
    
    strat_returns = portfolio * daily_log_returns
    strat_returns['cumulative_pf_returns'] = strat_returns.mean(axis=1).cumsum().apply(np.exp)
    
    return strat_returns['cumulative_pf_returns']
    
def EMAC(single_player_data, duration_EMA1=5, duration_EMA2=22, param_dict=None):    
    'Long only exponential moving average cross strategy for a single player'
    
    if param_dict:
        duration_EMA1 = param_dict['duration_EMA1']
        duration_EMA2 = param_dict['duration_EMA2']
    prices = single_player_data
    
    ma_x = prices.ewm(span=duration_EMA1).mean() - prices.ewm(span=duration_EMA2).mean()
    
    # Set trading positions (long only)
    ma_x[ma_x > 0] = 1
    ma_x[ma_x <=0] = 0
    ma_x.iloc[:max(duration_EMA1, duration_EMA2)] = 0 # Set holdings to 0 during warm-up period
    
    portfolio = ma_x
    
    # Shift portfolio positions 1 day forward to avoid look-ahead bias
    portfolio = portfolio.shift(1)
    
    # Gen daily log'ed returns for player
    daily_log_returns = single_player_data.apply(np.log).diff(1)
    
    strat_returns = portfolio * daily_log_returns
    strat_returns['cumulative_pf_returns'] = strat_returns.mean(axis=1).cumsum().apply(np.exp)
    
    return strat_returns['cumulative_pf_returns']