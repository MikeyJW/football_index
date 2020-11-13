# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 11:07:38 2020

@author: MikeyJW
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# Import from files...
df1 = pd.read_csv('matchday_scores.csv')
df1['Date'] = pd.to_datetime(df1['Date'], format='%d/%m/%Y')

df2 = pd.read_csv('player_info.csv')
df2.set_index('PlayerName', inplace=True)


# Extract average matchday score since game x
season_start = '2020-09-12' # Start of UK football season
df1 = df1[df1['Date'] >= season_start]

ave_matchday_score = df1.groupby('PlayerName')['MatchdayScore'].mean()
num_games_played = df1.groupby('PlayerName').size()

df2['ave_matchday_score'] = ave_matchday_score
df2['num_games_played'] = num_games_played

df = df2[['Position', 'Age', 'num_games_played', 'ave_matchday_score',  'CurrentPrice']]

# Clean up df
df['CurrentPrice'] = df['CurrentPrice'].str.strip('£').astype('float64')

# Generate positional dummies
df = pd.get_dummies(df, prefix='', prefix_sep='')

yX = df[df['num_games_played'] >= 5]

yX['Age^2'] = yX['Age']**2
yX['ave_matchday_score2'] = yX['ave_matchday_score']**2

# Regress
y = yX['CurrentPrice']
X = yX[['Age', 'ave_matchday_score', #'num_games_played', 
        'Forward', 'Midfielder', 'Defender', 'Goalkeeper']]

# Check for multicollinearity among explanatory vars
corr_table = X.corr()
#mask = np.zeros_like(corr_table)
#mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_table, cmap="YlGnBu")

# Proceed

model = sm.OLS(y, X).fit(cov_type='HC0')
model.summary()

# Optimise your spec

# Consider 
sm.graphics.plot_partregress('CurrentPrice', 'Age', ['ave_matchday_score', 'Forward', 'Midfielder', 'Defender', 'Goalkeeper'], data=yX, obs_labels=False)
# Try a square of age (check how you check if that's a valid change)
sm.graphics.plot_partregress('CurrentPrice', 'ave_matchday_score', ['Age', 'Forward', 'Midfielder', 'Defender', 'Goalkeeper'], data=yX, obs_labels=False)
# Try a square (or more?)
# SAY THEY COULD BE OUTLIERS THO. lOOK AT THEM PLOTS, THEY'RE PRETTY WILD


X_adjusted = yX[['Age', 'Age^2', 'ave_matchday_score', 'ave_matchday_score^2',#'num_games_played', 
        'Forward', 'Midfielder', 'Defender', 'Goalkeeper']]

model = sm.OLS(y, X_adjusted).fit(cov_type='HC0')
model.summary()

# Plot results...


# Plot with PCA?
# How to plot regression results? (datacamp???)