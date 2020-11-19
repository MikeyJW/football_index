# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 11:07:38 2020

@author: MikeyJW
"""

import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Import model data
df = pd.read_csv('model_data.csv')
df.set_index('PlayerName', inplace=True)

# Generate positional dummies
df = pd.get_dummies(df, prefix='', prefix_sep='')

yX['Age2'] = yX['Age']**2
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


X_adjusted = yX[['Age', 'Age2', 'ave_matchday_score', 'ave_matchday_score^2',#'num_games_played', 
        'Forward', 'Midfielder', 'Defender', 'Goalkeeper']]

model = sm.OLS(y, X_adjusted).fit(cov_type='HC0')
model.summary()