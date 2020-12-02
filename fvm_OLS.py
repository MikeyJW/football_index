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
path = 'file:///C:/Users/micha/Documents/Quant/football_index/model_data.csv'
df = pd.read_csv(path)
df.set_index('PlayerName', inplace=True)

# Generate positional dummies
df = pd.get_dummies(df, prefix='', prefix_sep='')

#df['Age2'] = df['Age']**2
#df['ave_matchday_score2'] = df['ave_matchday_score']**2

# Regress
y = df['CurrentPrice']
X = df[['Age', 'ave_matchday_score', #'num_games_played', 
        'Forward', 'Midfielder', 'Defender', 'Goalkeeper']]

# Check for multicollinearity among explanatory vars (especially the squares)
corr_table = X.corr()
#mask = np.zeros_like(corr_table)
#mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_table, annot=True, cmap="YlGnBu")

# Proceed

model = sm.OLS(y, X).fit(cov_type='HC0')
model.summary()

# Optimise your spec

# Consider partial regression plots
sm.graphics.plot_partregress('CurrentPrice', 'Age', ['ave_matchday_score', 'Forward', 'Midfielder', 'Defender', 'Goalkeeper'], data=df, obs_labels=False)

sm.graphics.plot_partregress('CurrentPrice', 'ave_matchday_score', ['Age', 'Forward', 'Midfielder', 'Defender', 'Goalkeeper'], data=df, obs_labels=False)

# SAY THEY COULD BE OUTLIERS THO. lOOK AT THEM PLOTS, THEY'RE PRETTY WILD


# Since squares are going to be highly correlated w/ their lower order counterparts we center them first
to_center = df[['Age', 'ave_matchday_score']]
df_centered = to_center.subtract(to_center.mean())
positional_indicators = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
df_centered[positional_indicators] = df[positional_indicators]


df_centered['Age2'] = df_centered['Age']**2
df_centered['ave_matchday_score2'] = df_centered['ave_matchday_score']**2


X_adjusted = df_centered[['Age', 'Age2', 'ave_matchday_score', 'ave_matchday_score2',#'num_games_played', 
        'Forward', 'Midfielder', 'Defender', 'Goalkeeper']]

model = sm.OLS(y, X_adjusted).fit(cov_type='HC0')
model.summary()