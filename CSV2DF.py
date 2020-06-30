# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:02 2020

@author: MikeyJW
"""

import pandas as pd
import numpy as np

csv_list = ['Q3_2019.csv', 'Q4_2019P1.csv',
            'Q4_2019P2.csv', 'Q1_2020P1.csv',
            'Q1_2020P2.csv', 'Q2_2020P1.csv']

# Initialise dataframe and combine seperate csv's
df = pd.DataFrame()


for file in csv_list:
    temp = pd.read_csv(file, low_memory=False)
    df = pd.concat([df, temp])

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

col_list = ['EndofDayPrice', 'MESellPrice',
            'MatchdayDividends', 'MediaDividends']

for col in col_list:
    df[col] = df[col].str.strip('£').astype('float64')

df.set_index(['Date', 'PlayerName'], inplace=True)
df.sort_index(level=0, inplace=True)

df['daily_returns'] = df['EndofDayPrice'].groupby(level=1).pct_change()

mkt_price = df['EndofDayPrice'].groupby(level=0).mean()
mkt_returns = df['daily_returns'].groupby(level=0).mean()

df['ave_mkt_price'] = mkt_price.reindex(df.index, level=0)
df['daily_mkt_returns'] = mkt_returns.reindex(df.index, level=0)

# pd.save_dataframe...


#############################################################


''' EDA '''

#TODO: MKT and corr plot + beta calculations
#TODO: Variance, box plots, on returns variable, the classics [make]
#TODO: Compare returns to SPY
#TODO: Think of more classic stock market stuff

import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(mkt_returns)
plt.plot(mkt_price)

plt.boxplot(mkt_returns.dropna())
plt.show()

sns.kdeplot(mkt_returns) #Comp to norm dist/SPY

cov = np.covariance(df['daily_returns'], df['daily_mkt_returns'])
mkt_beta = cov / df['daily_returns'].std() ** 2

# Correlation
corr = df[['daily_returns', 'daily_mkt_returns']].corr()


# How much do player returns correlate to market returns???


# Market beta dispersion, players should all be fairly close to 1?

# Correlation / Regression result [ON HIGH LIQUIDITY PLAYERS]??? R^2 val.?

from sklearn.linear_model import LinearRegression

X = df.loc[(slice(None), 'Jadon Sancho'), :]['daily_returns'].dropna().values.reshape(-1,1)
y = df.loc[(slice(None), 'Jadon Sancho'), :]['daily_mkt_returns'].dropna().values.reshape(-1,1)
reg = LinearRegression().fit(X, y)





