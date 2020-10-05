    # -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:02 2020

@author: MikeyJW
"""

import pandas as pd

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
mkt_returns = df['daily_returns'].groupby(level=0).mean() # ignore index???

df['ave_mkt_price'] = mkt_price.reindex(df.index, level=0)
df['daily_mkt_returns'] = mkt_returns.reindex(df.index, level=0)

# Save dataframe
#df.to_csv('out.csv')
#############################################################
















