    # -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:02 2020

@author: MikeyJW
"""

import pandas as pd

from config import PRICING_CSV_LIST, PRICING_DATA


def main():
    'Combines, cleans and writes raw Football Index Edge pricing data to single csv file'
    # Initialise dataframe and concat csv's
    df = pd.DataFrame()

    for filepath in PRICING_CSV_LIST:
        temp = pd.read_csv(filepath, low_memory=False)
        df = pd.concat([df, temp])

    # Convert string to date column
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Convert prices from string to float
    col_list = ['EndofDayPrice', 'MESellPrice',
                'MatchdayDividends', 'MediaDividends']
    for col in col_list:
        df[col] = df[col].str.strip('£').astype('float64')

    # Re-index
    df.set_index(['Date', 'PlayerName'], inplace=True)
    df.sort_index(level=0, inplace=True)

    # Spread average market price and market returns across all players
    df['daily_returns'] = df['EndofDayPrice'].groupby(level=1).pct_change()

    mkt_price = df['EndofDayPrice'].groupby(level=0).mean()
    mkt_returns = df['daily_returns'].groupby(level=0).mean()

    df['ave_mkt_price'] = mkt_price.reindex(df.index, level=0)
    df['daily_mkt_returns'] = mkt_returns.reindex(df.index, level=0)

    # Write to csv
    df.to_csv(PRICING_DATA)

    
    
    
if __name__ == '__main__':
    
    main()
    