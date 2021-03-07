# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:49:54 2020

@author: MikeyJW
"""

import pandas as pd

from config import MATCHDAY_SCORES, PLAYER_INFO, MODEL_DATA


def main():
    'Combines, cleans and writes raw Football Index Edge player data to single csv file'
    # Import from files
    df1 = pd.read_csv(MATCHDAY_SCORES)
    df1['Date'] = pd.to_datetime(df1['Date'], format='%d/%m/%Y')

    df2 = pd.read_csv(PLAYER_INFO)
    df2.set_index('PlayerName', inplace=True)

    # Extract average matchday scores and number of games played 
    season_start = '2020-09-12' # Start of 20/21 football season
    season_df = df1[df1['Date'] >= season_start].copy()

    ave_matchday_score = season_df.groupby('PlayerName')['MatchdayScore'].mean()
    num_games_played = season_df.groupby('PlayerName').size()

    # Change price col from string to float
    df2['CurrentPrice'] = df2['CurrentPrice'].str.strip('£').astype('float64')

    # Append new columns to player info
    df2['ave_matchday_score'] = ave_matchday_score
    df2['num_games_played'] = num_games_played

    # Extract useful columns into new dataframe
    df = df2[['Position', 'Age', 'num_games_played', 
              'ave_matchday_score',  'CurrentPrice']]

    # Generate position indicators for use in models
    df = pd.get_dummies(df, prefix='', prefix_sep='')

    # Slice off players where ave matchday score is unreliable
    final = df[df['num_games_played'] >= 5]

    # Export to csv
    final.to_csv(MODEL_DATA)




if __name__ == '__main__':
    
    main()
    