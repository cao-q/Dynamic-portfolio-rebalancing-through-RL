import numpy as np
import pandas as pd
import indicators
import config
import util
import sys
from pathlib import Path

def get_algo_dataset(choose_set_num: int):
    """ Returns df_list, date_range, trend_list, stocks
    """
    # Do not change run_set order. The order is hardcoded into below code
    run_set = ['portfolio1', 'portfolio2', 'portfolio3']
    choose_set = run_set[choose_set_num]
    df_list = []
    date_range = []
    trend_list = []
    stocks = []

    if choose_set == run_set[0]:
        stocks =['^BVSP', '^TWII', '^IXIC']
        for stock in stocks:
            df=pd.read_csv('data/rl/portfolio1/{}.csv'.format(stock), parse_dates=['Date'])
            df = df[df['Close'] > 0].reset_index(drop=True)
            df['returns'] = indicators.day_gain(df, 'Close').dropna()
            df_list.append(df)

        start = '1/1/2014'
        end = '31/12/2018'
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)

    elif choose_set == run_set[1]:
        stocks =['AXP', 'MCD', 'WMT']
        for stock in stocks:
            df=pd.read_csv('data/rl/portfolio2/{}.csv'.format(stock), parse_dates=['Date'])
            df = df[df['Close'] > 0].reset_index(drop=True)
            df['returns'] = indicators.day_gain(df, 'Close').dropna()
            df_list.append(df)

        start = '1/1/2014'
        end = '31/12/2018'
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)

    elif choose_set == run_set[2]:
        stocks =['UMBF', 'UNIT', 'FEYE']
        for stock in stocks:
            df=pd.read_csv('data/rl/portfolio3/{}.csv'.format(stock), parse_dates=['Date'])
            df = df[df['Close'] > 0].reset_index(drop=True)
            df['returns'] = indicators.day_gain(df, 'Close').dropna()
            df_list.append(df)

        start = '1/1/2014'
        end = '31/12/2018'
        date_range = remove_uncommon_dates(df_list)
        trend_list = util.get_trend_list(stocks, df_list, start=start, end=end)
    return df_list, date_range, trend_list, stocks


def remove_uncommon_dates(df_list):
    date_range = []
    for date in df_list[0]['Date']:
        empty = 0
        for df in df_list:
            temp_df = df[df['Date'] == date]
            if temp_df.empty:
                empty +=1
        if empty == 0:
            date_range.append(date)
    return date_range
