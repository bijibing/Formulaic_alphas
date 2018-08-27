import csv
import time
import re
import pandas as pd
import datetime
import numpy as np
import os


# 0418-?
datasets_path = '/home/lp0477/pcshare2/datasets/'
name_list = []
for name in os.listdir(datasets_path):
    name_list.append(name)
name_list.sort()

#
index_list = ['type', 'ms', 'ticker', 'bid_price_0', 'bid_volume_0', 'bid_price_1', 'bid_volume_1','bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'bid_price_4', 'bid_volume_4', 'bid_price_5', 'bid_volume_5', 'bid_price_6', 'bid_volume_6', 'bid_price_7', 'bid_volume_7', 'bid_price_8', 'bid_volume_8', 'bid_price_9', 'bid_volume_9', \
                'ask_price_0', 'ask_volume_0', 'ask_price_1', 'ask_volume_1','ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'ask_price_4', 'ask_volume_4', 'ask_price_5', 'ask_volume_5', 'ask_price_6', 'ask_volume_6', 'ask_price_7', 'ask_volume_7', 'ask_price_8', 'ask_volume_8', 'ask_price_9', 'ask_volume_9', \
                'price', 'volume', 'turnover', 'close', 'open', 'low', 'high', 's', 'us', 'update_time', 'id']
time_str = [str(a)+':'+str(b)+':'+str(c) for a in range(9,12) for b in range(0, 60) for c in ['00', '30']][60:-60] + [str(a)+':'+str(b)+':'+str(c) for a in range(13,15) for b in range(0, 60) for c in ['00', '30']]

for file_name in name_list:
    high_p_df, low_p_df, open_p_df, close_p_df, volume_df_early, volume_df_late = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    combination = pd.DataFrame()

    # load_data for chunk
    df = pd.read_csv('/home/lp0477/pcshare2/datasets/%s'%file_name, chunksize=1000000, header=None, names=index_list, usecols=['type', 'ticker', 'price', 'volume', 's'])
    counter = 0

    # daily time_stamp
    day_time = int(file_name[6:8])
    month_time = int(file_name[4:6])
    time_start_1 = time.mktime((2018, month_time, day_time, 9, 30, 0, 0, 0, 0))
    time_end_1 = time.mktime((2018, month_time, day_time, 11, 30, 0, 0, 0, 0))
    time_start_2 = time.mktime((2018, month_time, day_time, 13, 0, 0, 0, 0, 0))
    time_end_2 = time.mktime((2018, month_time, day_time, 15, 0, 0, 0, 0, 0))
    time_start = time_start_1

    # type1=stocks
    for chunk in df:
        # SH_stocks
        data_SH = chunk.loc[(chunk['type'] == 1) & ((chunk['ticker'] >= 'SH600000') & (chunk['ticker'] < 'SH700000'))]
        # SZ_stocks
        data_SZ = chunk.loc[(chunk['type'] == 1) & (((chunk['ticker'] >= 'SZ000000') & (chunk['ticker'] < 'SZ100000')) | ((chunk['ticker'] >= 'SZ300000') & (chunk['ticker'] < 'SZ400000')))]
        data = pd.concat([combination, data_SH, data_SZ])
        # 1 min
        while time_start+30 <= chunk['s'].max():
            # del 11:30--13:00
            if time_start >=time_end_1 and time_start < time_start_2:
                time_start += 30
                continue
            # del 15:00--
            elif time_start >= time_end_2:
                break
            else:
                data_1m = data[(time_start <= data['s']) & (data['s'] < time_start + 30)]
                grouped = data_1m.groupby('ticker')    # group
                # get high, low, open, close of 30 seconds
                high_p, low_p, open_p, close_p = grouped['price'].max().to_frame(), grouped['price'].min().to_frame(), grouped['price'].first().to_frame(), grouped['price'].last().to_frame()
                high_p_df = pd.concat([high_p_df, high_p], axis=1, join='outer', sort=True)
                low_p_df = pd.concat([low_p_df, low_p], axis=1, join='outer', sort=True)
                open_p_df = pd.concat([open_p_df, open_p], axis=1, join='outer', sort=True)
                close_p_df = pd.concat([close_p_df, close_p], axis=1, join='outer', sort=True)
                # get volume of 30 seconds
                volume_early, volume_late = grouped['volume'].first().to_frame(), grouped['volume'].last().to_frame()
                volume_df_early = pd.concat([volume_df_early, volume_early], axis=1, join='outer', sort=True)
                volume_df_late = pd.concat([volume_df_late, volume_late], axis=1, join='outer', sort=True)

                time_start += 30

        else:
            # chunks concat according to 1_min
            counter += 1
            print(counter)      # process
            combination = data[(time_start <= data['s']) & (data['s'] < time_start + 30)]

    volume_df_late = volume_df_late.fillna(method='ffill', axis=1).fillna(0)
    volume_df_early = volume_df_early.fillna(method='ffill', axis=1).fillna(0)
    volume_df = volume_df_late.diff(axis=1)
    volume_df.iloc[:, [0, 240]] = volume_df_late.iloc[:, [0, 240]] - volume_df_early.iloc[:, [0, 240]]
    volume_df.columns = time_str

    high_p_df.columns, low_p_df.columns, open_p_df.columns, close_p_df.columns = time_str, time_str, time_str, time_str
    high_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/%s/high_1min.csv'%file_name)
    low_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/%s/low_1min.csv'%file_name)
    open_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/%s/open_1min.csv'%file_name)
    close_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/%s/close_1min.csv'%file_name)
    volume_df.T.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/volume/%s/volume_1min.csv' % file_name)
    #
    returns = (close_p_df - open_p_df)/open_p_df
    returns = returns.replace([np.inf, -np.inf], [0, 0])
    returns.columns = time_str
    returns.T.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/returns/%s_returns.csv'%file_name)
