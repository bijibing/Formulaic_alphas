import csv
import time
import os
import pandas as pd
import datetime
import numpy as np


datasets_path = '/home/lp0477/pcshare2/datasets/'
name_list = []
for name in os.listdir(datasets_path):
    name_list.append(name)
name_list.sort()

index_list = ['type', 'ms', 'ticker', 'bid_price_0', 'bid_volume_0', 'bid_price_1', 'bid_volume_1','bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'bid_price_4', 'bid_volume_4', 'bid_price_5', 'bid_volume_5', 'bid_price_6', 'bid_volume_6', 'bid_price_7', 'bid_volume_7', 'bid_price_8', 'bid_volume_8', 'bid_price_9', 'bid_volume_9', \
                'ask_price_0', 'ask_volume_0', 'ask_price_1', 'ask_volume_1','ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'ask_price_4', 'ask_volume_4', 'ask_price_5', 'ask_volume_5', 'ask_price_6', 'ask_volume_6', 'ask_price_7', 'ask_volume_7', 'ask_price_8', 'ask_volume_8', 'ask_price_9', 'ask_volume_9', \
                'price', 'volume', 'turnover', 'close', 'open', 'low', 'high', 's', 'us', 'update_time', 'id']
time_str = [str(a)+':'+str(b)+':'+str(c) for a in range(9,12) for b in range(0, 60) for c in ['00', '30']][60:-60] + [str(a)+':'+str(b)+':'+str(c) for a in range(13,15) for b in range(0, 60) for c in ['00', '30']]

for file_name in name_list:
    vwap_df_early, vawp_df_late, vwap_df, volume_df_early, volume_df_late = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    combination = pd.DataFrame()

    # load_data for chunk
    df = pd.read_csv('/home/lp0477/pcshare2/datasets/%s'%file_name, chunksize=1000000, header=None, names=index_list, usecols=['type', 'ticker', 'price', 'volume', 's'])
    counter = 0

    # daily time_stamp
    day_time = int(file_name[6:8])
    month_time = int(file_name[4:6])
    time_start_1 = time.mktime((2018,month_time,day_time,9,30,0,0,0,0))
    time_end_1 = time.mktime((2018,month_time,day_time,11,30,0,0,0,0))
    time_start_2 = time.mktime((2018,month_time,day_time,13,0,0,0,0,0))
    time_end_2 = time.mktime((2018,month_time,day_time,15,0,0,0,0,0))
    # print(time_start_1, time_end_1)
    time_start = time_start_1

    # type1=stocks
    min_counter = 0
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
                min_counter += 1

                data_1m = data[(time_start <= data['s']) & (data['s'] < time_start + 30)]  # extract min_data ['type', 'ticker', 'price', 's']
                grouped = data_1m.groupby(data_1m['ticker'])  # group
                # indexs = data_1m['ticker'].drop_duplicates()
                # print(indexs.sort_values())

                volume_early, volume_late = grouped['volume'].first().to_frame(), grouped['volume'].last().to_frame()
                volume_df_early = pd.concat([volume_df_early, volume_early], axis=1, join='outer', sort=True)
                volume_df_early = volume_df_early.fillna(method='ffill', axis=1).fillna(0)
                volume_df_late = pd.concat([volume_df_late, volume_late], axis=1, join='outer', sort=True)
                volume_df_late = volume_df_late.fillna(method='ffill', axis=1).fillna(0)
                indexs = volume_df_late.index
                vwap_1min = pd.Series(index=indexs)
                if min_counter == 1 or min_counter == 241:
                    for share_name, group in grouped:
                        weight = group['volume'].diff().fillna(0)
                        if sum(weight) == 0:
                            vwap_1min[share_name] = np.nan
                        else:
                            vwap_1min[share_name] = np.average(group['price'], weights=weight)
                else:
                    # get high, low, open, close of 1 min
                    volume_min_add = volume_df_early.iloc[:, min_counter-1] - volume_df_late.iloc[:, min_counter-2]
                    volume_min_add[volume_min_add < 0] = 0
                    for share_name, group in grouped:
                        weight = group['volume'].diff().fillna(volume_min_add[share_name])
                        if sum(weight) == 0:
                            vwap_1min[share_name] = np.nan
                        else:
                            vwap_1min[share_name] = np.average(group['price'], weights=weight)

                vwap_df = pd.concat([vwap_df, vwap_1min.to_frame()], axis=1, join='outer', sort=True)
#                 print(vwap_df)
                time_start += 30

        else:
            counter += 1
            print(counter)      # process
            combination = data[(time_start <= data['s']) & (data['s'] < time_start + 30)]

    vwap_df = vwap_df.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    vwap_df.columns = time_str
    vwap_df.T.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/vwap/%s/vwap_1min.csv'%file_name)
