import csv
import time
import re
import pandas as pd
import datetime
import gc


index_list_1 = ['type', 'ms', 'ticker', 'bid_price_0', 'bid_volume_0', 'bid_price_1', 'bid_volume_1','bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'bid_price_4', 'bid_volume_4', 'bid_price_5', 'bid_volume_5', 'bid_price_6', 'bid_volume_6', 'bid_price_7', 'bid_volume_7', 'bid_price_8', 'bid_volume_8', 'bid_price_9', 'bid_volume_9', \
                'ask_price_0', 'ask_volume_0', 'ask_price_1', 'ask_volume_1','ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'ask_price_4', 'ask_volume_4', 'ask_price_5', 'ask_volume_5', 'ask_price_6', 'ask_volume_6', 'ask_price_7', 'ask_volume_7', 'ask_price_8', 'ask_volume_8', 'ask_price_9', 'ask_volume_9', \
                'price', ' volume', 'turnover', 'close', 'open', 'low', 'high', 's', 'us', 'update_time', 'id']
index_list_2 = ['type', 'ms', 'ticker', 'price', ' volume', 'turnover', 'close', 'open', 'low', 'high', 's', 'us', 'update_time', 'id', \
                'bid_price_0', 'bid_volume_0', 'bid_price_1', 'bid_volume_1','bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'bid_price_4', 'bid_volume_4', 'bid_price_5', 'bid_volume_5', 'bid_price_6', 'bid_volume_6', 'bid_price_7', 'bid_volume_7', 'bid_price_8', 'bid_volume_8', 'bid_price_9', 'bid_volume_9', \
                'ask_price_0', 'ask_volume_0', 'ask_price_1', 'ask_volume_1','ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'ask_price_4', 'ask_volume_4', 'ask_price_5', 'ask_volume_5', 'ask_price_6', 'ask_volume_6', 'ask_price_7', 'ask_volume_7', 'ask_price_8', 'ask_volume_8', 'ask_price_9', 'ask_volume_9']
# index_list = [i for i in range(54)]
# index_list_1 = [0, 2, 43, 50]
# index_list_2 = [0, 2, 3, 10]
time_str = [str(a)+':'+str(b) for a in range(9,12) for b in range(0, 60)][30:-30] + [str(a)+':'+str(b) for a in range(13,15) for b in range(0, 60)]

high_p_df_1, low_p_df_1, open_p_df_1, close_p_df_1 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
combination_1 = pd.DataFrame()
high_p_df_2, low_p_df_2, open_p_df_2, close_p_df_2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
combination_2 = pd.DataFrame()
# print(returns_df)
# df = pd.read_csv('/home/lp0477/pcshare/data_6_8/20180608', chunksize=100000)
# df = pd.read_csv('/home/lp0477/pcshare/data_6_8/20180608', chunksize=1000000, header=None, usecols=index_list)
df_1 = pd.read_csv('/home/lp0477/pcshare/data_6_8/20180608', chunksize=1000000, header=None, names=index_list_1, usecols=['type', 'ticker', 'price', 's'])
df_2 = pd.read_csv('/home/lp0477/pcshare/data_6_8/20180608', chunksize=1000000, header=None, names=index_list_2, usecols=['type', 'ticker', 'price', 's'])
# start = datetime.datetime.now()
counter = 0
# time_data = df.get_chunk(1)

# time_cut daily
time_start_1 = time.mktime((2018,6,8,9,30,0,0,0,0))
time_end_1 = time.mktime((2018,6,8,11,30,0,0,0,0))
time_start_2 = time.mktime((2018,6,8,13,0,0,0,0,0))
time_end_2 = time.mktime((2018,6,8,15,0,0,0,0,0))

start_1 = time_start_1
start_2 = time_start_1


for chunk_1 in df_1:
    # print(chunk_1['s'].max())
    # print(chunk_1['s'].iloc[-1])
    data_1_SH = chunk_1.loc[(chunk_1['type'] == 1) & ((chunk_1['ticker'] >= 'SH600000') & (chunk_1['ticker'] < 'SH700000'))]
    data_1_SZ = chunk_1.loc[(chunk_1['type'] == 1) & (((chunk_1['ticker'] >= 'SZ000000') & (chunk_1['ticker'] < 'SZ100000')) | ((chunk_1['ticker'] >= 'SZ300000') & (chunk_1['ticker'] < 'SZ400000')))]
    # (((chunk['s'] >= time_start_1) & (chunk['s'] <= time_end_1)) | ((chunk['s'] >= time_start_2) & (chunk['s'] <= time_end_2)))
    data_1 = pd.concat([combination_1, data_1_SZ, data_1_SH])
    # print(data_1)
    while start_1+60 <= chunk_1['s'].max():
        if start_1 >=time_end_1 and start_1 < time_start_2:
            start_1 += 60
            continue
        elif start_1 >= time_end_2:
            break
        else:
            data_1m = data_1[(start_1 <= data_1['s']) & (data_1['s'] < start_1 + 60)]
            grouped_1 = data_1m.groupby(data_1m['ticker'])
            high_p_1, low_p_1, open_p_1, close_p_1 = grouped_1['price'].max().to_frame(), grouped_1['price'].min().to_frame(), grouped_1['price'].first().to_frame(), grouped_1['price'].last().to_frame()
            high_p_df_1 = pd.concat([high_p_df_1, high_p_1], axis=1, join='outer', sort=True)
            low_p_df_1 = pd.concat([low_p_df_1, low_p_1], axis=1, join='outer', sort=True)
            open_p_df_1 = pd.concat([open_p_df_1, open_p_1], axis=1, join='outer', sort=True)
            close_p_df_1 = pd.concat([close_p_df_1, close_p_1], axis=1, join='outer', sort=True)
            # print(high_p_df_1)
            start_1 += 60
    else:
        counter += 1
        print(counter)
        combination_1 = data_1[(start_1 <= data_1['s']) & (data_1['s'] < start_1 + 60)]
# print(high_p_df_1)
del df_1, data_1, data_1_SH, data_1_SZ
gc.collect()


for chunk_2 in df_2:
    # print(chunk_2)
    data_2_SZ = chunk_2.loc[(chunk_2['type'] == 2) & ((chunk_2['ticker'] >= 'SZ300000') & (chunk_2['ticker'] < 'SZ400000'))]
    data_2 = pd.concat([combination_2, data_2_SZ])
    while start_2+60 <= chunk_2['s'].max():
        if start_2 >= time_end_1 and start_2 < time_start_2:
            start_2 += 60
            continue
        elif start_2 >= time_end_2:
            break
        else:
            data_2m = data_2[(start_2 <= data_2['s']) & (data_2['s'] < start_2 + 60)]
            # print(data_2m)
            grouped_2 = data_2m.groupby(data_2m['ticker'])
            high_p_2, low_p_2, open_p_2, close_p_2 = grouped_2['price'].max().to_frame(), grouped_2['price'].min().to_frame(), grouped_2['price'].first().to_frame(), grouped_2['price'].last().to_frame()
            # print(high_p_2)
            high_p_df_2 = pd.concat([high_p_df_2, high_p_2], axis=1, join='outer', sort=True)
            low_p_df_2 = pd.concat([low_p_df_2, low_p_2], axis=1, join='outer', sort=True)
            open_p_df_2 = pd.concat([open_p_df_2, open_p_2], axis=1, join='outer', sort=True)
            close_p_df_2 = pd.concat([close_p_df_2, close_p_2], axis=1, join='outer', sort=True)
            start_2 += 60
            # print(high_p_df_2)
    else:
        counter += 1
        print(counter)
        combination_2 = data_2[(start_2 <= data_2['s']) & (data_2['s'] < start_2 + 60)]
# print(high_p_df_2)

high_p_df = pd.concat([high_p_df_1, high_p_df_2], names=time_str, join='inner', sort=True)
low_p_df = pd.concat([low_p_df_1, low_p_df_2], names=time_str, join='inner', sort=True)
open_p_df=pd.concat([open_p_df_1, open_p_df_2], names=time_str, join='inner', sort=True)
close_p_df=pd.concat([close_p_df_1, close_p_df_2], names=time_str, join='inner', sort=True)
#
high_p_df.to_csv('/home/lp0477/pcshare/data_6_8/normal/high_1min.csv')
low_p_df.to_csv('/home/lp0477/pcshare/data_6_8/normal/low_1min.csv')
open_p_df.to_csv('/home/lp0477/pcshare/data_6_8/normal/open_1min.csv')
close_p_df.to_csv('/home/lp0477/pcshare/data_6_8/normal/close_1min.csv')
# # high_p_df_2.to_csv('/home/lp0477/pcshare/data_6_8/high_1min_t2.csv')
# # low_p_df_2.to_csv('/home/lp0477/pcshare/data_6_8/low_1min_t2.csv')
# # open_p_df_2.to_csv('/home/lp0477/pcshare/data_6_8/open_1min_t2.csv')
# # close_p_df_2.to_csv('/home/lp0477/pcshare/data_6_8/close_1min_t2.csv')
