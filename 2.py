import csv
import time
import re
import pandas as pd
import datetime


index_list_1 = ['type', 'ms', 'ticker', 'bid_price_0', 'bid_volume_0', 'bid_price_1', 'bid_volume_1','bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'bid_price_4', 'bid_volume_4', 'bid_price_5', 'bid_volume_5', 'bid_price_6', 'bid_volume_6', 'bid_price_7', 'bid_volume_7', 'bid_price_8', 'bid_volume_8', 'bid_price_9', 'bid_volume_9', \
                'ask_price_0', 'ask_volume_0', 'ask_price_1', 'ask_volume_1','ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'ask_price_4', 'ask_volume_4', 'ask_price_5', 'ask_volume_5', 'ask_price_6', 'ask_volume_6', 'ask_price_7', 'ask_volume_7', 'ask_price_8', 'ask_volume_8', 'ask_price_9', 'ask_volume_9', \
                'price', ' volume', 'turnover', 'close', 'open', 'low', 'high', 's', 'us', 'update_time', 'id']
index_list_2 = ['type', 'ms', 'ticker', 'price', ' volume', 'turnover', 'close', 'open', 'low', 'high', 's', 'us', 'update_time', 'id', \
                'bid_price_0', 'bid_volume_0', 'bid_price_1', 'bid_volume_1','bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'bid_price_4', 'bid_volume_4', 'bid_price_5', 'bid_volume_5', 'bid_price_6', 'bid_volume_6', 'bid_price_7', 'bid_volume_7', 'bid_price_8', 'bid_volume_8', 'bid_price_9', 'bid_volume_9', \
                'ask_price_0', 'ask_volume_0', 'ask_price_1', 'ask_volume_1','ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'ask_price_4', 'ask_volume_4', 'ask_price_5', 'ask_volume_5', 'ask_price_6', 'ask_volume_6', 'ask_price_7', 'ask_volume_7', 'ask_price_8', 'ask_volume_8', 'ask_price_9', 'ask_volume_9']
# index_list = [i for i in range(54)]
# index_list_1 = [0, 2, 43, 50]
# index_list_2 = [0, 2, 3, 10]

high_p_df_1, low_p_df_1, open_p_df_1, close_p_df_1 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
combination_1 = pd.DataFrame()
# high_p_df_2, low_p_df_2, open_p_df_2, close_p_df_2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
# combination_2 = pd.DataFrame()
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

time_start = time_start_1
for chunk in df_1:

    data_1 = chunk.loc[(chunk[0] == 1) & (((chunk[50]>=time_start_1)& (chunk[50]<=time_end_1))| ((chunk[50]>=time_start_2)&(chunk[50]<=time_end_2)))]
    # & (((chunk[2] >= 'SH600000') & (chunk[2] <= 'SH700000')) | ((chunk[2]) >= 'SZ000000') & (chunk[2] <= 'SZ100000'))#
    print(data_1)
    break
#     data_1 = pd.concat([combination_1, data_1])
#     # print(data_1.head(10))
#     # print(data_1.values.shape)
#     # data_2 = chunk.loc[chunk[0] == 2]
#     # data_2 = pd.concat([combination_2, data_2])
#     # print(data_2.values.shape)
#     while time_start+60 <= chunk[50].iloc[-1]:
#         # # print(data_1[1])
#         data_1m = data_1[(time_start <= data_1[50]) & (data_1[50] < time_start + 60)]
#         if data_1m.empty:
#             time_start += 60
#             continue
#         else:
#             grouped_1 = data_1m.groupby(data_1m[2])
#             high_p_1, low_p_1, open_p_1, close_p_1 = grouped_1[43].max().to_frame(), grouped_1[43].min().to_frame(), grouped_1[43].first().to_frame(), grouped_1[43].last().to_frame()
#             # close_p_1, open_p_1 = grouped_1[46].last(), grouped_1[47].first()
#             # returns_1 = (close_p_1-open_p_1)/open_p_1
#             high_p_df_1 =pd.concat([high_p_df_1, high_p_1], axis=1, join='outer', sort=True)
#             low_p_df_1 =pd.concat([low_p_df_1, low_p_1], axis=1, join='outer', sort=True)
#             open_p_df_1 =pd.concat([open_p_df_1, open_p_1], axis=1, join='outer', sort=True)
#             close_p_df_1 =pd.concat([close_p_df_1, close_p_1], axis=1, join='outer', sort=True)
#             # returns_1 = pd.DataFrame(returns_1)
#             # returns_df_1 = pd.concat([returns_df_1, returns_1], axis=1, join='outer')
#             # print(price_df_1)
#             # print(close_p.loc['SZ300693'])
#             # print(type(close_p))
#             # grouped = data_1.groupby(data_1[2])
#             # print(grouped[1528439848190 <= grouped[1] < 1528439848250])
#
#         # data_2m = data_2[(time_start <= data_2[10]) & (data_2[10] < time_start + 60)]
#         # # print(data_2m)
#         # if data_2m.empty:
#         #     time_start += 60
#         #     continue
#         # else:
#         #     grouped_2 = data_2m.groupby(data_2m[2])
#         #     high_p_2, low_p_2, open_p_2, close_p_2 = grouped_2[3].max().to_frame(), grouped_2[3].min().to_frame(), grouped_2[3].first().to_frame(), grouped_2[3].last().to_frame()
#         #     high_p_df_2 = pd.concat([high_p_df_2, high_p_2], axis=1, join='outer', sort=True)
#         #     low_p_df_2 = pd.concat([low_p_df_2, low_p_2], axis=1, join='outer', sort=True)
#         #     open_p_df_2 = pd.concat([open_p_df_2, open_p_2], axis=1, join='outer', sort=True)
#         #     close_p_df_2 = pd.concat([close_p_df_2, close_p_2], axis=1, join='outer', sort=True)
#         #
#             time_start += 60
#     else:
#         counter += 1
#         print(counter)
#         # combination_1 = data_1[(time_start <= data_1[50]) & (data_1[50] < time_start + 60)]
#         combination_2 = data_2[(time_start <= data_2[10]) & (data_2[10] < time_start + 60)]
#         # print(combination_1.tail(10))
#         # grouped_1 = data_1m.groupby(data_1m[2])
#         # close_p_1, open_p_1 = grouped_1[46].last(), grouped_1[47].first()
#
#     # print(datetime.datetime.now()-start)
#
# high_p_df_1.to_csv('/home/lp0477/pcshare/data_6_8/high_1min_t1.csv')
# low_p_df_1.to_csv('/home/lp0477/pcshare/data_6_8/low_1min_t1.csv')
# open_p_df_1.to_csv('/home/lp0477/pcshare/data_6_8/open_1min_t1.csv')
# close_p_df_1.to_csv('/home/lp0477/pcshare/data_6_8/close_1min_t1.csv')
# # high_p_df_2.to_csv('/home/lp0477/pcshare/data_6_8/high_1min_t2.csv')
# # low_p_df_2.to_csv('/home/lp0477/pcshare/data_6_8/low_1min_t2.csv')
# # open_p_df_2.to_csv('/home/lp0477/pcshare/data_6_8/open_1min_t2.csv')
# # close_p_df_2.to_csv('/home/lp0477/pcshare/data_6_8/close_1min_t2.csv')
