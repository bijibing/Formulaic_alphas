import pandas as pd
import struct
import lzma
import gc
import numpy as np
import os
import codecs


# file_path = '/home/lp0477/pcshare/ticker_data/add_data/'
# name_list = []
# # f_name = open('/home/lp0477/pcshare/ticker_data/2018_6_8/name', 'w+')
# for name in os.listdir(file_path):
#     name_list.append(name.split('.')[0])
# name_list.sort()
# # print(name_list)

name_list = ['20180102']
for file_name in name_list:
    # print(file_name)
    f = open('/home/lp0477/pcshare2/data_zip/%s.hq'%file_name, mode='rb')
    day_data = []
    try:
        while True:
            chunk = f.read(314)
            if not chunk:
                break
            stock_ticker = struct.unpack_from('2s', chunk, offset=38)[0].decode() + struct.unpack_from('6s', chunk, offset=0)[0].decode()
            trade_time = struct.unpack_from('i', chunk, offset=302)[0]//1000
            price = struct.unpack_from('i', chunk, offset=46)[0]/10000
            volume = struct.unpack_from('q', chunk, offset=70)[0]
            turnover = struct.unpack_from('q', chunk, offset=78)[0]
            # day_data.append([stock_ticker, price, volume, turnover])
            print([stock_ticker, price, turnover, volume])
            # break
            # print(pd.DataFrame(day_data, columns=['ticker', 'time', 'price', 'volume']).dtypes)
    finally:
        f.close()

    # # data_one = pd.DataFrame(day_data, columns=['ticker', 'time', 'price', 'volume'])
    # data_one = pd.read_csv('/home/lp0477/pcshare/ticker_data/0102_0404/%s.csv'%file_name, index_col=0)
    # # del day_data
    # # gc.collect()
    # # # print(data_one['volume'][data_one['ticker']=='SH600000'])
    # #
    # data_one = data_one[(((data_one['time']>=93000) & (data_one['time']<113000)) | ((data_one['time'] >= 130000) & (data_one['time']<150000)))]
    # data_one_SH = data_one[((data_one['ticker']>='SH600000') & (data_one['ticker']<='SH700000'))]
    # data_one_SZ = data_one[(((data_one['ticker']>='SZ000000') & (data_one['ticker']<='SZ100000')) | ((data_one['ticker']>= 'SZ300000') & (data_one['ticker']<='SZ400000')))]
    # data_one = pd.concat([data_one_SH, data_one_SZ], axis=0, sort=False)
    #
    # time_str = [str(a)+':'+str(b) for a in range(9, 12) for b in range(0, 60)][30:-30] + [str(a)+':'+str(b) for a in range(13,15) for b in range(0, 60)]
    # time_start = 93000
    #
    #
    # high_p_df, low_p_df, open_p_df, close_p_df, volume_df_early, volume_df_late, vwap_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # for i in range(240):
    #     print(i)
    #     if i < 30:
    #         data_1min = data_one[(data_one['time'] >= time_start+i*100) & (data_one['time']<time_start+100*(i+1))]
    #     elif 30 <= i < 90:
    #         data_1min = data_one[(data_one['time'] >= time_start+4000+i*100) & (data_one['time']<time_start+4000+100*(i+1))]
    #     elif 90 <= i < 120:
    #         data_1min = data_one[(data_one['time'] >= time_start+8000+i*100) & (data_one['time']<time_start+8000+100*(i+1))]
    #     elif 120 <= i < 180:
    #         data_1min = data_one[(data_one['time'] >= time_start+25000+i*100) & (data_one['time']<time_start+25000+100*(i+1))]
    #     else:
    #         data_1min = data_one[(data_one['time'] >= time_start+29000+i*100) & (data_one['time']<time_start+29000+100*(i+1))]
    #     grouped = data_1min.groupby('ticker')
    #     high_p, low_p, open_p, close_p = grouped['price'].max().to_frame(), grouped['price'].min().to_frame(), grouped['price'].first().to_frame(), grouped['price'].last().to_frame()
    #     high_p_df = pd.concat([high_p_df, high_p], axis=1, join='outer', sort=True)
    #     low_p_df = pd.concat([low_p_df, low_p], axis=1, join='outer', sort=True)
    #     open_p_df = pd.concat([open_p_df, open_p], axis=1, join='outer', sort=True)
    #     close_p_df = pd.concat([close_p_df, close_p], axis=1, join='outer', sort=True)
    #     volume_early, volume_late = grouped['volume'].first().to_frame(), grouped['volume'].last().to_frame()
    #     volume_df_early = pd.concat([volume_df_early, volume_early], axis=1, join='outer', sort=True)
    #     # volume_df_early = volume_df_early.fillna(method='ffill', axis=1).fillna(0)
    #     volume_df_late = pd.concat([volume_df_late, volume_late], axis=1, join='outer', sort=True)
    #     # volume_df_late = volume_df_late.fillna(method='ffill', axis=1).fillna(0)
    #
    # volume_df_late = volume_df_late.fillna(method='ffill', axis=1).fillna(0)
    # volume_df_early = volume_df_early.fillna(method='ffill',axis=1).fillna(0)
    # volume_df = volume_df_late.diff(axis=1)
    # volume_df.iloc[:,[0, 120]] = volume_df_late.iloc[:,[0, 120]]-volume_df_early.iloc[:,[0, 120]]
    #
    # returns = (close_p_df-open_p_df)/open_p_df
    # returns = returns.replace([np.inf, -np.inf], [0, 0])
    # returns.columns, high_p_df.columns, low_p_df.columns, open_p_df.columns, close_p_df.columns, volume_df.columns = time_str, time_str, time_str, time_str, time_str, time_str
    # returns.T.to_csv('/home/lp0477/pcshare/ticker_data/returns/%s_returns.csv'%file_name)
    # high_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/ohlc/%s/high_1min.csv'%file_name)
    # low_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/ohlc/%s/low_1min.csv'%file_name)
    # open_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/ohlc/%s/open_1min.csv'%file_name)
    # close_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/ohlc/%s/close_1min.csv'%file_name)
    # volume_df.T.to_csv('/home/lp0477/pcshare/ticker_data/volume/%s/volume_1min.csv'%file_name)
    # # returns.T.to_csv('/home/lp0477/pcshare/ticker_data/test/20180102_returns.csv')
    # # high_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/test/high_1min.csv')
    # # low_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/test/low_1min.csv')
    # # open_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/test/open_1min.csv')
    # # close_p_df.T.to_csv('/home/lp0477/pcshare/ticker_data/test/close_1min.csv')
    # # volume_df.T.to_csv('/home/lp0477/pcshare/ticker_data/test/volume_1min.csv')
    # # grouped = data_one.groupby('ticker').diff()
