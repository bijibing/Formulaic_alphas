import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as mpf
from matplotlib.pylab import date2num


# one stock
get_data_high = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/20180702/high_1min.csv', index_col=0)
get_data_low = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/20180702/low_1min.csv', index_col=0)
get_data_open = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/20180702/open_1min.csv', index_col=0)
get_data_close = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/20180702/close_1min.csv', index_col=0)

t = [i for i in range(len(get_data_open.columns))]
# stock_name_all = get_data_high.index
# stock_name = stock_name_all[1874]
stock_name = 'SH600000'
# print(stock_name)
data_high = get_data_high[stock_name].values
data_low = get_data_low[stock_name].values
data_open = get_data_open[stock_name].values
data_close = get_data_close[stock_name].values
data_plot = zip(t, data_open, data_high, data_low, data_close)
# print(data_plot)
fig, ax1 = plt.subplots(figsize=(30, 6))
fig.subplots_adjust(bottom=0.2)
ax1.set_title(stock_name)
ax1.set_ylabel('Price')
ax1.grid(True)
mpf.candlestick_ohlc(ax1, data_plot, width=0.5, colorup='r', colordown='g')
plt.show()


# # two or more stocks
# get_data_high = pd.read_csv('/home/lp0477/pcshare/ticker_data/2018_6_8/ohlc/high_1min.csv', index_col=0)
# get_data_low = pd.read_csv('/home/lp0477/pcshare/ticker_data/2018_6_8/ohlc/low_1min.csv', index_col=0)
# get_data_open = pd.read_csv('/home/lp0477/pcshare/ticker_data/2018_6_8/ohlc/open_1min.csv', index_col=0)
# get_data_close = pd.read_csv('/home/lp0477/pcshare/ticker_data/2018_6_8/ohlc/close_1min.csv', index_col=0)
#
# t = [i for i in range(len(get_data_open.columns))]
# # stock_name_all = get_data_high.index
# # stock_name = stock_name_all[1874]
# stock_name_1 = 'SH600354'
# stock_name_2 = 'SZ002657'
# # print(stock_name)
# data_high_1 = get_data_high.loc[stock_name_1].values
# data_low_1 = get_data_low.loc[stock_name_1].values
# data_open_1 = get_data_open.loc[stock_name_1].values
# data_close_1 = get_data_close.loc[stock_name_1].values
# data_plot_1 = zip(t, data_open_1, data_high_1, data_low_1, data_close_1)
#
# data_high_2 = get_data_high.loc[stock_name_2].values
# data_low_2 = get_data_low.loc[stock_name_2].values
# data_open_2 = get_data_open.loc[stock_name_2].values
# data_close_2 = get_data_close.loc[stock_name_2].values
# data_plot_2 = zip(t, data_open_2, data_high_2, data_low_2, data_close_2)
# # print(data_plot)
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(30, 12))
# fig.subplots_adjust(bottom=0.2)
# ax1.set_title(stock_name_1)
# ax1.set_ylabel('Price')
# ax1.grid(True)
# mpf.candlestick_ohlc(ax1, data_plot_1, width=0.5, colorup='r', colordown='g' )
#
# ax2.set_title(stock_name_2)
# ax2.set_ylabel('Price')
# ax2.grid(True)
# mpf.candlestick_ohlc(ax2, data_plot_2, width=0.5, colorup='r', colordown='g' )
# plt.show()
