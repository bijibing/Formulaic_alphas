import pandas as pd
import numpy as np
import mpl_finance as mpf
import matplotlib.pyplot as plt


# # merge & drop
# data_high_t1 = pd.read_csv('/home/lp0477/pcshare/data_6_8/high_1min_t1.csv')
# data_high_t2 = pd.read_csv('/home/lp0477/pcshare/data_6_8/high_1min_t2.csv')
# # print(data_ip_t1.values[0], data_ip_t2.values[0])
# df_1 = data_high_t1.loc[(data_high_t1.iloc[:, 0] >= 'SH600000') & (data_high_t1.iloc[:, 0] < 'SH700000')]
# # print(len(data_ip_t1.values[:, 0]))
# df_2 = data_high_t1.loc[(data_high_t1.iloc[:, 0] >= 'SZ000000') & (data_high_t1.iloc[:, 0] < 'SZ100000')]
# df_3 = data_high_t1.loc[(data_high_t1.iloc[:, 0] >= 'SZ300000') & (data_high_t1.iloc[:, 0] < 'SZ400000')]
# df_4 = data_high_t2.loc[(data_high_t2.iloc[:, 0] >= 'SZ300000') & (data_high_t2.iloc[:, 0] < 'SZ400000')]
# # print(df_1.values.shape, df_2.values.shape, df_3.values.shape, df_4.values.shape)
# # data_ip = pd.concat([df_1, df_2, df_3, df_4], sort=True)
# data_high = np.concatenate((df_1, df_2, df_3, df_4))
# # print(data_high[:,0])
# data_high_op = pd.DataFrame(data_high[:, 1:], index=data_high[:, 0])
# data_high_op.to_csv('/home/lp0477/pcshare/data_6_8/high_1min_use.csv')
# # print(data_ip)
# # print(df_1)


# plot K_line
# get_data_high = pd.read_csv('/home/lp0477/pcshare/data_6_8/high_1min_use.csv', index_col=0)
# get_data_low = pd.read_csv('/home/lp0477/pcshare/data_6_8/low_1min_use.csv', index_col=0)
get_data_open = pd.read_csv('/home/lp0477/pcshare/data_6_8/open_1min_use.csv', index_col=0)
get_data_close = pd.read_csv('/home/lp0477/pcshare/data_6_8/close_1min_use.csv', index_col=0)


stock_name_all = get_data_open.index
# print(len(get_data_open.columns))
# time_stamp = np.arange(len(get_data_open.columns), dtype=float)
# # print(time_stamp)
# stock_name = stock_name_all[875]
# data_high = np.array(get_data_high.loc[stock_name])
# data_low = np.array(get_data_low.loc[stock_name])
# data_open = np.array(get_data_open.loc[stock_name])
# data_close = np.array(get_data_close.loc[stock_name])
# data_plot = zip(time_stamp, data_open, data_high, data_low, data_close)
# # print(data_plot.shape)
# # data_plot = np.vstack((time_stamp, data_open, data_high, data_low, data_close))
# # print(data_plot[4])
# # print(data_close)
# # print(get_data_high.loc[stock_name])
# # data_high = get_data_high.values[:, 1:]
# # data_low = get_data_low.values[:, 1:]
# # data_open = get_data_open.values[:, 1:]
# # data_close = get_data_close.values[:, 1:]
# # print(get_data_high[])
# # print(get_data_high.values.shape)
# # print(name_all, data_high.shape, data_close.shape, data_open.shape, data_close.shape)
#
#
# fig, ax = plt.subplots(figsize=(15, 5))
# fig.subplots_adjust(bottom=0.2)
# ax.grid(True)
# # ax.xaxis_date()
# plt.xlabel('time')
# plt.ylabel('price')
# mpf.candlestick_ohlc(ax, data_plot, width=0.2, colorup='r', colordown='g')
# plt.show()

data_returns_all = (get_data_close-get_data_open)/get_data_open
# print(data_returns_all)
# for i in range(len(stock_name_all)):
#     data_returns = ((get_data_close.iloc[i] - get_data_open.iloc[i])/get_data_open.iloc[i]).to_frame()
#     data_returns_all = pd.concat([data_returns_all, data_returns],axis=1)
#     print(data_returns_all)
# print(data_returns_all)
# data_returns_all.to_csv('/home/lp0477/pcshare/data_6_8/returns_1min_use.csv')
df = data_returns_all.fillna(0)
corr_mat = np.corrcoef(df.values)
print(pd.DataFrame(corr_mat)[pd.DataFrame(corr_mat).isnull().all()])