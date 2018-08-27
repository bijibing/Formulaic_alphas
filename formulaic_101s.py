"""
101s 
"""
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import datetime
from sklearn.preprocessing import normalize
import os
import gc



close = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/close_0418_0702_30s.csv', index_col=0, iterator=True)
high = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/high_0418_0702_30s.csv', index_col=0, iterator=True)
open_d = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/open_0418_0702_30s.csv', index_col=0, iterator=True)
low = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/low_0418_0702_30s.csv', index_col=0, iterator=True)
volume = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/volume/volume_0418_0702_30s.csv', index_col=0, iterator=True)
vwap = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/vwap/vwap_0418_0702_30s.csv', index_col=0, iterator=True)
returns = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/returns/returns_0418_0702_30s.csv', index_col=0, iterator=True)
time_str = [str(a)+':'+str(b)+':'+str(c) for a in range(9,12) for b in range(0, 60) for c in ['00', '30']][61:-60] + [str(a)+':'+str(b)+':'+str(c) for a in range(13,15) for b in range(0, 60) for c in ['00', '30']][:-10]
close_1, open_1, low_1, high_1, volume_1, vwap_1, return_1 = pd.DataFrame(index=time_str), pd.DataFrame(index=time_str), pd.DataFrame(index=time_str), pd.DataFrame(index=time_str), pd.DataFrame(index=time_str), pd.DataFrame(index=time_str), pd.DataFrame(index=time_str)

"""
def decay_linear(na):
    decay_weights = np.arange(1, len(na)+1, 1)
    # decay_weights /= decay_weights.sum()
    # return (na*decay_weights).sum()
    return np.average(na, weights=decay_weights)
"""

def decay_linear(na, n):
    decay_weights = np.arange(1, n + 1, 1)
    return na.rolling(n).apply(lambda x:np.average(x, weights=decay_weights), raw=True)
"""
def current_rk(na):
    return na.rank(axis=0, pct=True).iloc[-1]
"""

def Ts_rank(na, n):
    return na.rolling(n).apply(lambda x:x.rank(axis=0, pct=True).iloc[-1], raw=False)

def correlation(na, nb, n):
    return na.rolling(n).corr(nb, pairwise=False)

def SignedPower(na, n):
    return na**n*np.sign(na)

def Ts_ArgMax(na, n):
    return na.rolling(n).apply(lambda x:x.idxmax(skipna=True), raw=False)


start = datetime.datetime.now()
loop = True
chunksize = 469
while loop:
    try:
        close_2 = close.get_chunk(chunksize)
        data_c = pd.concat([close_1, close_2], join='outer', sort=True)
        open_2 = open_d.get_chunk(chunksize)
        data_o = pd.concat([open_1, open_2], join='outer', sort=False)
        high_2 = high.get_chunk(chunksize)
        data_h = pd.concat([high_1, high_2], join='outer', sort=False)
        low_2 = low.get_chunk(chunksize)
        data_l = pd.concat([low_1, low_2], join='outer', sort=False)
        volume_2 = volume.get_chunk(chunksize)
        data_vo = pd.concat([volume_1, volume_2], join='outer', sort=False)
        vwap_2 = vwap.get_chunk(chunksize)
        data_vw = pd.concat([vwap_1, vwap_2], join='outer', sort=False)
        return_2 = returns.get_chunk(chunksize)
        data_r = pd.concat([return_1, return_2], join='outer', sort=False)
        data_dv = data_vw*data_vo
        date_d = close_2.index[0].split(' ')[0]
        print(date_d)

        # alpha_1 runned
        """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)"""
        if date_d == '20180418':
            alpha_1 = pd.DataFrame(columns=close_2.columns)
            alpha_1.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_1/1.csv')
        condition = data_r < 0
        indicator1 = data_r.rolling(20).std()
        indicator2 = data_c
        # (returns < 0) ? stddev(returns, 20): close
        part1 = indicator1[condition].fillna(0) + indicator2[~condition].fillna(0)
        part2 = part1 ** 2 * np.sign(part1)                             # SignedPower
        part3 = Ts_ArgMax(part2, 5)
        alpha_1 = part3.rank(axis=1, pct=True).iloc[469:] - 0.5
        alpha_1.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_1/1.csv', header=None,mode='a')

        # alpha_2 runned
        """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
        if date_d == '20180418':
            alpha_2 = pd.DataFrame(columns=close_2.columns)
            alpha_2.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_2/2.csv')
        part1 = np.log(data_vo).diff(2).rank(axis=1, pct=True)  # rank(delta(log(volume), 2))
        part2 = ((data_c-data_o)/data_o).rank(axis=1, pct=True)  # rank(((close - open) / open))
        alpha_2 = -1*correlation(part1, part2, 6).iloc[469:]
        alpha_2.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_2/2.csv', header=None, mode='a')

        # alpha_3 runned
        """(-1 * correlation(rank(open), rank(volume), 10))"""
        if date_d == '20180418':
            alpha_3 = pd.DataFrame(columns=close_2.columns)
            alpha_3.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_3/3.csv')
        part1 = data_o.rank(axis=1, pct=True)               # rank(open)
        part2 = data_vo.rank(axis=1, pct=True)              # rank(volume)
        alpha_3 = (-1 * correlation(part1, part2, 10)).iloc[469:]
        alpha_3.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_3/3.csv', header=None, mode='a')

        # alpha_4 runned
        """(-1 * Ts_Rank(rank(low), 9))"""
        if date_d == '20180418':
            alpha_4 = pd.DataFrame(columns=close_2.columns)
            alpha_4.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_4/4.csv')
        part1 = data_l.rank(axis=1, pct=True)               # rank(low)
        alpha_4 = (-1 * Ts_rank(part1, 9)).iloc[469:]
        alpha_4.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_4/4.csv', header=None,mode='a')

        # alpha_5 runned
        """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
        if date_d == '20180418':
            alpha_5 = pd.DataFrame(columns=close_2.columns)
            alpha_5.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_5/5.csv')
        part1 = (data_o-data_vw).rolling(10).mean().rank(axis=1, pct=True)     # rank((open - (sum(vwap, 10) / 10)))
        part2 = (data_c-data_vw).rank(axis=1, pct=True)                     # abs(rank((close - vwap)))
        alpha_5 = (-1*part1*part2).iloc[469:]
        alpha_5.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_5/5.csv', header=None,mode='a')


        # alpha_6 runned
        """(-1 * correlation(open, volume, 10))"""
        if date_d == '20180418':
            alpha_6 = pd.DataFrame(columns=close_2.columns)
            alpha_6.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_6/6.csv')
        alpha_6 = (-1*correlation(data_o, data_vo, 10)).iloc[469:]
        alpha_6.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_6/6.csv', header=None,mode='a')



        # alpha_7 runned
        """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) *sign(delta(close, 7))): (-1* 1))"""
        if date_d == '20180418':
            alpha_7 = pd.DataFrame(columns=close_2.columns)
            alpha_7.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_7/7.csv')
        condition = data_dv.rolling(20).mean() < data_vo
        part1 = Ts_rank(abs(data_c.diff(7)), 60)                        # ts_rank(abs(delta(close, 7)), 60)
        indicator1 = (-1 * part1*np.sign(data_c.diff(7)))
        indicator2 = -1 * pd.DataFrame(np.ones(data_c.shape), index=data_c.index, columns=data_c.columns).fillna(0)
        alpha_7 = indicator1[condition].fillna(0).iloc[469:] + indicator2[~condition].fillna(0).iloc[469:]
        alpha_7.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_7/7.csv', header=None,mode='a')


        # alpha_8 runned
        """(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))"""
        if date_d == '20180418':
            alpha_8 = pd.DataFrame(columns=close_2.columns)
            alpha_8.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_8/8.csv')
        part1 = data_o.rolling(5).sum() * data_r.rolling(5).sum()       # sum(open, 5) * sum(returns, 5)
        alpha_8 = -1*part1.diff(10).rank(axis=1, pct=True).iloc[469:]
        alpha_8.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_8/8.csv', header=None,mode='a')


        # alpha_9 runned
        """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1): ((ts_max(delta(close, 1), 5) < 0) ?
                                    delta(close, 1): (-1 * delta(close, 1))))"""
        if date_d == '20180418':
            alpha_9 = pd.DataFrame(columns=close_2.columns)
            alpha_9.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_9/9.csv')
        condition = data_c.diff(1).rolling(5).min() > 0                 # 0 < ts_min(delta(close, 1), 5)
        indicator1 = data_c.diff(1)                                     # delta(close, 1)
        condition1 = data_c.diff(1).rolling(5).max() < 0                # ts_max(delta(close, 1), 5)  < 0
        incidator2 = indicator1[condition1].fillna(0) - indicator1[~condition1].fillna(0)
        alpha_9 = indicator1[condition].fillna(0).iloc[469:] + indicator2[~condition].fillna(0).iloc[469:]
        alpha_9.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_9/9.csv', header=None,mode='a')


        # alpha_10 runned
        """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1): ((ts_max(delta(close, 1), 4) < 0)
            ? delta(close, 1): (-1 * delta(close, 1)))))"""
        if date_d == '20180418':
            alpha_10 = pd.DataFrame(columns=close_2.columns)
            alpha_10.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_10/10.csv')
        condition = data_c.diff(1).rolling(4).min() > 0                 # 0 < ts_min(delta(close, 1), 4)
        indicator1 = data_c.diff(1)                                     # delta(close, 1)
        condition1 = data_c.diff(1).rolling(4).max() < 0                # ts_max(delta(close, 1), 4) < 0
        incidator2 = indicator1[condition1].fillna(0) - indicator1[~condition1].fillna(0)
        alpha_10 = (indicator1[condition].fillna(0) + indicator2[~condition].fillna(0)).rank(axis=1, pct=True).iloc[469:]
        alpha_10.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_10/10.csv', header=None,mode='a')


        # alpha_11 runned
        """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))"""
        if date_d == '20180418':
            alpha_11 = pd.DataFrame(columns=close_2.columns)
            alpha_11.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_11/11.csv')
        part11 = (data_vw - data_c).rolling(3).max().rank(axis=1, pct=True)     # rank(ts_max((vwap - close), 3))
        part12 = (data_vw - data_c).rolling(3).min().rank(axis=1, pct=True)     # rank(ts_min((vwap - close), 3))
        part2 = data_vo.diff(3).rank(axis=1, pct=True)                          # rank(delta(volume, 3))
        alpha_11 = ((part11+part12)*part2).iloc[469:]
        alpha_11.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_11/11.csv', header=None,mode='a')

        # alpha_12 runned
        """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
        if date_d == '20180418':
            alpha_12 = pd.DataFrame(columns=close_2.columns)
            alpha_12.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_12/12.csv')
        alpha_12 = (-1*np.sign(data_vo.diff(1))*data_c.diff(1)).iloc[469:]
        alpha_12.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_12/12.csv', header=None,mode='a')

        # alpha_13 runned
        """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
        if date_d == '20180418':
            alpha_13 = pd.DataFrame(columns=close_2.columns)
            alpha_13.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_13/13.csv')
        part1 = data_c.rank(axis=1, pct=True)                           # rank(close)
        part2 = data_vo.rank(axis=1, pct=True)                          # rank(volume)
        alpha_13 = (-1*part1.rolling(5).cov(part2, pairwise=False)).iloc[469:]
        alpha_13.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_13/13.csv', header=None,mode='a')


        # alpha_14 runned
        """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
        if date_d == '20180418':
            alpha_14 = pd.DataFrame(columns=close_2.columns)
            alpha_14.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_14/14.csv')
        part1 = -1 * data_r.diff(3).rank(axis=1, pct=True)              # -1 * rank(delta(returns, 3))
        part2 = correlation(data_o, data_vo, 10)                        # correlation(open, volume, 10)
        alpha_14 = (part1 * part2).iloc[469:]
        alpha_14.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_14/14.csv', header=None,mode='a')


        # alpha_15 runned
        """(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
        if date_d == '20180418':
            alpha_15 = pd.DataFrame(columns=close_2.columns)
            alpha_15.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_15/15.csv')
        part1 = data_h.rank(axis=1, pct=True)                           # rank(high)
        part2 = data_vo.rank(axis=1, pct=True)                          # rank(volume)
        alpha_15 = (-1*correlation(part1, part2, 3).rank(1, pct=True).rolling(3).sum()).iloc[469:]
        alpha_15.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_15/15.csv', header=None,mode='a')


        # alpha_16 runned
        """(-1 * rank(covariance(rank(high), rank(volume), 5)))"""
        if date_d == '20180418':
            alpha_16 = pd.DataFrame(columns=close_2.columns)
            alpha_16.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_16/16.csv')
        part11 = data_h.rank(axis=1, pct=True)                          # rank(high)
        part12 = data_vo.rank(axis=1, pct=True)                         # rank(volume)
        part1 = part11.rolling(5).cov(part12, pairwise=False)           # covariance(rank(high), rank(volume), 5)
        alpha_16 = -1 *part1.rank(axis=1, pct=True).iloc[469:]
        alpha_16.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_16/16.csv', header=None,mode='a')


        # alpha_17 runned
        """(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))"""
        if date_d == '20180418':
            alpha_17 = pd.DataFrame(columns=close_2.columns)
            alpha_17.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_17/17.csv')
        part1 = -1 * Ts_rank(data_c, 10).rank(axis=1, pct=True)         # -1 * rank(ts_rank(close, 10))
        part2 = data_c.diff(1).diff(1).rank(axis=1, pct=True)           # rank(delta(delta(close, 1), 1))
        part31 = data_vo / data_dv.rolling(20).mean()                   # (volume / adv20)
        part3 = Ts_rank(part31, 5).rank(axis=1, pct=True)               # rank(ts_rank((volume / adv20), 5))
        alpha_17 = (part1*part2*part3).iloc[469:]
        alpha_17.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_17/17.csv', header=None,mode='a')

        # alpha_18
        """(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))"""
        if date_d == '20180418':
            alpha_18 = pd.DataFrame(columns=close_2.columns)
            alpha_18.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_18/18.csv')
        part1 = abs(data_c-data_o).rolling(5).std()                     # stddev(abs((close - open)), 5)
        part2 = data_c-data_o                                           # close - open
        part3 = correlation(data_c, data_o, 10)                         # correlation(close, open,10)
        alpha_18 = -1*(part1 +part2 +part3).rank(axis=1, pct=True).iloc[469:]
        alpha_18.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_18/18.csv', header=None,mode='a')


        # alpha_19 runned
        """((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))"""
        if date_d == '20180418':
            alpha_19 = pd.DataFrame(columns=close_2.columns)
            alpha_19.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_19/19.csv')
        part1 = -1 * np.sign(data_c.diff(7))                             # -1 * sign(((close - delay(close, 7)) + delta(close, 7)))
        part2 = (data_r.rolling(250).sum()).rank(axis=1, pct=True) + 1  # 1 + rank((1 + sum(returns,250)))
        alpha_19 = (part1 * part2).iloc[469:]
        alpha_19.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_19/19.csv', header=None,mode='a')


        # alpha_20 runned
        """(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))"""
        if date_d == '20180418':
            alpha_20 = pd.DataFrame(columns=close_2.columns)
            alpha_20.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_20/20.csv')
        part1 = -1*(data_o - data_h.shift(1)).rank(axis=1, pct=True)
        part2 = (data_o-data_c.shift(1)).rank(axis=1, pct=True)
        part3 = (data_o-data_l.shift(1)).rank(axis=1, pct=True)
        alpha_20 = (part1*part2*part3).iloc[469:]
        alpha_20.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_20/20.csv', header=None,mode='a')

        # alpha_21 runned
        """((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1): (((sum(close,2) / 2) < ((sum(close,8) / 8)
        - stddev(close, 8))) ? 1: (((1 < (volume / adv20)) | | ((volume /adv20) == 1)) ? 1: (-1 * 1))))"""
        if date_d == '20180418':
            alpha_21 = pd.DataFrame(columns=close_2.columns)
            alpha_21.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_21/21.csv')
        part01 = data_c.rolling(8).mean() + data_c.rolling(8).std()                 # (sum(close, 8) / 8) + stddev(close, 8)
        part02 = data_c.rolling(2).mean()                                           # (sum(close, 2) / 2)
        condition = part01 < part02
        indicator1 = -1 * pd.DataFrame(np.ones_like(data_c),index=data_c.index,columns=data_c.columns)
        part11 = part02                                                             # (sum(close, 2) / 2)
        part12 = data_c.rolling(8).mean() - data_c.rolling(8).std()
        condition1 = part11 < part12
        indicator2 = -indicator1
        condition2 = (data_vo / data_dv.rolling(20).mean())  >= 1
        indicator3 = indicator2
        indicator4 = indicator1
        # Recursive from back to front
        indicator21 = indicator3[condition2].fillna(0) + indicator4[~condition2].fillna(0)
        indicator11 = indicator2[condition1].fillna(0) + indicator21[~condition1].fillna(0)
        alpha_21 = indicator1[condition].fillna(0).iloc[469:] + indicator11[~condition].fillna(0).iloc[469:]
        alpha_21.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_21/21.csv', header=None,mode='a')



        # alpha_22 runned
        """(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
        if date_d == '20180418':
            alpha_22 = pd.DataFrame(columns=close_2.columns)
            alpha_22.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_22/22.csv')
        part1 = correlation(data_h, data_vo, 5).diff(5)                 # delta(correlation(high, volume, 5), 5)
        part2 = data_c.rolling(20).std().rank(axis=1, pct=True)         # rank(stddev(close, 20))
        alpha_22 = (-1* part1 * part2).iloc[469:]
        alpha_22.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_22/22.csv', header=None,mode='a')

        # alpha_23 runned
        """(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)): 0)"""
        if date_d == '20180418':
            alpha_23 = pd.DataFrame(columns=high_2.columns)
            alpha_23.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_23/23.csv')
        condition = data_h.rolling(20).mean() < data_h                  # (sum(high, 20) / 20) < high
        indicator1 = -1 * data_h.diff(2)                                # -1 * delta(high, 2)
        indicator2 = pd.DataFrame(np.zeros_like(data_h), index=data_h.index, columns=data_h.columns)
        alpha_23 = indicator1[condition].fillna(0).iloc[469:] + indicator2[~condition].fillna(0).iloc[469:]
        alpha_23.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_23/23.csv', header=None,mode='a')

        # alpha_24 runned
        """((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) | |
          ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ?
        (-1 * (close - ts_min(close,100))): (-1 * delta(close, 3)))"""
        if date_d == '20180418':
            alpha_24 = pd.DataFrame(columns=close_2.columns)
            alpha_24.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_24/24.csv')
        part11 = data_c.rolling(100).mean().diff(100)       # delta((sum(close, 100) / 100), 100)
        part12 = data_c.shift(100)                          # delay(close, 100)
        condition = part11/part12 <= 0.05
        indicator1 = -1 * (data_c - data_c.rolling(100).min())  # -1 * (close - ts_min(close,100))
        indicator2 = -1 * data_c.diff(3)                        # -1 * delta(close, 3)
        alpha_24  = indicator1[condition].fillna(0).iloc[469:] + indicator2[~condition].fillna(0).iloc[469:]
        alpha_24.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_24/24.csv', header=None,mode='a')

        # alpha_25 runned
        """rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
        if date_d == '20180418':
            alpha_25 = pd.DataFrame(columns=close_2.columns)
            alpha_25.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_25/25.csv')
        part1 = data_dv.rolling(20).mean()              # adv20
        part2 = -1*data_r*part1*data_vw                 # ((-1 * returns) * adv20) * vwap
        part3 = data_h-data_c                           # high - close
        part4= (part2*part3).iloc[469:]
        alpha_25 = part4.rank(axis=1, pct=True)
        alpha_25.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_25/25.csv', header=None,mode='a')

        # alpha_26 runned
        """(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
        if date_d == '20180418':
            alpha_26 = pd.DataFrame(columns=high_2.columns)
            alpha_26.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_26/26.csv')
        part1 = Ts_rank(data_vo, 5)                                     # ts_rank(volume, 5)
        part2 = Ts_rank(data_h, 5)                                      # ts_rank(high, 5)
        part3 = correlation(part1, part2, 5)                            # correlation
        alpha_26 = -1 * part3.rolling(3).max(3).iloc[469:]
        alpha_26.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_26/26.csv', header=None,mode='a')

        # alpha_27 runned
        """((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1): 1)"""
        if date_d == '20180418':
            alpha_27 = pd.DataFrame(columns=close_2.columns)
            alpha_27.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_27/27.csv')
        part11 = data_vo.rank(axis=1, pct=True)                         # rank(volume)
        part12 =data_vw.rank(axis=1, pct=True)                          # rank(vwap)
        part1 = correlation(part11, part12, 6).rolling(2).mean()        # correlation
        condition = part1.rank(axis=1, pct=True) <= 0.5
        indicator = pd.DataFrame(np.ones_like(data_vw), index=data_vw.index, columns=data_vw.columns)
        alpha_27 = indicator[condition].fillna(0).iloc[469:] - indicator[~condition].fillna(0).iloc[469:]
        alpha_27.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_27/27.csv', header=None,mode='a')

        # alpha_28 runned
        """scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
        if date_d == '20180418':
            alpha_28 = pd.DataFrame(columns=close_2.columns)
            alpha_28.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_28/28.csv')
        part11= data_dv.rolling(20).mean()                          # adv20
        part1 = correlation(part1, data_l, 5)                       # correlation(adv20, low, 5)
        part2 = (data_h +data_l)/2                                  # (high + low) / 2
        part3 = part1 +part2 - data_c
        alpha_28 = normalize(part3.fillna(0), norm='l1', axis=1).iloc[469:]
        alpha_28.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_28/28.csv', header=None,mode='a')

        # # alpha_29
        # """(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) +
        #  ts_rank(delay((-1 * returns), 6), 5))"""
        # if date_d == '20180418':
        #     alpha_29 = pd.DataFrame(columns=close_2.columns)
        #     alpha_29.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_29/29.csv')
        # part11 = -1 * ((data_c - 1).diff(5).rank(axis=1, pct=True))             # -1 * rank(delta((close - 1),5))
        # part12 = part11.rank(1, pct=True).rolling(2).min()                      # ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2)
        # part1 = part12.rank(1, pct=True).rolling(5).min()
        # part21 = -1 * data_r.shift(6)                                           # delay((-1 * returns), 6)
        # part2 = Ts_rank(part21, 5)                                              # ts_rank(delay((-1 * returns), 6), 5)
        # alpha_29 = part1.iloc[469:] + part2.iloc[469:]
        # # alpha_29.index = [date_d + ' ' + x for x in time_str]
        # alpha_29.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_29/29.csv', header=None, mode='a')

        # alpha_30  runned
        """(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +
        sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))"""
        if date_d == '20180418':
            alpha_30 = pd.DataFrame(columns=close_2.columns)
            alpha_30.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_30/30.csv')
        part11 = np.sign(data_c.diff(1))                                    # sign((close - delay(close, 1)))
        part12 = np.sign(data_c.diff(1).shift(1))                           # sign((delay(close, 1) - delay(close, 2)))
        part13 = np.sign(data_c.diff(1).shift(2))                           # sign((delay(close, 2) - delay(close, 3)))
        part1 = 1 - (part11+part12+part13).rank(axis=1, pct=True)
        part2 = data_vo.rolling(5).sum() / data_vo.rolling(20).sum()        # sum(volume, 5) / sum(volume, 20)
        alpha_30 = (part1*part2).iloc[469:]
        alpha_30.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_30/30.csv',header=None, mode='a')

        # alpha_31  runned
        """((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *
        delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))"""
        if date_d == '20180418':
            alpha_31 = pd.DataFrame(columns=close_2.columns)
            alpha_31.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_31/31.csv')
        # weights = np.arange(1, 11, 1)/55
        # part1 = (-1*data_c.diff(10).rank(axis=1, pct=True)).rolling(10).apply(decay_linear, raw=True).rank(axis=1, pct=True)
        # part1 = (-1*data_c.diff(10).rank(axis=1, pct=True)).rolling(10).apply(lambda x:(x*weights).sum(), raw=False).rank(axis=1, pct=True)
        part11 = -1*data_c.diff(10).rank(axis=1, pct=True)          # -1 * rank(rank(delta(close, 10)))
        part1 = decay_linear(part11, 10).rank(axis=1, pct=True)     # rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10))))
        part2 = (-1*data_c.diff(3)).rank(axis=1, pct=True)          # rank((-1 *delta(close, 3))))
        adv20 = data_dv.rolling(20).mean()
        part3 = np.sign(correlation(adv20, data_l, 12))             # sign(scale(correlation(adv20, low, 12)))
        alpha_31 = part1.iloc[469:] + part2.iloc[469:] + part3.iloc[469:]
        alpha_31.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_31/31.csv', header=None, mode='a')

        # alpha_32  runned
        """(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))"""
        if date_d == '20180418':
            alpha_32 = pd.DataFrame(columns=close_2.columns)
            alpha_32.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_32/32.csv')
        # scale(((sum(close, 7) / 7) - close))
        part1 = normalize((data_c.rolling(7).mean() -data_c).fillna(0), norm='l1', axis=1)
        # (20 * scale(correlation(vwap, delay(close, 5), 230)))
        part2 = normalize(correlation(data_vw, data_c.shift(5), 230).fillna(0), norm='l1', axis=1)*20
        alpha_32 = part1.iloc[469:] + part2.iloc[469:]
        alpha_32.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_32/32.csv', header=None, mode='a')

        # alpha_33  runned
        """rank((-1 * ((1 - (open / close))^1)))"""
        if date_d == '20180418':
            alpha_33 = pd.DataFrame(columns=close_2.columns)
            alpha_33.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_33/33.csv')
        alpha_33 = (data_o/data_c - 1).rank(axis=1, pct=True).iloc[469:]
        alpha_33.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_33/33.csv', header=None, mode='a')

        # alpha_34  runned
        """rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
        if date_d == '20180418':
            alpha_34 = pd.DataFrame(columns=close_2.columns)
            alpha_34.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_34/34.csv')
        part11 = data_r.rolling(2).std()/ data_r.rolling(5).std()       # stddev(returns, 2) / stddev(returns, 5)
        part1 = part11.rank(axis=1, pct=True)
        part2 = data_c.diff(1).rank(axis=1, pct=True)                   # rank(delta(close, 1))
        alpha_34 = (2 - part1.iloc[469:] - part2.iloc[469:]).rank(axis=1, pct=True)
        alpha_34.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_34/34.csv',header=None, mode='a')

        # alpha_35 done

        # alpha_36 runned
        """"
        (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) +
           (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) +
         (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
         """
        if date_d == '20180418':
            alpha_36 = pd.DataFrame(columns=close_2.columns)
            alpha_36.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_36/36.csv')
        # 2.21 * rank(correlation((close - open), delay(volume, 1), 15))
        part1 = 2.21 * correlation(data_c-data_o, data_vo.shift(1), 15).rank(axis=1, pct=True)
        part2 = 0.7 * (data_o - data_c).rank(axis=1, pct=True)          # 0.7 * rank((open- close))
        part31 = (-1* data_r).shift(6)                                  # delay((-1 * returns), 6)
        part3 = 0.73 * Ts_rank(part31, 5).rank(axis=1, pct=True)        # 0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5))
        part41 = data_dv.rolling(20).mean()                             # adv20
        # rank(abs(correlation(vwap, adv20, 6)))
        part4 = correlation(data_vw, part41, 6).abs().rank(axis=1, pct=True)
        part51 = data_c.rolling(200).mean()-data_o                      # sum(close, 200) / 200) - open
        part52 = data_c -data_o                                         # close - open
        # 0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))
        part5 = 0.6 * (part51*part52).rank(axis=1, pct=True)
        alpha_36 = part1.iloc[469:] + part2.iloc[469:] +part3.iloc[469:] +part4.iloc[469:] +part5.iloc[469:]
        alpha_36.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_36/36.csv',header=None, mode='a')


        # alpha_37 done

        # alpha_38 runned
        """((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"""
        if date_d == '20180418':
            alpha_38 = pd.DataFrame(columns=close_2.columns)
            alpha_38.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_38/38.csv')
        part1 = -1*Ts_rank(data_c, 10).rank(axis=1, pct=True)               # -1 * rank(Ts_Rank(close, 10))
        part2 = (data_c/data_o).rank(axis=1, pct=True)                      # rank((close / open))
        alpha_38 = (part1*part2).iloc[469:]
        alpha_38.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_38/38.csv',header=None, mode='a')


        # alpha_39  runned
        """((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))"""
        if date_d == '20180418':
            alpha_39 = pd.DataFrame(columns=close_2.columns)
            alpha_39.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_39/39.csv')
        part11 = data_c.diff(7)                                             # delta(close, 7)
        part121 = data_vo/ data_dv.rolling(20).mean()                       # volume / adv20
        part12 = 1 - decay_linear(part121, 9).rank(axis=1, pct=True)        # 1-rank(decay_linear((volume / adv20), 9))
        part1 = -1 * (part11*part12).rank(axis=1, pct=True)
        part2 = data_r.rolling(250).sum().rank(axis=1, pct=True) + 1        # 1 +rank(sum(returns, 250))
        alpha_39 = part1.iloc[469:]*part2.iloc[469:]
        alpha_39.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_39/39.csv',header=None, mode='a')


        # alpha_40 done

        # alpha_41 done

        # # alpha_42
        # """(rank((vwap - close)) / rank((vwap + close)))"""
        # if date_d == '20180418':
        #     alpha_42 = pd.DataFrame(columns=close_2.columns)
        #     alpha_42.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_42/42.csv')
        # part1 = (data_vw-data_c).rank(axis=1, pct=True)
        # part2 = (data_vw+data_c).rank(axis=1, pct=True)
        # alpha_42 = part1.iloc[469:] / part2.iloc[469:]
        # alpha_42.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_42/42.csv', header=None, mode='a')

        # alpha_43  runned
        """(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
        if date_d == '20180418':
            alpha_43 = pd.DataFrame(columns=close_2.columns)
            alpha_43.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_43/43.csv')
        part11 = data_vo/data_dv.rolling(20).mean()                         # volume / adv20
        part1 = Ts_rank(part11, 20)                                         # (ts_rank((volume / adv20), 20)
        part2 = Ts_rank(-1*data_c.diff(7), 8)                               # ts_rank((-1 * delta(close, 7)), 8)
        alpha_43 = (part1*part2).iloc[469:]
        alpha_43.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_43/43.csv',header=None, mode='a')

        # alpha_44  runned
        """(-1 * correlation(high, rank(volume), 5))"""
        if date_d == '20180418':
            alpha_44 = pd.DataFrame(columns=high_2.columns)
            alpha_44.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_44/44.csv')
        alpha_44 = -1*data_h.rolling(5).corr(data_vo.rank(axis=1, pct=True).rolling(5), pairwise=False).iloc[469:]
        alpha_44.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_44/44.csv',header=None, mode='a')

        # alpha_45  runned
        """(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *
               rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
        if date_d == '20180418':
            alpha_45 = pd.DataFrame(columns=high_2.columns)
            alpha_45.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_45/45.csv')
        part1 = data_c.shift(5).rolling(20).mean().rank(axis=1, pct=True)       # rank((sum(delay(close, 5), 20) / 20))
        part2 = correlation(data_c, data_vo, 2)                                 # correlation(close, volume, 2)
        # rank(correlation(sum(close, 5), sum(close, 20), 2))
        part3 = correlation(data_c.rolling(5).sum(), data_c.rolling(20).sum(), 2).rank(axis=1, pct=True)
        alpha_45 = (-1*part1*part2*part3).iloc[469:]
        alpha_45.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_45/45.csv',header=None, mode='a')

        # alpha_46  runned
        """((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
        (-1 * 1): (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ?
        1:((-1 * 1) * (close - delay(close,1)))))"""
        if date_d == '20180418':
            alpha_46 = pd.DataFrame(columns=high_2.columns)
            alpha_46.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_46/46.csv')
        condition = data_c.diff(10).diff(10)/10 > 0.25
        indicator1 = -1 * pd.DataFrame(np.ones_like(data_c), index=data_c.index, columns=data_c.columns)
        condition1 = data_c.diff(10).diff(10)/10 < 0
        indicator21 = pd.DataFrame(np.ones_like(data_c), index=data_c.index, columns=data_c.columns)
        indicator22 = -1*data_c.diff(1)                             # (-1 * 1) * (close - delay(close,1))
        indicator2 = indicator21[condition1].fillna(0) + indicator22[~condition1].fillna(0)
        alpha_46 = indicator1[condition].fillna(0).iloc[469:] + indicator2[~condition].fillna(0).iloc[469:]
        alpha_46.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_46/46.csv',header=None, mode='a')


        # # alpha_47
        # # ((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5)) - rank((vwap - delay(vwap, 5)))
        # if date_d == '20180418':
        #     alpha_47 = pd.DataFrame(columns=close_2.columns)
        #     alpha_47.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_47/47.csv')
        # part1 = (1 / data_c).rank(axis=1, pct=True) * data_vo / (data_vo * data_vw).rolling(20).mean()
        # part2 = (data_h - data_c).rank(axis=1, pct=True) * data_h / data_h.rolling(5).mean()
        # part3 = data_vw.diff(5).rank(axis=1, pct=True)
        # alpha_47 = part1.iloc[469:] * part2.iloc[469:] - part3.iloc[469:]
        # alpha_47.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_47/47.csv', header=None, mode='a')

        # alpha_48 not
        """(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / 
                        close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1)) ^ 2), 250))"""


        # alpha_49  runned
        """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ?
        1: ((-1 * 1) * (close - delay(close, 1))))"""
        if date_d == '20180418':
            alpha_49 = pd.DataFrame(columns=close_2.columns)
            alpha_49.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_49/49.csv')
        condition = data_c.diff(10).diff(10)/10 < -0.1
        indicator1 = pd.DataFrame(np.ones_like(data_c), index=data_c.index, columns=data_c.columns)
        indicator2 = -1*data_c.diff(1)                                  # (-1 * 1) * (close - delay(close, 1))
        alpha_49 = indicator1[condition].fillna(0).iloc[469:] + indicator2[~condition].fillna(0).iloc[469:]
        alpha_49.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_49/49.csv',header=None, mode='a')




        # alpha_50  runned
        """(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=high_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        part1 = data_vo.rank(axis=1, pct=True)                      # rank(volume)
        part2 = data_vw.rank(axis=1, pct=True)                      # rank(vwap)
        # rank(correlation(rank(volume), rank(vwap), 5))
        part3 = correlation(part1, part2, 5).rank(axis=1, pct=True)
        alpha_50 = -1*part3.rolling(5).max().iloc[469:]
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv',header=None, mode='a')



        # # alpha_51
        # """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05))
        #     ? 1: ((-1 * 1) * (close - delay(close, 1))))"""
        # if date_d == '20180418':
        #     alpha_51 = pd.DataFrame(columns=close_2.columns)
        #     alpha_51.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_51/51.csv', mode='a')
        # condition = data_c.diff(periods=10).diff(periods=10) / 10 < -0.05
        # indicator1 = pd.DataFrame(np.ones(data_c.shape), index=data_c.index, columns=data_c.columns)
        # indicator2 = -close.diff(periods=1)
        # alpha_51 = indicator1[condition] + indicator2[~condition]
        # alpha_51.index = [date_d +' '+ x for x in time_str]
        # alpha_51.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_51/51.csv', header=None, mode='a')

        # alpha_52
        """((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -
            sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
        if date_d == '20180418':
            alpha_52 = pd.DataFrame(columns=close_2.columns)
            alpha_52.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_52/52.csv')
        part11 = -1*data_l.rolling(5).min()                             # -1 * ts_min(low, 5)
        part12 = data_l.rolling(5).min().shift(5)                       # delay(ts_min(low, 5), 5)
        part1 = part11 +part12
        # rank(((sum(returns, 240) -sum(returns, 20)) / 220))
        part2 = data_r.shift(20).rolling(220).mean().rank(axis=1, pct=True)
        part3 = Ts_rank(data_vo, 5)                                     # ts_rank(volume, 5)
        alpha_52 = (part1*part2*part3).iloc[469:]
        alpha_52.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_52/52.csv',header=None, mode='a')

        # alpha_53 done

        # alpha_54 done

        # alpha_55
        """(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))"""
        if date_d == '20180418':
            alpha_55 = pd.DataFrame(columns=close_2.columns)
            alpha_55.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_55/55.csv')
        part11 = data_c - data_l.rolling(12).min()                      # close - ts_min(low, 12)
        part12 = data_h.rolling(12).max()-data_l.rolling(12).min()      # ts_max(high, 12) - ts_min(low,12)
        part1 = (part11/part12).rank(axis=1, pct=True)
        part2 = data_vo.rank(axis=1, pct=True)                          # rank(volume)
        alpha_55 = -1*correlation(part1, part2, 6).iloc[469:]
        alpha_55.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_55/55.csv', header=None,mode='a')

        # alpha_56  not
        """(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""

        # alpha_57
        # (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        if date_d == '20180418':
            alpha_57 = pd.DataFrame(columns=close_2.columns)
            alpha_57.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_57/57.csv')
        part1 = data_c - data_vw                                    # close - vwap
        part21 = Ts_ArgMax(data_c, 30).rank(axis=1, pct=True)       # rank(ts_argmax(close, 30))
        part2 = decay_linear(part21, 2)
        alpha_57 = -1*(part1/part2).iloc[469:]
        alpha_57.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_57/57.csv', header=None,mode='a')

        # alpha_58 not
        """(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,
        3.92795), 7.89291), 5.50322))"""

        # alpha_59 not
        """(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *
        (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))"""

        # alpha_60
        """(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -
                   scale(rank(ts_argmax(close, 10))))))"""
        if date_d == '20180418':
            alpha_60 = pd.DataFrame(columns=close_2.columns)
            alpha_60.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_60/60.csv')
        # (((close - low) - (high - close)) / (high - low)) * volume
        part11 = (2*data_c-data_l-data_h)/(data_h-data_l)*data_vo
        part1 = 2*normalize(part11.rank(axis=1, pct=True), norm='l1', axis=1)
        # scale(rank(ts_argmax(close, 10)))
        part2 = normalize(Ts_ArgMax(data_c, 10).rank(axis=1, pct=True), norm='l1', axis=1)
        alpha_60 = -1*(part1*part2).iloc[469:]
        alpha_60.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_60/60.csv', header=None,mode='a')

        # # alpha_61
        # # (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
        # if date_d == '20180418':
        #     alpha_61 = pd.DataFrame(columns=close_2.columns)
        #     alpha_61.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_61/61.csv', mode='a')
        # condition = (data_vw - data_vw.rolling(16).min()).rank(axis=1, pct=True) < data_vw.rolling(17).corr((data_dv).rolling(180).mean().rolling(17), pairwise=False)
        # indicator1 = pd.DataFrame(np.ones(data_vw.shape), index=data_vw.index, columns=data_vw.columns)
        # indicator2 = -pd.DataFrame(np.ones(data_vw.shape), index=data_vw.index, columns=data_vw.columns)
        # part1 = indicator1[condition].iloc[469:]
        # part2 = indicator2[~condition].iloc[469:]
        # alpha_61 = part1 +part2
        # alpha_61.index = [date_d +' '+ x for x in time_str]
        # alpha_61.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_61/61.csv', header=None, mode='a')

        # alpha_62
        """((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) <
        (rank(((high + low) / 2)) + rank(high))))) * -1)"""
        if date_d == '20180418':
            alpha_62 = pd.DataFrame(columns=close_2.columns)
            alpha_62.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_62/62.csv')
        part11 = data_dv.rolling(20).mean()                                 # adv20
        part12 = part11.rolling(22).sum()                                   # sum(adv20, 22.4101)
        # rank(correlation(vwap, sum(adv20, 22.4101), 9.91009))
        part1 = correlation(data_vw, part12, 10).rank(axis=1, pct=True)
        part21 = 2*data_o.rank(axis=1, pct=True)                            # rank(open) + rank(open)
        part221 = (data_h+data_l).rank(axis=1, pct=True)                    # rank(((high + low) / 2))
        part222 = data_h.rank(axis=1, pct=True)                             # rank(high)
        part22 = part221+part222
        part2 = (part21 < part22).rank(axis=1, pct=True)
        indicator = pd.DataFrame(np.ones_like(data_h), index=data_h.index, columns=data_h.columns)
        alpha_62 = -1*(indicator[part1 < part2].fillna(0) - indicator[part1>=part2].fillna(0)).iloc[469:]
        alpha_62.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_62/62.csv', header=None,mode='a')

        # alpha_63 not
        """((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))
          - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), 
        sum(adv180,37.2467), 13.557),12.2883))) * -1)"""

        # alpha_64
        """((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) <
          rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))),3.69741))) * -1)"""
        if date_d == '20180418':
            alpha_64 = pd.DataFrame(columns=close_2.columns)
            alpha_64.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_64/64.csv')
        part111 = data_o * 0.178404 + data_c *(1-0.178404)
        part11 = part111.rolling(12.7054).mean()
        part121 = data_dv.rolling(120).mean()
        part12 = part121.rolling(12.7054).sum()
        part1 = correlation(part11, part12, 16.6208).rank(axis=1, pct=True)
        part211 = (data_h+data_l)/2*0.178404
        part212 = data_vw*(1-0.178404)
        part21 = part211+part212
        part2 = part21.diff(3.69741).rank(axis=1, pct=True)
        indicator = pd.DataFrame(np.ones_like(data_h), index=data_h.index, columns=data_h.columns)
        alpha_64 = -1*(indicator[part1<part2].fillna(0)-indicator[part1>=part2].fillna(0)).iloc[469:]
        alpha_64.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_64/64.csv', header=None,mode='a')

        # alpha_65
        # (rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374))
        # 	< rank((open - ts_min(open, 13.635)))) * -1
        # if date_d == '20180418':
        #     alpha_65 = pd.DataFrame(columns=close_2.columns)
        #     alpha_65.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_65/65.csv', mode='a')
        # rank1 = (data_o * 0.00817205 + data_vw * (1 - 0.00817205)).rolling(7).corr((data_dv).rolling(60).mean().rolling(9).sum(), pairwise=False).rank(axis=1, pct=True)
        # rank2 = (data_o - data_o.rolling(13).min()).rank(axis=1, pct=True)
        # indicator1 = pd.DataFrame(np.ones(data_vw.shape), index=data_vw.index, columns=data_vw.columns)
        # indicator2 = -pd.DataFrame(np.ones(data_vw.shape), index=data_vw.index, columns=data_vw.columns)
        # part1 = indicator1[rank1 > rank2]
        # part2 = indicator2[rank1 <= rank2]
        # alpha_65 = part1.iloc[469:] +part2.iloc[469:]
        # alpha_65.index = [date_d +' '+ x for x in time_str]
        # alpha_65.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_65/65.csv', header=None, mode='a')

        # alpha_66
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_67
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_68
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_69
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_70
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_72
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_73
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_74
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_75
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_76
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_77
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_78
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_80
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_81
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_82
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_83
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_84
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_85
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_86
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_87
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_88
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_89
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_90
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_91
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_92
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_93
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_94
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_95
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_96
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_97
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_98
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_99
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_100
        if date_d == '20180418':
            alpha_50 = pd.DataFrame(columns=close_2.columns)
            alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv')
        alpha_50.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_50/50.csv', header=None,
                        mode='a')

        # alpha_101 done

        close_1, open_1, high_1, low_1, volume_1, vwap_1, return_1 = close_2, open_2, high_2, low_2, volume_2, vwap_2, return_2
        print(datetime.datetime.now() - start)
    except StopIteration:
        loop = False
        print('Iteration is stopped')
