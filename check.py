import pandas as pd
import numpy as np
from scipy.stats import rankdata
import datetime
from sklearn.preprocessing import normalize
import os
import gc



close = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/ohlc/close_0418_0702_30s.csv', index_col=0, iterator=True)
volume = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/volume/volume_0418_0702_30s.csv', index_col=0, iterator=True)
vwap = pd.read_csv('/home/lp0477/pcshare/ticker_data/30_seconds/vwap/vwap_0418_0702_30s.csv', index_col=0, iterator=True)
time_str = [str(a)+':'+str(b)+':'+str(c) for a in range(9,12) for b in range(0, 60) for c in ['00', '30']][61:-60] + [str(a)+':'+str(b)+':'+str(c) for a in range(13,15) for b in range(0, 60) for c in ['00', '30']][:-10]
close_1, volume_1, vwap_1 = pd.DataFrame(index=time_str), pd.DataFrame(index=time_str), pd.DataFrame(index=time_str)


def decay_linear(na, n):
    decay_weights = np.arange(1, n + 1, 1)
    return na.rolling(n).apply(lambda x:np.average(x, weights=decay_weights), raw=True)


def Ts_rank(na, n):
    return na.rolling(n).apply(lambda x:x.rank(axis=0, pct=True).iloc[-1], raw=False)


def correlation(na, nb, n):
    return na.rolling(n).corr(nb, pairwise=False)


start = datetime.datetime.now()
loop = True
chunksize = 469
while loop:
    try:
        close_2 = close.get_chunk(chunksize)
        data_c = pd.concat([close_1, close_2], join='outer', sort=True)
        # data_c = data_c.iloc[469:]
        volume_2 = volume.get_chunk(chunksize)
        data_vo = pd.concat([volume_1, volume_2], join='outer', sort=False)
        # data_vo = data_vo.iloc[469:]
        vwap_2 = vwap.get_chunk(chunksize)
        data_vw = pd.concat([vwap_1, vwap_2], join='outer', sort=False)
        data_dv = data_vw*data_vo


        # alpha_2
        # (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        start_1 = datetime.datetime.now()
        part1 = correlation(data_c, data_vo, 10)
        runned_1 = datetime.datetime.now()-start_1
        data_c = data_c.iloc[469:]
        data_vo = data_vo.iloc[469:]
        start_2 = datetime.datetime.now()
        part2 = correlation(data_c, data_vo, 10)
        runned_2 = datetime.datetime.now() -start_2
        print(runned_1, runned_2)

        close_1, volume_1, vwap_1 = close_2, volume_2, vwap_2
        # print(datetime.datetime.now() - start)
    except StopIteration:
        loop = False
        print('Iteration is stopped')