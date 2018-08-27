"""formulaic alphas 101"""
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import datetime
from sklearn.preprocessing import normalize


class formulaic_alphas:
    def __init__(self, file_path):
        self.close = pd.read_csv(file_path + '/ohlc/close_0418_0702_30s.csv', index_col=0,iterator=True)
        self.high = pd.read_csv(file_path + '/ohlc/high_0418_0702_30s.csv', index_col=0,iterator=True)
        self.open_d = pd.read_csv(file_path + '/ohlc/open_0418_0702_30s.csv', index_col=0,iterator=True)
        self.low = pd.read_csv(file_path + '/ohlc/low_0418_0702_30s.csv', index_col=0,iterator=True)
        self.volume = pd.read_csv(file_path + '/volume/volume_0418_0702_30s.csv', index_col=0,iterator=True)
        self.vwap = pd.read_csv(file_path + '/vwap/vwap_0418_0702_30s.csv', index_col=0,iterator=True)
        self.returns = pd.read_csv(file_path + '/returns/returns_0418_0702_30s.csv',index_col=0, iterator=True)
        self.begin = pd.DataFrame(index=[i for i in range(469)])

    def decay_linear(self, na, n):
        decay_weights = np.arange(1, n + 1, 1)
        return na.rolling(n).apply(lambda x: np.average(x, weights=decay_weights), raw=True)

    def Ts_rank(self, na, n):
        return na.rolling(n).apply(lambda x: x.rank(axis=0, pct=True).iloc[-1], raw=False)

    def correlation(self, na, nb, n):
        return na.rolling(n).corr(nb, pairwise=False)

    def SignedPower(self,na, n):
        return na ** n * np.sign(na)

    def Ts_ArgMax(self, na, n):
        return na.rolling(n).apply(lambda x: x.idxmax(skipna=True), raw=False)


    def alpha_001(self):
        start = datetime.datetime.now()
        loop = True
        chunksize = 469
        while loop:
            try:
                close_2 = self.close.get_chunk(chunksize)
                data_c = pd.concat([close_1, close_2], join='outer', sort=True)
                return_2 = self.returns.get_chunk(chunksize)
                data_r = pd.concat([return_1, return_2], join='outer', sort=False)
                date_d = close_2.index[0].split(' ')[0]
                print(date_d)

                # alpha_1
                """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)"""
                if date_d == '20180418':
                    alpha_1 = pd.DataFrame(columns=close_2.columns)
                    alpha_1.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_1/1.csv')
                condition = data_r < 0
                indicator1 = data_r.rolling(20).std()
                indicator2 = data_c
                part1 = indicator1[condition].fillna(0) + indicator2[~condition].fillna(0)
                part2 = part1 ** 2 * np.sign(part1)
                part3 = Ts_ArgMax(part2, 5)
                alpha_1 = part3.rank(axis=1, pct=True).iloc[469:] - 0.5
                alpha_1.to_csv('/home/lp0477/pcshare/ticker_data/30_seconds/formulaic_alphas/alpha_1/1.csv',
                               header=None, mode='a')

                close_1, return_1 = close_2, return_2
                print(datetime.datetime.now() - start)
            except StopIteration:
                loop = False
                print('Iteration is stopped')