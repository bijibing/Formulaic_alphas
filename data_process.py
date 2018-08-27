import pandas as pd


df = pd.read_csv('/home/lp0477/pcshare/ticker_data/2018_6_8/20180608', chunksize=10000, header=None, usecols=[i for i in range(54)])


data = pd.DataFrame()
for chunk in df:
    data = pd.concat([data, chunk])
    print(data)