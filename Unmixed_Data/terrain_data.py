import os

import pandas as pd
import numpy as np
import scipy.stats as st
from personalutils import *

pp = [os.pardir,'data']
dpath = os.path.join(*pp)
pfile = 'delta_3B42_precip.pkl'
ifile = 'inun_minremoved_v1v2.pkl'

pdata = pd.read_pickle(os.path.join(dpath, pfile))
idata = pd.read_pickle(os.path.join(dpath, ifile))

start = '2001-01-18'
startp = '2001-01-08'
end = '2012-12-30'
dn = 'Amazon'
w=50
std = 179
i = idata[dn][:][start:end]
p = pdata[dn][:][startp:]
precip = pd.rolling_window(p, window=w, win_type='gaussian', std=std)
#put on 10 year time scale matched to inudation
end = min(precip.index[-1], i.index[-1])
iclip = i[:end]
pclip = precip[iclip.index[0]:end:10]
mask = (np.isfinite(iclip) & np.isfinite(pclip))
np.testing.assert_array_equal(pclip.index, iclip.index)
# pclip,iclip = cleanNans(pclip,iclip)
# pclip.to_pickle("{}_{}_{}.pkl".format(dn, w,std))