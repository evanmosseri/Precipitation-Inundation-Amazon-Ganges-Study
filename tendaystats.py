import os

import pandas as pd
import numpy as np
import scipy.stats as st
from personalutils import *
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats as st
import utils
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import colors

dpath = 'Amazon'
ifile = 'Amazon_inun_clip.pkl'
pmfile = 'Amazon_precip_10D_avg.pkl'

idata = pd.read_pickle(os.path.join(dpath, ifile))
pmdata = pd.read_pickle(os.path.join(dpath, pmfile))

fig = plt.figure()
itm = idata.std(axis=1)
pmtm = pmdata.std(axis=1)
itm,pmtm = cleanNans(itm,pmtm)
slope, intercept, _, _, _ = st.linregress(itm, pmtm)
ax = addScatter(fig,plt,itm,pmtm,1,1,1)
ax.set_xlabel("Inundation")
ax.set_ylabel("Precipitation")
ax.plot(itm, slope*itm + intercept, color='k')
ax.set_title("Correlation is: {}".format(st.pearsonr(itm,pmtm)[0]),y=1.02)
# plt.savefig("./graphs/tendaystats.png")
corrs = []
for i, p in zip(idata.values.T, pmdata.values.T):
    mask = (np.isfinite(i) & np.isfinite(p))
    corrs.append(st.pearsonr(i[mask],p[mask])[0])

print len(corrs)
globe = utils.gridEdges('data')
delta = utils.get_data('data', 'delta_3B42_precip.pkl', 'Amazon')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

cmap = cm.GMT_drywet

vmax = np.nanmax(corrs)
# vmin = -vmax
vmin = 0
norm = colors.Normalize(vmin=vmin, vmax=vmax)
bm = utils.basemap(ax1)
X, Y = bm(globe['lons'], globe['lats'])


globe['map'][delta['inds'][0], delta['inds'][1]] = np.array(corrs)

im = bm.pcolormesh(X, Y, np.ma.masked_invalid(globe['map']),cmap=cmap, norm=norm)
cbar = bm.colorbar(im, "bottom", cmap=cmap, norm=norm)  
ax1.axis(utils.mapbounds['Amazon'])
ax1.set_title("Ten Day Correlations")
# plt.show()
plt.savefig("./graphs/tendaystds.png")
