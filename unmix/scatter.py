
import numpy as np
import matplotlib
from scipy import stats
import matplotlib.pyplot as plt
import utils
import os
import scipy.stats as st
import pickle
import pandas as pd

ifile = 'Amazon_umix_inun.pkl'
pfile = 'Amazon_umix_precip.pkl'
datadir = 'data'
delta = 'Amazon'
p_data = utils.get_data(datadir, pfile, delta)
precip_data = p_data['data']
i_data = utils.get_data(datadir, ifile, delta)
inun_data = i_data['data']
"""
delta = utils.get_data(datadir, datafile,deltaname)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
locations = delta['data'].iloc[:, 0]
y = (1.2* locations ) + (.3* np.random.randn(len(locations)))
print locations

print stats.pearsonr(locations, y)
plt.savefig('plot.png')
"""
################################
precip = pd.rolling_window(precip_data.mean(axis=1),
			window=45, win_type='gaussian', std=39)
end = min(precip.index[-1], inun_data.index[-1])
iclip = inun_data.mean(axis=1)[:end] 
pclip = precip[iclip.index[0]:end:10]
mask = (np.isfinite(iclip) & np.isfinite(pclip))

################################



#print alldeltas['inundation']['Ganges']

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#locations = alldeltas['precip']['Ganges']
#ax1.plot(locations)
#print stats.pearsonr(locations, y)
#plt.savefig('plot.png')

#print alldeltas['precip']['Ganges']



cleanin = []
cleanp = []
for i,j in zip(pclip,iclip):
    if np.isfinite(i) and np.isfinite(j):
        cleanin.append(i)
        cleanp.append(j)
print stats.pearsonr(cleanin, cleanp)

slope, intercept, _, _, _ = st.linregress(cleanin, cleanp)
ax1.plot(pclip, slope*pclip + intercept, color='k')
ax1.set_title("Correlation: {:.3f}".format(st.pearsonr(cleanin, cleanp)[0]))
ax1.scatter(cleanin, cleanp)
ax1.set_ylim([min(cleanp), max(cleanp)])
ax1.set_xlim([min(cleanin), max(cleanin)])
ax1.set_ylabel("Precipitation")
ax1.set_xlabel("Inundation")
plt.savefig('inunvsprecip_test.png')