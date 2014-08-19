
import numpy as np
import matplotlib
matplotlib.use('cairo')
from scipy import stats
import matplotlib.pyplot as plt
import utils
import os
import scipy.stats as st
import pickle

datadir = ''
datafile = 'delta_byName_time_series.v2.pkl'
deltaname = 'Ganges'
"""
delta = utils.get_data(datadir, datafile,deltaname)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
locations = delta['data'].iloc[:,0]
y = (1.2* locations ) + (.3* np.random.randn(len(locations)))
print locations

print stats.pearsonr(locations, y)
plt.savefig('plot.png')
"""

with open(os.path.join(datadir, datafile ), 'r') as f:
        alldeltas = pickle.load(f)
print alldeltas.keys()

#print alldeltas['inundation']['Ganges']

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#locations = alldeltas['precip']['Ganges']
#ax1.plot(locations)
#print stats.pearsonr(locations, y)
#plt.savefig('plot.png')

#print alldeltas['precip']['Ganges']

data1 = alldeltas['precip']['Ganges']
data2 = alldeltas['inundation']['Ganges']

cleanin = []
cleanp = []
for i,j in zip(data1,data2):
    if np.isfinite(i) and np.isfinite(j):
        cleanin.append(i)
        cleanp.append(j)
print stats.pearsonr(cleanin, cleanp)
mask = np.isfinite(data1) & np.isfinite(data2)
slope, intercept, _, _, _ = st.linregress(data1[mask], data2[mask])
ax1.plot(data1, slope*data1 + intercept, color='k')
ax1.set_title("Correlation: {:.3f}".format(st.pearsonr(data1[mask], data2[mask])[0]))
ax1.scatter(cleanin, cleanp)
ax1.set_ylim([min(cleanp), max(cleanp)])
ax1.set_xlim([min(cleanin), max(cleanin)])
plt.savefig('inunvsprecip_linearregr.png')