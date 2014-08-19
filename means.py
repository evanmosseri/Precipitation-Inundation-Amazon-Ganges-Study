import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.animation as animation
from matplotlib import colors
import sys
import utils

"""
python means.py [delta]

"""
deltaname = str(sys.argv[1])

datadir = 'data'

datafile = 'inun_minremoved_v1v2.pkl'

#Array of nans the size of dataset based on size of map
#Array of bounding map edges
globalMap = utils.gridEdges(datadir)

#Actual inundation data for specified river
delta = utils.get_data(datadir, datafile, deltaname)
dates = delta['data'].index


timeMean = delta['data'].mean(axis=1)
locMean = delta['data'].mean(axis=0)

cmap = cm.GMT_drywet
vmin = np.amin(locMean)
vmax = np.amax(locMean)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

fig = plt.figure()
ax = fig.add_subplot(221)
title='Map of Inundation Means \nby Location'
#sets up basemap object to plot
utils.plotGrid(ax, locMean, delta, globalMap,deltaname, title, cmap, norm)

ax1 = fig.add_subplot(222)
title = 'Time Series of Inundation Means'
ax1.set_title(title)
ax1.set_ylabel('Average Inundation')
ax1.plot_date(dates , timeMean, '-' )
ax1.tick_params(labelsize=8)


datafile = 'delta_3B42_precip.pkl'
delta = utils.get_data(datadir, datafile, deltaname)
dates = delta['data'].index

timeMean = delta['data'].mean(axis=1)
locMean = delta['data'].mean(axis=0)

cmap = cm.GMT_drywet
vmin = np.amin(locMean)
vmax = np.amax(locMean)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

ax2 = fig.add_subplot(223)
title = 'Map of Precip Means by Location'
utils.plotGrid(ax2, locMean, delta, globalMap, deltaname, title, cmap, norm)

ax3 = fig.add_subplot(224)
title = 'Time Series of Precip Means'
ax3.set_title(title)
ax3.set_ylabel('Average Precip')
ax3.plot_date(dates , timeMean, '-' )
ax3.tick_params(labelsize=8)
plt.savefig('means.png')
