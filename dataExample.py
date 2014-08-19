
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.animation as animation
from matplotlib import colors

import utils
import PCAmekong

datadir = 'data'
deltaname = 'Mekong'
datafile = 'delta_3B42_precip.pkl'

#Array of nans the size of dataset based on size of map
#Array of bounding map edges
#Actual inundation data for specified river
map = utils.gridEdges(datadir)

delta = utils.get_data(datadir, datafile,deltaname)


cmap = cm.GMT_drywet
vmin = np.nanmin(delta['data'])
vmax = np.nanmax(delta['data'])
norm = colors.Normalize(vmin=vmin, vmax=vmax)

fig = plt.figure()
ax = fig.add_subplot(2,2,1)
#sets up basemap object to plot
bm = utils.basemap(ax)

X, Y = bm(map['lons'], map['lats'])
#sets axis for the river we are observing
ax.axis(utils.mapbounds['Mekong'])

#fills map with inundation data based on location of delta
day = 0;
map['map'][delta['inds'][0], delta['inds'][1]] = delta['data'].iloc[day,:]
date = delta['data'].index[0].strftime('%Y-%m-%d')
#sets up colormesh to display over map
im = bm.pcolormesh(X, Y, np.ma.masked_invalid(map['map']),cmap=cmap, norm=norm)
cbar = bm.colorbar(im, "bottom", cmap=cmap, norm=norm)  
ax.set_title("Inundation: {}".format(date))


ax1 = fig.add_subplot(2,2,2)
ax1.plot(delta['data'].iloc[:,0])
ax1.set_title('Location 1 data')

#Clean Matrix without nan
MekongMat, cleanind,delta_data  = PCAmekong.preprocess(delta['data'])
#creates an array of mean values for each locations inundation
meanLoc = []
#take the mean of each column(location) of mekong inundation data
for i in range(MekongMat.shape[1]):
    meanLoc.append(np.mean(MekongMat[:,i]))

map['map'][delta['inds'][0], delta['inds'][1]] = meanLoc
#create mean basemap
ax2= fig.add_subplot(2,2,3)
bm2 = utils.basemap(ax2)
ax2.axis([31500000, 32250000, 7250000, 7750000])
im2 = bm2.pcolormesh(X,Y, np.ma.masked_invalid(map['map']), cmap=cmap)
cbar = bm2.colorbar(im, "bottom", cmap=cmap)
ax2.set_title("Inundation averages")

#create a plot of averages over time series
meanTime = []
for i in range(MekongMat.shape[0]):
    meanTime.append(np.mean(MekongMat[i,:]))
ax3 = fig.add_subplot(2,2,4)
ax3.set_ylabel('Average Inundation')
ax3.plot(meanTime)
ax3.set_title("Inundation averages 1990 - 2010?")

plt.show()
# fig.savefig('{}.png'.format('Inundation data'))

