import pickle
from matplotlib import figure
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt 
import os
from mpl_toolkits.basemap import Basemap,cm
from matplotlib import colors

def preprocess(arr):
            
    Xmatrix = np.vstack((arr[cf] for cf in arr)).T
    garbagerow = []
    cleanrow = []
    cleanind = []
    for i,row in enumerate( Xmatrix):
        if np.isnan(row).any():
            garbagerow.append(i)
        else:
            cleanrow.append(row)
            cleanind.append(i)
    cleanMatrix = np.vstack(cleanrow)
    nanRows = np.vstack(garbagerow)
    return cleanMatrix , nanRows, cleanind

# open cell centers
lons = np.load('lons.npy')
lats = np.load('lats.npy')

fullmap = np.zeros(lons.shape) * np.nan
# get data, indicies, and insert data into full grid
mekong_ind = np.load('Mekong/Mekong_indicies.npy')

with open('inun_minremoved_v1v2.pkl','r') as f:
    alldeltas = pickle.load(f)
mekong_data = alldeltas['Mekong']

print mekong_data.shape
MekongMat, nanRows, cleanind =preprocess(mekong_data)

pca = PCA(n_components=10)
x_pca = pca.fit_transform(MekongMat)
print x_pca.shape


#Fill in bad values with nans
pca_fill = np.ones((mekong_data.shape[0],10))*np.nan
pca_fill[cleanind] = x_pca
fig = figure.Figure(figsize=(20,8))
canvas = FigureCanvas(fig)
x = mekong_data.index[:]
print len(x)

ax = fig.add_subplot(211)
ax.plot_date(x , pca_fill[:,0], '-', label='Component 1')
ax.plot_date(x , pca_fill[:,1], '-', label='Component 2')
ax.plot_date(x , pca_fill[:,2], '-', label='Component 3')
ax.set_title('PCA projections inundation')
ax.legend(loc=8, ncol=3)
ax.set_xlim((min(x),max(x)))


ax1 = fig.add_subplot(212)
ax1.plot_date(x , mekong_data[1], '-')
ax1.set_title('Inundations data 10days 1996')
canvas.print_figure('plot.png')

"""
# EASE Grid is simply a particular grid in a cylindrical equal area projection
bm = Basemap(projection='cea',  # cylindrical equal area
             lon_0=0,
             lat_0=30,
             rsphere=6371228,
             resolution='l')

# open cell centers
lons = np.load('lons.npy')
lats = np.load('lats.npy')


# estimate cell boundaries from cell centers, important for correct pcolormesh
# plotting
LONedge = np.zeros((lons.shape[0] + 1, lons.shape[1] + 1))
LATedge = np.zeros((lats.shape[0] + 1, lats.shape[1] + 1))
LONedge[1:-1,1:-1] = lons[:-1,:-1] + .5 * np.diff(lons[:-1,:], axis=1)
LATedge[1:-1,1:-1] = lats[:-1,:-1] + .5 * np.diff(lats[:,:-1], axis=0)
LONedge[:,0] = 2*LONedge[:,1] - LONedge[:,2]
LONedge[:,-1] = 2*LONedge[:,-2] - LONedge[:,-3]
LONedge[0,:] = 2*LONedge[1,:] - LONedge[2,:]
LONedge[-1,:] = 2*LONedge[-2,:] - LONedge[-3,:]
LATedge[:,0] = 2*LATedge[:,1] - LATedge[:,2]
LATedge[:,-1] = 2*LATedge[:,-2] - LATedge[:,-3]
LATedge[0,:] = 2*LATedge[1,:] - LATedge[2,:]
LATedge[-1,:] = 2*LATedge[-2,:] - LATedge[-3,:]

# project cell boundaries to EASE grid coordinates
X, Y = bm(LONedge, LATedge)
# fill map with nans
fullmap1 = np.zeros(lons.shape) * np.nan
fullmap2 = np.zeros(lons.shape) * np.nan
fullmap3 = np.zeros(lons.shape) * np.nan

# get data, indicies, and insert data into full grid
mekong_ind = np.load('Mekong/Mekong_indicies.npy')

fullmap1[mekong_ind[0], mekong_ind[1]] = x_pca[:,0]
fullmap2[mekong_ind[0], mekong_ind[1]] = x_pca[:,1]
fullmap3[mekong_ind[0], mekong_ind[1]] = x_pca[:,2]

vmax = np.nanmax(abs(MekongMat))
vmin = -vmax

norm = colors.normalize(vmin=vmin, vmax=vmax)

cmap='BrBG'
fig = plt.figure()
ax1 = fig.add_subplot(231)
im=bm.pcolormesh(X, Y, np.ma.masked_invalid(fullmap1), cmap=cmap, norm=norm)
bm.colorbar(im,'bottom',cmap=cmap, norm=norm)
bm.drawcoastlines()
bm.fillcontinents(zorder=0)
ax1.axis([31500000, 32250000, 7250000, 7750000])


ax2 = fig.add_subplot(232)
im=bm.pcolormesh(X, Y, np.ma.masked_invalid(fullmap2), cmap=cmap, norm=norm)
bm.colorbar(im,'bottom',cmap=cmap, norm=norm)
bm.drawcoastlines()
bm.fillcontinents(zorder=0)
ax2.axis([31500000, 32250000, 7250000, 7750000])


ax3 = fig.add_subplot(233)
im=bm.pcolormesh(X, Y, np.ma.masked_invalid(fullmap3), cmap=cmap, norm=norm)
bm.colorbar(im,'bottom',cmap=cmap, norm=norm)
bm.drawcoastlines()
bm.fillcontinents(zorder=0)
ax3.axis([31500000, 32250000, 7250000, 7750000])

X_w = pca.components_. T
print X_w.shape

ind = np.arange(X_w.shape[0])  # the x locations for the groups
width = 1.0
            
ax4 = fig.add_subplot(234)
ax4.set_ylabel("Component 1")
ax4.bar(ind, X_w[:,0], width=width, color='#1B9E77', zorder=100)
ax4.axhline(color='k')
    
ax4.autoscale(tight=True)
ax4.set_xticks([])
    
ax5 = fig.add_subplot(235)
ax5.set_ylabel("Component 2")
ax5.bar(ind, X_w[:,1], width=width, color='#D95F02',zorder=100)
ax5.axhline(color='k')
ax5.autoscale(tight=True)
ax5.set_xticks([])
    
ax6 = fig.add_subplot(236)
ax6.set_ylabel("Component 3")
ax6.bar(ind, X_w[:,2], width=width, color='#7570B3', zorder=100)
ax6.axhline(color='k')
ax6.set_xticks(ind+width/2)
#ax3.set_xticklabels(labels, rotation=45, ha='center')
ax6.autoscale(tight=True)
    
ymin, ymax= ax1.get_ylim()
for a in [ax1, ax2, ax3]:
    a.vlines(ind, ymin, ymax, linestyle='--', color='lightgray')
        

canvas.print_figure('plot.png')

plt.show()
"""

