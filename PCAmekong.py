import pickle
from matplotlib import figure
import numpy as np

from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt 
import os
from mpl_toolkits.basemap import Basemap,cm
from matplotlib import colors
import calendar

import utils
from cmap import colors as cdict


def preprocess(delta_data):    
    rawMatrix = np.vstack((delta_data[cf] for cf in delta_data)).T
    cleanrow = []
    cleanind = []
    for i,row in enumerate( rawMatrix):
        if not np.isnan(row).any():
            cleanrow.append(row)
            cleanind.append(i)
    cleanMatrix = np.vstack(cleanrow)
    return cleanMatrix, cleanind, delta_data 


if __name__ == '__main__':
    datadir = 'data'
    datafile = 'inun_minremoved_v1v2.pkl'
    deltaname = 'Mekong'
    components = 10

    delta = utils.get_data(datadir,datafile, deltaname)


    MekongMat, cleanind, delta_data = preprocess(delta['data'])
    
    dates = delta_data.index[:]

    pca = PCA(n_components=components)
    x_pca = pca.fit_transform(MekongMat)
   
    pca_fill = np.ones((delta_data.shape[0],components))*np.nan
    pca_fill[cleanind] = x_pca
    
    #plot projections=timeseries
    fig = plt.figure(figsize=(20,8))
        

    pca_weights = pca.components_.T
    bar_locations = np.arange(pca_weights.shape[0])
    width=1.0
    
    map = utils.gridEdges(datadir)
    # fill map with nans
    fullmap1 = map['map'].copy()
    fullmap2 = map['map'].copy()
    fullmap3 = map['map'].copy()

    # get data, indicies, and insert data into full grid
    di0, di1 = delta['inds'][0], delta['inds'][1]

    fullmap1[di0, di1] = pca_weights[:,0]
    fullmap2[di0, di1] = pca_weights[:,1]
    fullmap3[di0, di1] = pca_weights[:,2]

    vmax = np.nanmax(abs(pca_weights))
    vmin = -vmax

    norm = colors.normalize(vmin=vmin, vmax=vmax)

    cmap = 'BrBG'

    ax1 = fig.add_subplot(211) 
    base = utils.basemap(ax1)
    ax1.axis([31500000, 32250000, 7250000, 7750000]) 
    x, y = base(map['lons'], map['lats'])
    im = base.pcolormesh(x, y, np.ma.masked_invalid(fullmap1), cmap=cmap, norm=norm)
    cbar = base.colorbar(im,'bottom', cmap=cmap, norm=norm) 
    ax1.set_title('Component 1 Weights')
    ax1.set_xlim((bar_locations.min(),bar_locations.max()))
    
    ax = fig.add_subplot(212)
    ax.plot_date(dates , pca_fill[:,0], '-', label='Component 1')
    ax.plot_date(dates , pca_fill[:,1], '-', label='Component 2')
    ax.plot_date(dates , pca_fill[:,2], '-', label='Component 3')
    ax.set_title('PCA projections inundation')
    ax.legend(loc=8, ncol=3)
    ax.set_xlim((min(dates),max(dates)))


    fig.savefig('inunplot.png') 
    """
    #fig2 = figure.Figure()
    fig2 = plt.figure()
    canvas = FigureCanvas(fig2)
    ax3 = fig2.add_subplot(111)
    bm = utils.basemap(ax3)
    x, y = bm(map['lons'], map['lats'])
    ax3.axis([31500000, 32250000, 7250000, 7750000]) 
    
    im = bm.pcolormesh(x, y, np.ma.masked_invalid(fullmap1), 
                       cmap=cmap, norm=norm)
    cbar = bm.colorbar(im,'bottom', cmap=cmap, norm=norm)
    ax3.set_title('mekong inundation component 1')
    fig2.savefig('{}.png'.format(deltaname))
    #canvas.print_figure('{}.png'.format(deltaname))
    """
    #plot fall off
    eigenList = pca.explained_variance_ratio_

    fig3 = plt.figure()
    
    ax4 = fig3.add_subplot(111)
    ax4.plot(eigenList, '-')
    ax4.set_title('Fall off of PCA Components')
    fig3.savefig('{}.png'.format('falloff'))

    #scatter plot of component 1 vs component 2
    component1 = pd.Series(pca_fill[:,0] , dates)
    component2 = pd.Series(pca_fill[:,1] , dates)

   

    fig4 = plt.figure(figsize=(20,10))
    
    ax5 = fig4.add_subplot(111)
    ax5.plot(component1,component2,'--', color='darkgray')
    
    ax5.set_title('Component 1 vs Component 2')
    ax5.set_xlabel('Component 1')
    ax5.set_ylabel('Component 2')
    step = 13
    for i in range(1,step):
        c1 = pca_fill[(dates.month==i), 0]
        c2 = pca_fill[(dates.month==i), 1]
        ax5.scatter(c1, c2, c=cdict[i],s=100, zorder=500, label=calendar.month_name[i])  
    ax5.legend(loc=8, ncol=4, scatterpoints=1)
    fig4.savefig('{}.png'.format('components'))


