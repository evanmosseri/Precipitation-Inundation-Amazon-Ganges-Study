import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.animation as animation
from matplotlib import colors

import utils

def make_movie(datadir, deltaname, variable, framenum):
    
    datafile = utils.datafiles[variable]
    delta = utils.get_data(datadir, datafile, deltaname)
    mp = utils.gridEdges(datadir)

    cmap = utils.cmap[variable]
    vmin = np.nanmin(delta['data'])
    vmax = np.nanmax(delta['data'])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bm = utils.basemap(ax)
    
    X, Y = bm(mp['lons'], mp['lats'])
    print ax.get_xlim()
    print ax.get_ylim()
    
    ax.axis(utils.mapbounds[deltaname])


    def updatefig(i):
        mp['map'][delta['inds'][0], delta['inds'][1]] = delta['data'].iloc[i,:]
        date = delta['data'].index[i].strftime('%Y-%m-%d')
        im = bm.pcolormesh(X, Y, np.ma.masked_invalid(mp['map']),
                           cmap=cmap, norm=norm)
        cbar = bm.colorbar(im, "bottom", cmap=cmap, norm=norm)  
        ax.set_title("{}: {}".format(utils.fullname[variable], date))
    
        framenum = 5
    ani = animation.FuncAnimation(fig, updatefig, frames=framenum)
    ani.save('{}_{}_{}.mp4'.format(utils.fullname[variable],
                                       deltaname, framenum))


if __name__ == '__main__':
    datadir = 'data'
    deltaname = 'Mekong'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('delta', choices=['Amazon', 'Ganges','Mekong'])
    parser.add_argument('variable', choices=['i', 'p']) 
    parser.add_argument('-f', default=5)
    args = parser.parse_args()
    
    make_movie(datadir, args.delta, args.variable, args.f)
    