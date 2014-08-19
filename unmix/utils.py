import os
import pickle

import numpy as np

from mpl_toolkits.basemap import Basemap, cm

fullname = {'i':'Inundation', 
             'p': 'Precipitation'}
             
datafiles = {'i':'inun_minremoved_v1v2.pkl', 
             'p':'delta_3B42_precip.pkl'} 

cmap = {'i': cm.GMT_drywet, 
         'p': cm.s3pcpn_l}

(0.0, 40031606.158245593)
(0.0, 12742456.0)

mapbounds = {'Mekong': [31500000, 32250000, 7250000, 7750000], 
             'Amazon': [14000000, 15000000, 5750000, 6750000], 
             'Ganges': [29500000, 30500000, 8500000, 9250000]}
         
def get_data(datadir, datafile, delta):
    """Pulls inundation data""" 
    delta = delta.lower().capitalize()
    delta_inds = np.load(os.path.join(datadir, 
                         "{}_indicies.npy".format(delta)))
    with open(os.path.join(datadir, datafile ), 'r') as f:
        alldeltas = pickle.load(f)
        
    return {'data':alldeltas[delta], 
             'inds':delta_inds}
             
def gridEdges(datadir):
    """
    estimate cell boundaries from cell centers, 
    important for correct pcolormesh plotting
    """
    lons = np.load(os.path.join(datadir, 'lons.npy'))
    lats = np.load(os.path.join(datadir, 'lats.npy'))
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
    
    #fill map with nans
    fullmap = np.ones(lons.shape)*np.nan
    
    return {'lats':LATedge, 'lons':LONedge, 'map':fullmap}
    
def basemap(ax):
    """Basemap object for inundation data
    """    
    bm = Basemap(projection='cea', 
                 lon_0=0, lat_0=30, 
                 rsphere=6371228,
                 resolution='l', 
                 ax=ax)
    bm.drawcoastlines()
    bm.fillcontinents(zorder=0, color='.9')
    return bm
