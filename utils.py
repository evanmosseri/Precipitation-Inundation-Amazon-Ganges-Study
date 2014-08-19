import os
import pickle

import numpy as np
from sklearn.decomposition import PCA, FastICA,ProjectedGradientNMF
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
    
def preprocess(delta_data):    
    rawMatrix = np.vstack((delta_data[cf] for cf in delta_data)).T
    cleanrow = []
    cleanind = []
    for i,row in enumerate( rawMatrix):
        if not np.isnan(row).any():
            cleanrow.append(row)
            cleanind.append(i)
    cleanMatrix = np.vstack(cleanrow)
    processed_data = {'cleanMatrix':cleanMatrix, 
                      'cleanind': cleanind, 
                      'delta_data': delta_data
                      }
    return processed_data
    
def calcPCA(delta_data, components):

    data = preprocess(delta_data)
    pca = PCA(n_components=components)
    x_pca = pca.fit_transform(data['cleanMatrix'])
   
    pca_fill = np.ones((delta_data.shape[0],components))*np.nan
    pca_fill[data['cleanind']] = x_pca
    pca_weights = pca.components_.T
    delta_pca = {'transform':pca_fill,
                 'weights' : pca_weights,
                 'eigenValues' : pca.explained_variance_ratio_}
    return delta_pca

def calcICA(delta_data, components):

    data = preprocess(delta_data)
    ica = FastICA(n_components=components)
    x_ica = ica.fit_transform(data['cleanMatrix'])
   
    ica_fill = np.ones((delta_data.shape[0],components))*np.nan
    ica_fill[data['cleanind']] = x_ica
    ica_weights = ica.components_.T
    delta_ica = {'transform':ica_fill,
                 'weights' : ica_weights,
                }
    return delta_ica

def calcNMF(delta_data, components):

    data = preprocess(delta_data)
    nmf = ProjectedGradientNMF(n_components=components)
    x_nmf = nmf.fit_transform(data['cleanMatrix'])
   
    nmf_fill = np.ones((delta_data.shape[0],components))*np.nan
    nmf_fill[data['cleanind']] = x_nmf
    nmf_weights = nmf.components_.T
    delta_nmf = {'transform':nmf_fill,
                 'weights' : nmf_weights,
                }
    return delta_nmf

def plotGrid(ax, data, inds, globalMap, deltaName, title, cmap, norm):

    fullmap1 = globalMap['map'].copy()    
    di0, di1 = inds['inds'][0], inds['inds'][1]
    fullmap1[di0, di1] = data
    bm = basemap(ax)
    ax.axis(mapbounds[deltaName]) 
    x, y = bm(globalMap['lons'], globalMap['lats'])
    im = bm.pcolormesh(x, y, np.ma.masked_invalid(fullmap1), cmap=cmap, norm=norm)
    cbar = bm.colorbar(im,'bottom', cmap=cmap, norm=norm)
    # cbar.set_fontsize(3)
  
 
    ax.set_title(title)
