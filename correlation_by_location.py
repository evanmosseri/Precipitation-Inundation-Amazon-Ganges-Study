import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
import utils
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import colors
from personalutils import *
import scipy.stats as st
import math


deltaname = "Amazon"
datadir = "./data/"
inun_data = "inun_minremoved_v1v2.pkl"
precip_data = "delta_3B42_precip.pkl"

indicies_data = "{}{}_indicies.npy".format(datadir,deltaname)
indicies = np.load(indicies_data)

m = utils.gridEdges(datadir)

precip_frame = utils.get_data(datadir, precip_data,deltaname)
inun_frame = utils.get_data(datadir, inun_data,deltaname)

pdata = precip_frame['data']
idata = inun_frame['data']

end = min(pdata.index[-1], idata.index[-1])

iclip = idata[:end]
pclip = pdata[idata.index[0]:end:10]

correlations = []

# print type(iclip[0])

mask = ((np.isfinite(iclip)) & (np.isfinite(pclip)))
# iclip,pclip = iclip[mask],pclip[mask]
iclip,pclip = cleanNans(iclip,pclip)
printTwoLists(iclip.values,pclip.values)
for i in iclip.columns:
	x,y = gaussianMovingAverage(pclip[i],iclip[i],51,180)
	# x,y = iclip[i],pclip[i]
	correlations.append(st.pearsonr(x,y))

vmax = np.amax(precip_frame['data'].mean(axis=0))
vmin = np.amin(precip_frame['data'].mean(axis=0))
norm = colors.Normalize(vmin=vmin, vmax=vmax)

utils.plotGrid(plt.figure().add_subplot(1,1,1),precip_frame['data'].mean(axis=0),precip_frame,m,'Amazon','Hello World',cm.GMT_drywet,norm)
# plt.show()