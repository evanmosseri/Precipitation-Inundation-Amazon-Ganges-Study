import os
import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import scipy
import utils
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import colors
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from personalutils import *
import time

t = time.time()
windowRange = 180
sigmaRange = 180
# oidata,opdata = getAllData('Amazon')
oidata,opdata = getAllData('Mekong')
corrs = getCorrolationsAtPoints(oidata,opdata,range(1,windowRange+1),range(1,sigmaRange+1))
fig = plt.figure()

cmap= mcm.Blues
ax = fig.add_subplot(1,1,1)
ax.set_title("The Largest Corrolation is {} at Position {}".format(corrs.max(),np.unravel_index(corrs.argmax(), corrs.shape)))

norm = mcolors.BoundaryNorm(np.arange(0,1.1,.1), cmap.N)

im = ax.matshow(corrs,cmap="copper",norm=norm)
# fig.colorbar(im,ax=ax,cmap="copper")

# plt = ax.imshow(data, interpolation='nearest', cmap="copper", norm=norm)
x = np.arange(180)
# ax.plot(x, x*2, color='k')
ax.set_xlim((0,windowRange))
ax.set_ylim((sigmaRange,0))

ax.set_xlabel("Standard Deviation")
ax.set_ylabel("Window Size")
cbar = fig.colorbar(im, ax=ax, fraction=0.045,cmap="copper")

plt.savefig("./graphs/sigma_window_comparison{}by{}.png".format(windowRange,sigmaRange))
print time.time() - t