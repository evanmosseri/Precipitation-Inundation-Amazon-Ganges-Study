import os
import pandas as pd
import numpy as np
import scipy.stats as st
from personalutils import *
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import scipy
import utils
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import colors

fig = plt.figure()
oidata,opdata = getAllData('Amazon')
windowRange = 50
sigmaRange = 1
x = range(windowRange)
y = getCorrolationsAtPoint(oidata,opdata,range(1,windowRange+1),range(sigmaRange,sigmaRange+1))
# print y
ax = addPlot(fig,plt,x,y,2,1,1)
ax.set_title("Shifting Window Size With Sigma of 180")
ax.set_ylim(0,1)
ax.set_ylabel("Correlation")
windowRange = 10
sigmaRange = 180
x = range(sigmaRange)
y = getCorrolationsAtPoint(oidata,opdata,range(windowRange,windowRange+1),range(1,sigmaRange+1))
ax2 = addPlot(fig,plt,x,y,2,1,2)
ax2.set_title("Shifting Sigma With Window Size of 50")
ax2.set_ylim(0,1)
ax2.set_ylabel("Correlation")
plt.savefig("./graphs/Window_and_Standard_Deviation_Variations_titled.png")