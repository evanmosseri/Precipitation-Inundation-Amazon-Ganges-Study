import os
import pickle
from math import isnan
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap, cm

def clean(l1,l2):
	X = []
	Y = []
	for i in zip(l1,l2):
		if isnan(i[0]) != True and isnan(i[1]) != True:
			X.append(i[0])
			Y.append(i[1])
	return X,Y
datadir = "data_old"
datafile = "delta_byName_time_series.v2.pkl"

with open(os.path.join(datadir, datafile ), 'r') as f:
        alldeltas = pickle.load(f)
inun = alldeltas["inundation"]["Amazon"]
precip = alldeltas["precip"]["Amazon"]
inun,precip =  clean(inun,precip)

print inun[0:20]
print precip[0:20]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title("Correlation Between Precipitation and Inundation",y=1.04)
ax.scatter(inun,precip)
ax.set_xlabel("Inundation")
ax.set_ylabel("Precipitation")
ax.set_ylim([min(precip[0:20]), max(precip[0:20])])


# plt.show()
plt.savefig("./graphs/multirivercorrelation.png")