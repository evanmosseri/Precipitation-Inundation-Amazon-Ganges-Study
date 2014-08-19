import sys
import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import colors


def get(analysisType,data,row):
	ax = 0;
	if row:
		ax = 1;
		if analysisType == "mean":
			return data.mean(axis=ax)
		elif analysisType == "std":
			return data.std(axis=ax)
		elif analysisType == "sum":
			return data.sum(axis=ax)
	else:
		if analysisType == "mean":
			return data.mean(axis=ax)
		elif analysisType == "std":
			return data.std(axis=ax)
		elif analysisType == "sum":
			return data.sum(axis=ax)


rowOrColumn = sys.argv[1].lower()
analysisType = sys.argv[2].lower()
deltaName = sys.argv[3].title()

row = (rowOrColumn == "row")

datadir = 'data_old'
datafile = 'delta_3B42_precip.pkl'

delta = utils.get_data(datadir, datafile, deltaName)
results = get(analysisType,delta['data'],row);
f = open('output.txt','w')
for i in results:
	f.write(str(i)+"\n")
f.close()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
# ax1.plot(np.arange(len(results)),results)

# i dont know what I am doing starting over here
globe = utils.gridEdges(datadir)

cmap = cm.GMT_drywet
# cmap = "copper"
vmin = np.nanmin(results.values)
vmax = np.nanmax(results.values)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
bm = utils.basemap(ax1)
X, Y = bm(globe['lons'], globe['lats'])

#results = np.array(results)
ax1.axis(utils.mapbounds[deltaName])
ax1.set_title("Precipitation {} {}s by location".format(deltaName,analysisType))
globe['map'][delta['inds'][0], delta['inds'][1]] = results


im = bm.pcolormesh(X, Y, np.ma.masked_invalid(globe['map']),cmap=cmap, norm=norm)
cbar = bm.colorbar(im, "bottom", cmap=cmap, norm=norm)  
plt.savefig("./graphs/basic stats/inun_{}_{}s_{}.png".format(deltaName,rowOrColumn,analysisType))

