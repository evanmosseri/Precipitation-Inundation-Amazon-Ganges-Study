import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import utils

figure = plt.figure()

ax1 = figure.add_subplot(2,2,1)
ax1.plot([1,2,3,4],[.5,7.5,6,2])
ax1.plot([1,2,3,4],[-.5,-7.5,-6,-2])

ax2 = figure.add_subplot(2,2,2)
bm = utils.basemap(ax2)
ax3 = figure.add_subplot(2,2,3)

ax4= figure.add_subplot(2,2,4)

plt.show()