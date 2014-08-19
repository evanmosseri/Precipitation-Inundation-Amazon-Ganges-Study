import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import utils

datadir = 'data'
datafile = 'delta_3B42_precip.pkl'
deltaName = "Mekong"

delta = utils.get_data(datadir, datafile, deltaName)
col = delta['data'].iloc[:,0]

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# y = (1.2* col ) + (.5* np.random.randn(len(col)))
# ax.scatter(col,y)
# print stats.pearsonr(col,y)
# plt.show()
gap = 5

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(col[0:-gap],col[gap:])

print stats.pearsonr(col[0:-gap],col[gap:])
plt.show()