import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import utils

datadir = 'data'
datafile = 'delta_3B42_precip.pkl'
deltaName = "Amazon"

delta = utils.get_data(datadir, datafile, deltaName)
col1 = delta['data'].iloc[:,0]

col2 = delta['data'].iloc[:,1]

gap = 5

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(col1,col2)
ax.set_title("Column Correlation")
print stats.pearsonr(col1,col2)

plt.savefig("column_correlation.png")