import numpy as np
import pandas as pd
import os
import scipy.stats as st
def cleanNans(a,b):
	mask = np.isfinite(a) & np.isfinite(b)
	return a[mask],b[mask]
def cleanNansSingle(a):
	mask = np.isfinite(a)
	return a[mask]
def addScatter(fig,plt,x,y,a,b,c):
	ax = fig.add_subplot(a,b,c)
	ax.scatter(x,y)
	return ax
def addPlot(fig,plt,x,y,a,b,c):
	ax = fig.add_subplot(a,b,c)
	ax.plot(x,y)
	return ax
def getAllData(river):
	dpath = 'data'
	ifile = 'inun_minremoved_v1v2.pkl'
	pmfile = 'delta_3B42_precip.pkl'

	idata = pd.read_pickle(os.path.join(dpath, ifile))
	pmdata = pd.read_pickle(os.path.join(dpath, pmfile))
	return idata['Amazon'],pmdata['Amazon']

def getSomeData(river,delta):
	dpath = 'data'
	ifile = 'inun_minremoved_v1v2.pkl'
	pmfile = 'delta_3B42_precip.pkl'

	idata = pd.read_pickle(os.path.join(dpath, ifile))
	pmdata = pd.read_pickle(os.path.join(dpath, pmfile))
	return idata['Amazon'],pmdata['Amazon']
def gaussianMovingAverage(pdata,idata, windowSize, StDev):
	precip = pd.rolling_window(pdata.mean(axis=1), window=windowSize, win_type='gaussian', std=StDev)
	end = min(precip.index[-1], idata.index[-1])
	iclip = idata.mean(axis=1)[:end]
	pclip = precip[iclip.index[0]:end:10]
	#np.testing.assert_array_equal(iclip.index, pclip.index)
	return pclip, iclip

def getCorrolationsAtPoints(oidata,opdata,windowRange,sigmaRange):
	# t = 0;
	vals = np.empty((len(windowRange), len(sigmaRange)))*np.nan
	for i in windowRange:
		for x in sigmaRange:
			# t+=1;
			pdata, idata = gaussianMovingAverage(opdata, oidata, i, x)
			mask = (np.isfinite(idata) & np.isfinite(pdata))
			# idata,pdata = cleanNans(idata,pdata)
			# np.testing.assert_array_equal(idata[mask].index, pdata[mask].index)
			corr = st.pearsonr(idata[mask],pdata[mask])[0]
			vals[i-1,x-1] = corr
			# print "'%f' percent complete" % (float(i)/(180*180))
	return vals	

def getCorrolationsAtPoint(oidata,opdata,windowRange,sigmaRange):
	vals = []
	for i in windowRange:
		for x in sigmaRange:
			pdata, idata = gaussianMovingAverage(opdata,oidata, i, x)
			idata,pdata = cleanNans(idata,pdata)
			corr = st.pearsonr(idata,pdata)[0]
			vals.append(corr)
	return vals
def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
        
    return L
def printTwoLists(l1,l2):
	for i in range(max(len(l1),len(l2))):
		if (i < len(l1)) & (i < len(l2)):
			print str(i)+": " + str(l1[i]) + ", " + str(l2[i])
		else:
			if(i<len(l1)):
				print str(i)+":" + str(l1[i])
			elif(i < len(l2)):
				print str(i)+":" + str(l2[i])
			else:
				break     