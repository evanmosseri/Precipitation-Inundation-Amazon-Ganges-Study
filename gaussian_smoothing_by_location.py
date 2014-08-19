from personalutils import *

oidata,opdata = getAllData('Amazon')

locs = len(oidata.iloc[0,:])

idata = []
pdata = []

for i in range(locs):
	idata.append(oidata.iloc[i,:])
	pdata.append(opdata.iloc[i,:])

print idata[1]
