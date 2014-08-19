import os
import pandas as pd

precip_pkl = os.path.join('precip_delta_maps', 'delta_3B42_precip.pkl')
inun_pkl = os.path.join('inun_delta_maps', 'inun_minremoved_v1v2.pkl')

precip = pd.load(precip_pkl)
inun = pd.load(inun_pkl)

#reconstruct

#weights -> fit()
#transforms -> transform()
#original data = X

#reconstruct = pca.mean_ + np.dot(Xtransform, pca.components.T))