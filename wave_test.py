#!/usr/env/bin python3

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA
import hdbscan
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='MPCA modelling of the sine wave')

parser.add_argument('-d', '--dataset', default='sine', choices= ['sine', 'cosine'])

parser.add_argument('-c', '--clus', default = 'hdb', choices= ['hdb', 'db','db-hdb', 'kmeans'])

parser.add_argument('-a', '--alg', default='', choices=['kPCA', 'AE', 'Arima'])

parser.add_argument('-l', '--arglist',default=[1e-02, 50, 100, 4], nargs = 4)

args = parser.parse_args()

tw = tuple(args.arglist)
epsilon, min_samp, min_csize, cents = args.arglist

fig, ax = plt.subplots(3)
#wave data
freq = 8e03
cycles = 5
sample_freq = 8e03

x = np.arange(sample_freq)

if (args.dataset == 'sine'):
	y = np.sin(2* np.pi*cycles *x / freq)

else:
	y = np.sin(2*np.pi*cycles *x /freq + np.pi/2)

noise = np.random.normal(0,0.12,8000)
y_noise = y+noise+10

ax[0].scatter(x, y_noise,s =1)
#plt.show()


#clustering

y_fit = np.array(y_noise).reshape(-1,1)

if args.clus == 'db': 

	clustering = DBSCAN(eps = epsilon, min_samples=min_samp).fit(y_fit)

elif args.clus == 'hdb':
	clustering = hdbscan.HDBSCAN(min_cluster_size=min_csize, min_samples= min_samp).fit(y_fit)

elif args.clus == 'db-hdb':
	clustering = hdbscan.HDBSCAN(min_cluster_size=min_csize, min_samples = min_samp, cluster_selection_epsilon = epsilon).fit(y_fit)

elif args.clus == 'kmeans':
	clustering = KMeans(n_clusters = cents).fit(y_fit)


#algorithms

#mpca
err = []	
for name in list(set(clustering.labels_)):
	
	if (name == -1):
		continue
	else:
		blob = [y_fit[i] for i, n in enumerate(clustering.labels_) if n == name]
		keys = [j for j, m in enumerate(clustering.labels_) if m ==name]
		blob_train = np.array(blob).reshape(-1,1)
		pca = PCA(n_components =0.9)
		blob_trfmd = pca.fit_transform(blob_train)
		blob_inv = pca.inverse_transform(blob_trfmd)	
		err.append(1/len(blob_inv) * np.sum(np.sqrt(blob_train, blob_inv)))
		ax[1].scatter(keys, blob_inv, s=1, label = name)
		
print('Error = {}'.format(sum(err)))
ax[1].legend()

#kpca

if args.alg == 'kPCA':
	kpca = KernelPCA(n_components=None, kernel='rbf', fit_inverse_transform=True)
	trfmd_alg = kpca.fit_transform(y_fit)	
	inv_alg = kpca.inverse_transform(trfmd_alg)
	ax[2].scatter(x, inv_alg, s=1)
	kerr =1/len(inv_alg) * np.sum(np.sqrt(y_fit, inv_alg))		
	print('Error = {}'.format(kerr))

plt.show()

print(set(clustering.labels_))
#spca
