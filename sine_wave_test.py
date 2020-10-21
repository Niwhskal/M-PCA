#!/usr/env/bin python3

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import hdbscan
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='MPCA modelling of the sine wave')

parser.add_argument('-d', '--dataset', default='sine', choices= ['sine', 'cosine'])

parser.add_argument('-c', '--clus', default = 'hdb', choices= ['hdb', 'db','db-hdb', 'kmeans'])

parser.add_argument('-a', '--alg', default='SPCA', choices=['kPCA', 'AE', 'Arima'])

args = parser.parse_args()

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
y_noise = y+noise

#plt.scatter(x,y_noise, s=1)
#plt.show()


#clustering

y_fit = np.array(y_noise).reshape(-1,1)

if args.clus == 'db': 

	clustering = DBSCAN(eps = 1e-02, min_samples=30).fit(y_fit)

elif args.clus == 'hdb':
	clustering = hdbscan.HDBSCAN(min_cluster_size=100, min_samples = 1).fit(y_fit)

elif args.clus == 'db-hdb':
	clustering = hdbscan.HDBSCAN(min_cluster_size=30, min_samples = 1, cluster_selection_epsilon = 1e-02).fit(y_fit)

elif args.clus == 'kmeans':
	clustering = KMeans(n_clusters = 5).fit(y_fit)


#algorithms

#mpca

zipped = list(zip(x, zip(y_fit, clustering.labels_)))
grouped = sorted(zipped, key=lambda zipped : zipped[1][1])
		
for name in list(set(clustering.labels_)):
	
	if (name == -1):
		continue
	else:
		blob = [(i,y_fit[i]) for n, i in enumerate(clustering.labels_) if n == name]
		d = dict(blob)
		blob_train = np.array(list(d.values())).reshape(-1,1)
		pca = PCA(n_components =0.93)
		blob_trfmd = pca.fit_transform(blob_train)
		blob_inv = pca.inverse_transform(blob_trfmd)	
		plt.scatter(list(d.keys()), blob_inv, s=1, c=name/len(set(clustering.labels_)))
		
plt.show()

print(set(clustering.labels_))
#spca
"""
spca = PCA(n_components = 0.93).fit_transform(y_fit)

y_inv = spca.inverse_transform(y_fit)



colors = [(col+1)/len(set(clustering.labels_)) if col>=0 else 0 for col in clustering.labels_] 

plt.scatter(x, y_fit[:,0], s=1, c=colors)
plt.show() """
