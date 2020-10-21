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
clustering = DBSCAN(eps = 1e-02, min_samples=1).fit(y_fit)


cluster_colors = [sns.desaturate(palette[col], sat) if col >= 0 else (0.5, 0.5, 0.5) for col, sat in zip(clustering.labels_, clustering.probabilities_)]


plt.scatter(y_fit[0], y_fit[1], c=clustering_colors, **plot_kwds)
print(set(clustering.labels_))
