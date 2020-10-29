#!/usr/env/bin python3

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA
from sknn.ae import Layer, AutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hdbscan
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='MPCA modelling of the sine wave')

parser.add_argument('-d', '--dataset', default='sine', choices= ['sine', 'cosine', 'noise'])

parser.add_argument('-c', '--clus', default = 'hdb', choices= ['hdb', 'db','db-hdb', 'kmeans'])

parser.add_argument('-a', '--alg', default='', choices=['kPCA', 'AE', 'Arima'])

parser.add_argument('-l', '--arglist',default=[1e-02, 50, 100, 4], nargs = 4)

args = parser.parse_args()

tw = tuple(args.arglist)
epsilon, min_samp, min_csize, cents = args.arglist
np.random.seed(19234)
fig, ax = plt.subplots(3)
#wave data
freq = 8e03
cycles = 10
sample_freq = 8e03

x = np.arange(sample_freq)

if (args.dataset == 'sine'):
	y = np.sin(2* np.pi*cycles *x / freq)

elif(args.dataset== 'cosine'):
	y = np.sin(2*np.pi*cycles *x /freq + np.pi/2)

else:
	nse1 = np.random.randn(len(x))
	y = np.sin(2*np.pi*cycles*x) +nse1 

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
		
print('multi pca Error = {}'.format(sum(err)))
ax[1].legend()

#kpca

if args.alg == 'kPCA':
	kpca = KernelPCA(n_components=None, kernel='rbf', fit_inverse_transform=True)
	trfmd_alg = kpca.fit_transform(y_fit)	
	inv_alg = kpca.inverse_transform(trfmd_alg)
	ax[2].scatter(x, inv_alg, s=1)
	kerr =1/len(inv_alg) * np.sum(np.sqrt(y_fit, inv_alg))		
	print('kernel pca error = {}'.format(kerr))

#autoencoder

if args.alg=='AE':
	ip = torch.tensor(y_fit, requires_grad =True, dtype=torch.float)
	class autoencoder(nn.Module):
		
		def __init__(self):
			super(autoencoder, self).__init__()
			
			self.lin1 = nn.Linear(1, 10)
			self.lin2 = nn.Linear(10,1)
		def forward(self, x):
			x = F.tanh(self.lin1(x))
			x = F.tanh(self.lin2(x))
			return x

	ae = autoencoder()
	optim = optim.SGD(ae.parameters(), lr = 0.01) 	
	lmet = nn.MSELoss()
	for ep in range(10):
		t_loss = 0.0
		optim.zero_grad()
		preds = ae(ip)
		loss = lmet(ip, ip)
		loss.backward()
		optim.step()
		t_loss += loss
	ax[2].scatter(x, preds.detach().numpy(), s=1)
	print("Autoencoder error = {}".format(t_loss.item()))

plt.show()

print(set(clustering.labels_))
#spca
