# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:35:33 2023

@author: jwark
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv(r"C:\Users\jwark\Downloads\archive (1)\Mall_Customers.csv")
data.head()
# Select relevant columns
X = data.iloc[:, [3, 4]].values

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X)
data['clust']=clusters
df=data.iloc[:,[5,3,4]]
df
#here we formed the cluster
df.iloc[:,2:4].groupby(df.clust).mean()

# Visualize the clusters
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], s=100, c=colors[i], label=f'Cluster {i + 1}')