#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 22:36:59 2019

@author: fahadtariq
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#using elbow method to find the optimal Number of clusters
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmean = KMeans(n_clusters = i , init= 'k-means++' , max_iter = 300 , n_init = 10, random_state=0)
    kmean.fit(X)
    wcss.append(kmean.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


kmean = KMeans(n_clusters = 5 , init= 'k-means++' , max_iter = 300 , n_init = 10, random_state=0)

y_kmean = kmean.fit_predict(X)

plt.scatter(X[y_kmean == 0,0],X[y_kmean ==0,1],s = 100 , c ='red',label = 'Cluster 1')
plt.scatter(X[y_kmean == 1,0],X[y_kmean ==1,1],s = 100 , c ='blue',label = 'Cluster 2')
plt.scatter(X[y_kmean == 2,0],X[y_kmean ==2,1],s = 100 , c ='green',label = 'Cluster 3')
plt.scatter(X[y_kmean == 3,0],X[y_kmean ==3,1],s = 100 , c ='black',label = 'Cluster 4')
plt.scatter(X[y_kmean == 4,0],X[y_kmean ==4,1],s = 100 , c ='yellow',label = 'Cluster 5')
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],s = 300 , c='green',label='Centroid')
plt.title('Clusters of cliets')
plt.xlabel('Annual_income$')
plt.ylabel('Spending Score(1-10')
plt.legend
plt.show()
