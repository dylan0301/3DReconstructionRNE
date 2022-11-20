#A6_3DdistanceStairClustering.py

import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN


def dividePlane_DBSCAN(cluster, NewClusterPointMap, hyperparameter):
    Newclusterpm = NewClusterPointMap
    cluster_Points = np.array([[p.x, p.y, p.z] for p in cluster])
    clustering = DBSCAN(eps = hyperparameter.eps_centerPoint, min_samples = hyperparameter.min_samples_centerPoint)
    labels = clustering.fit_predict(cluster_Points)

    clusterNow = len(Newclusterpm)
    for i in range(len(cluster)):
        if labels[i] != -1:
            Newclusterpm[labels[i]+clusterNow].append(cluster[i]) 

    for i in range(max(labels) + 1):
        if len(Newclusterpm[i + clusterNow]) <= 0:
            del Newclusterpm[i + clusterNow]
            
    return Newclusterpm

def divideAllPlane_DBSCAN(NewClusterPointMap, clusterPointMap, hyperparameter):
    for Cluster in clusterPointMap.values():
        temp = dividePlane_DBSCAN(Cluster, NewClusterPointMap, hyperparameter)
        NewClusterPointMap = temp
    return NewClusterPointMap
