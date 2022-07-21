import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN

def divideCluster_DBSCAN(cluster, NewClusterPointMap, hyperparameter):
    Newclusterpm = NewClusterPointMap
    cluster_Points = np.array([[p.x, p.y, p.z] for p in cluster])
    clustering = DBSCAN(eps = hyperparameter.eps_point, min_samples = hyperparameter.min_samples_point)
    labels = clustering.fit_predict(cluster_Points)

    clusterNow = len(Newclusterpm)
    for i in range(len(cluster)):
        if labels[i] != -1:
            Newclusterpm[labels[i]+clusterNow].append(cluster[i]) 

    return Newclusterpm




def divideAllCluster_DBSCAN(NewClusterPointMap, clusterPointMap, hyperparameter):
    for Cluster in clusterPointMap.values():
        temp = divideCluster_DBSCAN(Cluster, NewClusterPointMap, hyperparameter)
        NewClusterPointMap = temp
    return NewClusterPointMap