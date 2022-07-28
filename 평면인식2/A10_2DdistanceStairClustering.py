import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN


def divideEdge_DBSCAN(Edge, NewEdgePointMap, hyperparameter):
    NewEdgepm = NewEdgePointMap
    cluster_Points = np.array([[p.x, p.y, p.z] for p in Edge])
    clustering = DBSCAN(eps = hyperparameter.eps_edgePoint, min_samples = hyperparameter.min_samples_edgePoint)
    labels = clustering.fit_predict(cluster_Points)

    clusterNow = len(NewEdgepm)
    for i in range(len(Edge)):
        if labels[i] != -1:
            NewEdgepm[labels[i]+clusterNow].append(Edge[i]) 

    for i in range(max(labels) + 1):
        if len(NewEdgepm[i + clusterNow]) <= 10:
            del NewEdgepm[i + clusterNow]
            
    return NewEdgepm

def divideAllEdge_DBSCAN(NewEdgePointMap, edgePointMap, hyperparameter):
    for Edge in edgePointMap.values():
        temp = divideEdge_DBSCAN(Edge, NewEdgePointMap, hyperparameter)
        NewEdgePointMap = temp
    return NewEdgePointMap
