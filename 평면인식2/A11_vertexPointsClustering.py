import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN


def mergeVertex_DBSCAN(VertexPoints, hyperparameter):
    cluster_Points = np.array([[p.x, p.y, p.z] for p in VertexPoints])
    clustering = DBSCAN(eps = hyperparameter.eps_vertexPoint, min_samples = hyperparameter.min_samples_vertexPoint)
    labels = clustering.fit_predict(cluster_Points)

    vertexPointMap = defaultdict(list) #index = cluster 번호, index 안에는 포인트
    for i in range(len(VertexPoints)):
        if labels[i] != -1:
            vertexPointMap[labels[i]].append(VertexPoints[i])

    size = len(VertexPoints)
    negativeOneIndexes = set() #-1인 label들의 index
    for i in range(size):
        if labels[i] == -1:
            negativeOneIndexes.add(i)

    aftervertexPoints = []
    aftervertexLabel = []
    for i in range(size):
        if i not in negativeOneIndexes:
            aftervertexPoints.append(VertexPoints[i])
            aftervertexLabel.append(labels[i])
        
    return vertexPointMap, aftervertexPoints, aftervertexLabel
