import numpy as np
from sklearn.cluster import DBSCAN

from collections import defaultdict

def directionDBSCAN(EdgePoints, hyperparameter):
    Duplicatedvectors = np.array([p.direction for p in EdgePoints] + [(-1)*p.direction for p in EdgePoints])
    clustering = DBSCAN(eps = hyperparameter.eps_direction, min_samples = hyperparameter.min_samples_direction)
    edgelabels = clustering.fit_predict(Duplicatedvectors)

    
    oppositeVector = []
    for i in range(len(EdgePoints)):
        if edgelabels[i] != -1 and sorted([edgelabels[i], edgelabels[i+len(EdgePoints)]]) not in oppositeVector:
            oppositeVector.append(sorted([edgelabels[i], edgelabels[i+len(EdgePoints)]]))
 
    del_target_vec = [oppositeVector[i][0] for i in range(len(oppositeVector))]
    
    edgeNewLabel = [None] * len(EdgePoints)
    for i in range(len(edgelabels)):
        if edgelabels[i] not in del_target_vec:
            edgeNewLabel[i%len(EdgePoints)] = edgelabels[i]



    clusterPointMap = defaultdict(list) #index = cluster 번호, index 안에는 포인트
    for i in range(len(EdgePoints)):
        if edgeNewLabel[i] != -1:
            clusterPointMap[edgeNewLabel[i]].append(EdgePoints[i])

    #label -1인거 지우기
    size = len(EdgePoints)
    negativeOneIndexes = set() #-1인 label들의 index
    for i in range(size):
        if edgelabels[i] == -1:
            negativeOneIndexes.add(i)
    
    afterEdgePoints = []
    afterLabel = []
    for i in range(size):
        if i not in negativeOneIndexes:
            afterEdgePoints.append(EdgePoints[i])
            afterLabel.append(edgeNewLabel[i])

    return afterEdgePoints, clusterPointMap, afterLabel
