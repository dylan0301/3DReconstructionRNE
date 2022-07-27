import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from collections import defaultdict


def vectorHierarchicalClustering(CenterPoints, hyperparameter):
    numOfCluster = 2*hyperparameter.numOfCluster

    Duplicatedvectors = np.array([p.normal for p in CenterPoints] + [(-1)*p.normal for p in CenterPoints])
    ac = AgglomerativeClustering(n_clusters=numOfCluster, affinity="euclidean", linkage="complete")
    labels = ac.fit_predict(Duplicatedvectors)

    #print(list(labels))
    
    oppositeVector = []
    for i in range(len(CenterPoints)):
        if sorted([labels[i], labels[i+len(CenterPoints)]]) not in oppositeVector:
            oppositeVector.append(sorted([labels[i], labels[i+len(CenterPoints)]]))
 
    del_target_vec = [oppositeVector[i][0] for i in range(len(oppositeVector))]
    
    newLabel = [None] * len(CenterPoints)
    for i in range(len(labels)):
        if labels[i] not in del_target_vec:
            newLabel[i%len(CenterPoints)] = labels[i]

    clusterPointMap = defaultdict(list) #index = cluster 번호, index 안에는 포인트
    for i in range(len(CenterPoints)):
        clusterPointMap[newLabel[i]].append(CenterPoints[i])
    return CenterPoints, clusterPointMap, newLabel

def vectorDBSCAN(CenterPoints, hyperparameter):
    Duplicatedvectors = np.array([p.normal for p in CenterPoints] + [(-1)*p.normal for p in CenterPoints])
    clustering = DBSCAN(eps = hyperparameter.eps_vector, min_samples = hyperparameter.min_samples_vector)
    labels = clustering.fit_predict(Duplicatedvectors)

    
    oppositeVector = []
    for i in range(len(CenterPoints)):
        if labels[i] != -1 and sorted([labels[i], labels[i+len(CenterPoints)]]) not in oppositeVector:
            oppositeVector.append(sorted([labels[i], labels[i+len(CenterPoints)]]))
 
    del_target_vec = [oppositeVector[i][0] for i in range(len(oppositeVector))]
    
    newLabel = [None] * len(CenterPoints)
    for i in range(len(labels)):
        if labels[i] not in del_target_vec:
            newLabel[i%len(CenterPoints)] = labels[i]



    clusterPointMap = defaultdict(list) #index = cluster 번호, index 안에는 포인트
    for i in range(len(CenterPoints)):
        if newLabel[i] != -1:
            clusterPointMap[newLabel[i]].append(CenterPoints[i])

    #label -1인거 지우기
    size = len(CenterPoints)
    negativeOneIndexes = set() #-1인 label들의 index
    for i in range(size):
        if labels[i] == -1:
            negativeOneIndexes.add(i)
    
    afterCenterPoints = []
    afterLabel = []
    for i in range(size):
        if i not in negativeOneIndexes:
            afterCenterPoints.append(CenterPoints[i])
            afterLabel.append(newLabel[i])

    return afterCenterPoints, clusterPointMap, afterLabel
