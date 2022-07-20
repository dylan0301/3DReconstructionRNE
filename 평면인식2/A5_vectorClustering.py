import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

def vectorClustering(CenterPoints, hyperparameter):
    numOfCluster = hyperparameter.numOfCluster

    Duplicatedvectors = np.array([p.normal for p in CenterPoints] + [(-1)*p.normal for p in CenterPoints])
    ac = AgglomerativeClustering(n_clusters=numOfCluster, affinity="euclidean", linkage="complete")
    labels = ac.fit_predict(Duplicatedvectors)
    

    oppositeVector = []
    for i in range(len(CenterPoints)):
        if sorted([labels[i], labels[i+len(CenterPoints)]]) not in oppositeVector:
            oppositeVector.append(sorted([labels[i], labels[i+len(CenterPoints)]]))

    #print(oppositeVector)
        
    del_target_vec = [oppositeVector[i][0] for i in range(len(oppositeVector))]
    
    #print(del_target_vec)
    
    newLabel = [None] * len(CenterPoints)
    for i in range(len(labels)):
        if labels[i] not in del_target_vec:
            newLabel[i%len(CenterPoints)] = labels[i]

    clusterPointMap = defaultdict(list) #index = cluster 번호, index 안에는 포인트
    for i in range(len(CenterPoints)):
        clusterPointMap[newLabel[i]].append(CenterPoints[i])
    return clusterPointMap