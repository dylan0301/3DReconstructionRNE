from collections import defaultdict
from A1_classes import *
from sklearn.metrics.pairwise import euclidean_distances

def boundaryRemoveNoise(BoundaryPoints, hyperparameter):
    size = len(BoundaryPoints)
    pointxyz = [[p.x, p.y, p.z] for p in BoundaryPoints]
    distMat = euclidean_distances(pointxyz, pointxyz)

    del_candidateIndex = set() #index들 집합
    for i in range(size):
        nearby = []
        for j in range(size):
            if distMat[i][j] <= hyperparameter.boundaryR:
                nearby.append(BoundaryPoints[j]) #자기자신도 포함
        if len(nearby) < hyperparameter.boundaryOutlierThreshold:
            del_candidateIndex.add(i)
    

    NewBoundaryPoints = []
    for i in range(size):
        if i not in del_candidateIndex:
            NewBoundaryPoints.append(BoundaryPoints[i])

    return NewBoundaryPoints

    
    
    
    
    distMat = defaultdict(dict)


    size = len(BoundaryPoints)
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (BoundaryPoints[i].distance(BoundaryPoints[j]),j) 

    #노이즈 제거
    del_candidate = [] #point들 리스트
    for i in range(size):
        l = distMat[i]
        res = sorted(l.values(), key = lambda x: x[0])
        if res[hyperparameter.boundaryOutlierThreshold][0] > hyperparameter.boundaryR:
            del_candidate.append(BoundaryPoints[i])

    BoundarySet = set(BoundaryPoints)

    for p in del_candidate:
        BoundarySet.remove(p)
    
    NewBoundaryPoints = []
    for p in BoundarySet:
        NewBoundaryPoints.append(p)

    return NewBoundaryPoints
    