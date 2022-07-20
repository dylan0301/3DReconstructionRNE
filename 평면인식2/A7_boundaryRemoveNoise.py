from collections import defaultdict
from A1_classes import *

def boundaryRemoveNoise(BoundaryPoints, hyperparameter):
    size = len(BoundaryPoints)
    distMat = defaultdict(dict)
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (BoundaryPoints[i].distance(BoundaryPoints[j]),j) 

    #노이즈 제거
    del_candidate = [] #index들 리스트
    for i in range(size):
        l = distMat[i]
        res = sorted(l.values(), key = lambda x: x[0])
        if res[hyperparameter.boundaryOutlierThreshold][0] > hyperparameter.boundaryR:
            del_candidate.append(BoundaryPoints[i])

    BoundaryPointsSet = set(BoundaryPoints)
    for p in del_candidate:
        BoundaryPointsSet.remove(p)
    
    NewBoundaryPoints = []
    for p in BoundaryPointsSet:
        NewBoundaryPoints.append(p)
        
    return NewBoundaryPoints
    