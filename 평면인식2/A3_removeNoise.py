from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

#removenoise 하고 nearby까지 찾아줌
def removeNoise2(AllPoints, hyperparameter):
    #nearby= point들의 리스트 가 들어있음
    size = len(AllPoints)
    pointxyz = [[p.x, p.y, p.z] for p in AllPoints.values()]
    distMat = euclidean_distances(pointxyz, pointxyz)
    del_candidate = set() #point들 집합
    del_candidateIndex = set() #index들 집합
    for i in range(size):
        for j in range(size):
            if distMat[i][j] <= hyperparameter.R:
                AllPoints[i].nearby.append(AllPoints[j]) #자기자신도 포함
        if len(AllPoints[i].nearby) < hyperparameter.OutlierThreshold:
            del_candidate.add(AllPoints[i])
            del_candidateIndex.add(i)
    
    for i in del_candidateIndex:
        del AllPoints[i]

    for i in AllPoints.keys():
        newNearby = []
        for p in AllPoints[i].nearby:
            if p not in del_candidate:
                newNearby.append(p)
        AllPoints[i].nearby = newNearby
    
    return AllPoints

