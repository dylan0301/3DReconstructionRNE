from A3_1_nearby import updateNearby
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances


def removeNoise(AllPoints, hyperparameter):
    size = len(AllPoints)
    distMat = defaultdict(dict)
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (AllPoints[i].distance(AllPoints[j]),AllPoints[j]) #2차원 딕셔너리, (dist, point)

    newDistMat = defaultdict(list)
    for i in range(size):
        res = updateNearby(AllPoints[i], distMat, True)
        newDistMat[i] = res

    #노이즈 제거
    del_candidate = set() #point들 집합
    del_candidateIndex = set() #index들 집합
    for i in range(size):
        if AllPoints[i].nearby[hyperparameter.OutlierThreshold][0] > hyperparameter.R:
            del_candidate.add(AllPoints[i])
            del_candidateIndex.add(i)

    for i in del_candidateIndex:
        del AllPoints[i]
            
    for i in AllPoints.keys():
        updateNearby(AllPoints[i], newDistMat, False, hyperparameter, del_candidate)
        

    return AllPoints


def removeNoise2(AllPoints, hyperparameter):
    #nearby에는 (dist, point) 가 들어있음
    size = len(AllPoints)
    pointxyz = [[p.x, p.y, p.z] for p in AllPoints.values()]
    distMat = euclidean_distances(pointxyz, pointxyz)
    del_candidate = set() #point들 집합
    del_candidateIndex = set() #index들 집합
    for i in range(size):
        for j in range(size):
            if distMat[i][j] <= hyperparameter.R:
                AllPoints[i].nearby.append((distMat[i][j], AllPoints[j])) #자기자신도 포함
        if len(AllPoints[i].nenarby) < hyperparameter.OutlierThreshold:
            del_candidate.add(AllPoints[i])
            del_candidateIndex.add(i)
    
    for i in del_candidateIndex:
        del AllPoints[i]

    for i in range(size):
        newNearby = []
        for tup in AllPoints[i].nearby:
            if tup[1] not in del_candidate:
                newNearby.append(tup)
        AllPoints[i].nearby = newNearby[:]
    
    return AllPoints

