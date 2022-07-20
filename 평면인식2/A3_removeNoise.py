from A3_1_nearby import updateNearby
from collections import defaultdict

def removeNoise(AllPoints, hyperparameter):
    size = len(AllPoints)
    distMat = defaultdict(dict)
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (AllPoints[i].distance(AllPoints[j]),AllPoints[j]) #2차원 딕셔너리, (dist, point)

    newDistMat = defaultdict(list)
    for i in range(size):
        res = updateNearby(AllPoints[i], distMat)
        newDistMat[i] = res

    #노이즈 제거
    del_candidate = set() #point들 집합
    del_candidateIndex = set() #index들 집합
    for i in range(size):
        if AllPoints[i].nearby[hyperparameter.OutlierThreshold][0] > hyperparameter.noiseR:
            del_candidate.add(AllPoints[i])
            del_candidateIndex.add(i)

    for j in del_candidateIndex:
        del AllPoints[j]
            
    for i in AllPoints.keys():
        updateNearby(AllPoints[i], newDistMat, del_candidate)

    return AllPoints