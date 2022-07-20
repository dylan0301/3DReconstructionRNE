from A3_1_nearby import updateNearby
from collections import defaultdict

def removeNoise(AllPoints, hyperparameter):
    size = len(AllPoints)
    distMat = defaultdict(dict)
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (AllPoints[i].distance(AllPoints[j]),AllPoints[j]) 

    newDistMat = defaultdict(list)
    for i in range(size):
        res = updateNearby(AllPoints[i], distMat)
        newDistMat[i] = res

    #노이즈 제거
    del_candidate = set() #index들 리스트
    for i in range(size):
        if AllPoints[i].nearby[hyperparameter.OutlierThreshold][0] > hyperparameter.noiseR:
            del_candidate.add(AllPoints[i])
            
    for j in list(del_candidate):
        del AllPoints[j]
            
    for i in AllPoints.keys():
        updateNearby(AllPoints[i], newDistMat, del_candidate)

    return AllPoints