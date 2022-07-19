import _3_nearby

from collections import defaultdict
def removeNoise(AllPoints):
    global friend, r, OutlierThreshold
    size = len(AllPoints)
    distMat = defaultdict(dict)
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (AllPoints[i].distance(AllPoints[j]),j) 

    for i in range(size):
        updateNearby(AllPoints[i], distMat)

    #노이즈 제거
    del_candidate = [] #index들 리스트
    for i in range(size):
        if AllPoints[i].nearby[OutlierThreshold-1][0] > r:
            del_candidate.append(i)
            del distMat[i]
            
    for j in del_candidate:
        del AllPoints[j]
        
    for i in AllPoints.keys():
        for j in del_candidate:
            del distMat[i][j]
            
    for i in AllPoints.keys():
        updateNearby(AllPoints[i], distMat)
