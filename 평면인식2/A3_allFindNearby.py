from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

def allFindNearby(AllPoints, hyperparameter):
    size = len(AllPoints)
    pointxyz = [[p.x, p.y, p.z] for p in AllPoints.values()]
    distMat = euclidean_distances(pointxyz, pointxyz)
    for i in range(size):
        for j in range(size):
            if distMat[i][j] <= hyperparameter.R1:
                AllPoints[i].nearby1.append(AllPoints[j]) #자기자신도 포함
            if distMat[i][j] <= hyperparameter.R2:
                AllPoints[i].nearby2.append(AllPoints[j]) #자기자신도 포함
