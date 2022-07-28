from collections import defaultdict
from A1_classes import *
from sklearn.metrics.pairwise import euclidean_distances

from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

def boundaryFindNearby(BoundaryPoints, hyperparameter):
    size = len(BoundaryPoints)
    pointxyz = [[p.x, p.y, p.z] for p in BoundaryPoints]
    distMat = euclidean_distances(pointxyz, pointxyz)
    for i in range(size):
        for j in range(size):
            if distMat[i][j] <= hyperparameter.R2:
                BoundaryPoints[i].nearby2.append(BoundaryPoints[j]) #자기자신도 포함
