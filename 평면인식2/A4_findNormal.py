from A4_2_onePointFindNormal import *

def findNormalSTD(AllPoints, BoundaryPoints, CenterPoints, hyperparameter):
    friend = hyperparameter.friend
    for p in AllPoints.values():
        p.nearby = p.nearby[:friend]
        BoundaryPoints, CenterPoints = normalVectorizeSTD(p, hyperparameter, BoundaryPoints, CenterPoints)
    return BoundaryPoints, CenterPoints