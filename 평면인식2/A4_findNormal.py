from A4_2_onePointFindNormal import *


def findNormal(AllPoints, BoundaryPoints, CenterPoints, hyperparameter):
    BoundaryError = []
    CenterError = []
    for p in AllPoints.values():
        BoundaryPoints, CenterPoints, BoundaryError, CenterError = normalVectorizeRatio(p, BoundaryPoints, CenterPoints, hyperparameter, BoundaryError, CenterError)
    print('BoundaryError')
    print(BoundaryError[:100])
    print('CenterError')
    print(CenterError[:100])
    return BoundaryPoints, CenterPoints
