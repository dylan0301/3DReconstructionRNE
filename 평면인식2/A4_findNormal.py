from A4_2_onePointFindNormal import *


def findNormalError(AllPoints, BoundaryPoints, CenterPoints, hyperparameter):
    BoundaryError = []
    CenterError = []
    for p in AllPoints.values():
        BoundaryPoints, CenterPoints, BoundaryError, CenterError = normalVectorizeError(p, BoundaryPoints, CenterPoints, hyperparameter, BoundaryError, CenterError)
    # print('BoundaryError')
    # print(BoundaryError[:100])
    # print('CenterError')
    # print(CenterError[:100])
    return BoundaryPoints, CenterPoints
