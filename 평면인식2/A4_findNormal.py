from email.errors import BoundaryError
from A4_2_onePointFindNormal import *

def findNormalSTD(AllPoints, BoundaryPoints, CenterPoints, hyperparameter):
    for p in AllPoints.values():
        BoundaryPoints, CenterPoints = normalVectorizeSTD(p, BoundaryPoints, CenterPoints, hyperparameter)
    return BoundaryPoints, CenterPoints

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
