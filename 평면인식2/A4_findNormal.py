from A4_2_findNormalSTD import normalVectorizeSTD

def findNormalSTD(AllPoints, hyperparameter):
    for p in AllPoints.values():
        normalVectorizeSTD(p, hyperparameter)