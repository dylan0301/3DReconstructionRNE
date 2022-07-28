import numpy as np
import random

#Input: point, Output: nearby2를 포함하는 직선, Method: RANSAC
def nearbyRansacLine(point, hyperparameter):
    def findLine(p1, p2):
        direction = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
        return (direction, p2)
    
    #p에서 line까지 거리
    def pointLineDistance(p, direction, p2):
        PA = np.array([p.x-p2.x, p.y-p2.y, p.z-p2.z])
        res = np.linalg.norm(np.cross(PA, direction))/np.linalg.norm(direction)
        return res

    pts = point.nearby2

    numOfpts = len(pts)
    maxScore = 0
    bestLine = None
    for trial in range(50):
        line = None
        while line == None:
            i1 = random.randrange(0,numOfpts)
            i2 = random.randrange(0,numOfpts)
            while i1 == i2:
                i2 = random.randrange(0,numOfpts)
            line = findLine(pts[i1], pts[i2])
        score = 0
        for p in pts:
            d = pointLineDistance(p, line[0], line[1])
            if d < hyperparameter.H2:
                score +=1
        if score > maxScore:
            maxScore = score
            bestLine = line
    return bestLine[0], maxScore #방향벡터, 점수 



def directionVectorizeRatio(point, VertexPoints, EdgePoints, hyperparameter, VertexRatio, EdgeRatio):
    if len(point.nearby2) < hyperparameter.OutlierThreshold2:
        VertexPoints.append(point)
        VertexRatio.append(None)
        return VertexPoints, EdgePoints, VertexRatio, EdgeRatio
    directionVector, maxScore = nearbyRansacLine(point, hyperparameter)
    directionVector /= np.linalg.norm(directionVector)

    lineRatio = maxScore / len(point.nearby2)

    
    if lineRatio > hyperparameter.ratioThreshold2:
        point.direction = directionVector
        EdgePoints.append(point)
        EdgeRatio.append(lineRatio)
    else: 
        VertexPoints.append(point)
        VertexRatio.append(lineRatio)
    return VertexPoints, EdgePoints, VertexRatio, EdgeRatio

    



def findDirection(BoundaryPoints, EdgePoints, VertexPoints, hyperparameter):
    EdgeRatio = []
    VertexRatio = []
    for p in BoundaryPoints:
        VertexPoints, EdgePoints, VertexRatio, EdgeRatio = directionVectorizeRatio(p, VertexPoints, EdgePoints, hyperparameter, VertexRatio, EdgeRatio)
    print('VertexRatio')
    print(VertexRatio[:100])
    print('EdgeRatio')
    print(EdgeRatio[:100])
    return VertexPoints, EdgePoints
