#A9_processAllObj.py

from collections import defaultdict
from A1_classes import *
import random
from sklearn.decomposition import PCA

#one obj에서 planes edges
def proccessOneObj(obj, availableEdgeLabel, EdgePoints, hyperparameter):
    planeSetEdgeMap = defaultdict(set)
    #key: label기준으로 정렬된 tuple (plane1, plane2)
    #value: edgepoints set
    
    for p in obj.objBoundaryPoints:
        planeNearP = set()
        
        for q in p.nearby1:
            if q.planeClass:
                planeNearP.add(q.planeClass)
        
        if len(planeNearP) == 2:
            planeNearP = list(planeNearP)
            planeNearP.sort(key=lambda x: x.label)
            planeNearP = tuple(planeNearP)
            planeSetEdgeMap[planeNearP].add(p)
            EdgePoints.append(p)

    for planePair in planeSetEdgeMap.keys():
        plane1 = planePair[0]
        plane2 = planePair[1]
        newEdge = Edge(availableEdgeLabel, list(planeSetEdgeMap[planePair]))
        availableEdgeLabel += 1
        midpoint = np.array([float(0), float(0), float(0)])
        for p in newEdge.linePoints:
            p.edgeClass = newEdge
            midpoint += np.array([p.x, p.y, p.z])
        midpoint /= len(newEdge.linePoints)
        newEdge.midpoint = midpoint
        #newEdge.directionVec, newEdge.pointOnLine = nearbyRansacLine(newEdge.linePoints, hyperparameter)
        newEdge.directionVec, newEdge.pointOnLine = PCAline(newEdge.linePoints) #pointonline = mean
        plane1.planeEdgeDict[plane2] = newEdge
        plane2.planeEdgeDict[plane1] = newEdge
        plane1.containedObj.add(obj)
        plane2.containedObj.add(obj)
        obj.planes.add(plane1)
        obj.planes.add(plane2)
        obj.edges.add(newEdge)
        
    return availableEdgeLabel, EdgePoints

#Input: pts, Output: pts를 포함하는 line, Method: PCA
def PCAline(pts):
    X = np.array([[p.x,p.y,p.z] for p in pts])
    pca = PCA(n_components=1)
    pca.fit(X)
    return pca.components_[0], pca.mean_

#Input: pts, Output: pts를 포함하는 line, Method: RANSAC
def nearbyRansacLine(pts, hyperparameter):
    random.seed(0)
    
    def findLine(p1, p2):
        direction = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
        if np.linalg.norm(direction) < 0.000001:
            return None
        return (direction, p2)
    
    #p에서 line까지 거리
    def pointLineDistance(p, direction, p2):
        PA = np.array([p.x-p2.x, p.y-p2.y, p.z-p2.z])
        res = np.linalg.norm(np.cross(PA, direction))/np.linalg.norm(direction)
        return res


    numOfpts = len(pts)
    if numOfpts < 2:
        raise Exception('len(pts) < 2')
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
            if d < hyperparameter.edgeRansacH: 
                score +=1
        if score > maxScore:
            maxScore = score
            bestLine = line
    bestLine = line
    return bestLine[0], bestLine[1] #방향벡터, 직선위 점 하나 


def processAllObj(objList, hyperparameter):
    availableEdgeLabel = 0
    EdgePoints = []
    for obj in objList:
        availableEdgeLabel, EdgePoints = proccessOneObj(obj, availableEdgeLabel, EdgePoints, hyperparameter)
    return EdgePoints