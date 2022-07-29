#boundarypoint들을 위치를 기준으로 DBSCAN함.
#같은 클러스터로 분류된 boundarypoint들을 같은 물체로 분류
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from A1_classes import *

def boundaryClustering(BoundaryPoints, hyperparameter):
    boundary_points_np = np.array([[p.x, p.y, p.z] for p in BoundaryPoints])
    clustering = DBSCAN(eps = hyperparameter.eps_finalBoundaryPoint, min_samples = hyperparameter.min_samples_finalBoundaryPoint)
    labels = clustering.fit_predict(boundary_points_np)
    BoundaryCluster = defaultdict(list())
    objList = []
    
    for i in range(len(BoundaryPoints)):
        BoundaryCluster[labels[i]].append(BoundaryPoints[i])
    
    for i, points in BoundaryCluster.item():
        obj = Object(i, points)
        objList.append(obj)
            
    return objList

#one obj에서 planes edges vertex 다 만들어줌
def proccessOneObj(obj, availableEdgeLabel, availableVertexLabel):
    planeSetEdgeMap = defaultdict(set)
    #key: set(plane1, plane2)
    #value: edgepoints set
    localVerticesPoints = set()
    edgeSetVerticesMap = defaultdict(set)
    #key: set of edges
    #value: connected vertices points set

    for p in obj.BoundaryPoints:
        planeNearP = set()
        for q in p.nearby1:
            if q.planeClass:
                planeNearP.add(q.planeClass)
        
        if len(planeNearP) == 2:
            planeSetEdgeMap[planeNearP].add(p)
        if len(planeNearP) > 2:
            localVerticesPoints.add(p)

    for planePair in planeSetEdgeMap.keys():
        planePairList = list(planePair)
        plane1 = planePairList[0]
        plane2 = planePairList[1]
        newEdge = Edge(availableEdgeLabel, planeSetEdgeMap[planePair])
        availableEdgeLabel += 1
        for p in newEdge.linePoints:
            p.edgeClass = newEdge

        plane1.planeEdgeDict[plane2] = newEdge
        plane2.planeEdgeDict[plane1] = newEdge
        plane1.containedObj.add(obj)
        plane2.containedObj.add(obj)
        obj.planes.add(plane1)
        obj.planes.add(plane2)
        obj.edges.add(newEdge)

    for p in localVerticesPoints:
        edgeNearP = set()
        for q in p.nearby1:
            if q.edgeClass:
                edgeNearP.add(q.edgeClass)
        edgeSetVerticesMap[edgeNearP].add(p)
    
    for edgeSet in edgeSetVerticesMap.keys():
        newVertex = Vertex(availableVertexLabel, edgeSetVerticesMap[edgeSet])
        availableVertexLabel += 1
        for connectedEdge in edgeSet:
            connectedEdge.vertex.add(newVertex)
            newVertex.edges.add(connectedEdge)
        mainPoint = np.array([float(0),float(0),float(0)])
        for p in newVertex.dotPoints:
            p.vertexClass = newVertex
            mainPoint += np.array([p.x, p.y, p.z])
        mainPoint /= len(newVertex.dotPoints)
        newVertex.mainPoint = mainPoint
        obj.vertices.add(newVertex)
        
    return availableEdgeLabel, availableVertexLabel

            

def processGraph(planeList):
    for plane in planeList:
        for obj in plane.containedObj:
            newPlane = Plane(None, None)
            newPlane.containedObj = obj #원래는 obj들의 set인데 이제부터 그냥 obj 하나
            connectedEdges = set()
            connectedVertices = set()
            for plane2 in obj:
                if plane2 in plane.planeEdgeDict.keys():
                    newPlane.planeEdgeDict[plane2] = plane.planeEdgeDict[plane2]
                    connectedEdges.add(plane.planeEdgeDict[plane2])




 
def ObjectSegmentation(BoundaryPoints, planeList, hyperparameter):
    objList = boundaryClustering(BoundaryPoints, hyperparameter)
    availableEdgeLabel = 0
    availableVertexLabel = 0
    for i in range(len(objList)):
        availableEdgeLabel, availableVertexLabel = proccessOneObj(objList[i], availableEdgeLabel, availableVertexLabel)
    processGraph(planeList)