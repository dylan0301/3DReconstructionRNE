#boundarypoint들을 위치를 기준으로 DBSCAN함.
#같은 클러스터로 분류된 boundarypoint들을 같은 물체로 분류
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from A1_classes import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def boundaryClustering(BoundaryPoints, hyperparameter):
    boundary_points_np = np.array([[p.x, p.y, p.z] for p in BoundaryPoints])
    clustering = DBSCAN(eps = hyperparameter.eps_finalBoundaryPoint, min_samples = hyperparameter.min_samples_finalBoundaryPoint)
    labels = clustering.fit_predict(boundary_points_np)
    BoundaryCluster = defaultdict(list)
    objList = []
    
    for i in range(len(BoundaryPoints)):
        BoundaryCluster[labels[i]].append(BoundaryPoints[i])
    
    for i, points in BoundaryCluster.items():
        obj = Object(i, points)
        objList.append(obj)
        
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ap = np.array([[p.x, p.y, p.z] for p in BoundaryPoints])
    ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=labels, marker='o', s=15, cmap='rainbow')
    plt.show()

    return objList

globalEdgePoints = []
globalVertexPoints = []


#one obj에서 planes edges vertex 다 만들어줌
def proccessOneObj(obj, availableEdgeLabel, availableVertexLabel):
    planeListEdgeMap = defaultdict(set)
    #key: label기준으로 정렬된 tuple (plane1, plane2)
    #value: edgepoints set
    localVerticesPoints = set()
    edgeListVerticesMap = defaultdict(set)
    #key: label기준으로 정렬된 tuple (edge1, edge2, edge3, ...)
    #value: connected vertices points set

    for p in obj.BoundaryPoints:
        planeNearP = set()
        
        for q in p.nearby1:
            if q.planeClass:
                planeNearP.add(q.planeClass)
        
        if len(planeNearP) == 2:
            planeNearP = list(planeNearP)
            planeNearP.sort(key=lambda x: x.label)
            planeNearP = tuple(planeNearP)
            planeListEdgeMap[planeNearP].add(p)
            globalEdgePoints.append(p)
        if len(planeNearP) > 2:
            localVerticesPoints.add(p)
            globalVertexPoints.append(p)

    for planePair in planeListEdgeMap.keys():
        planePairList = list(planePair)
        plane1 = planePairList[0]
        plane2 = planePairList[1]
        newEdge = Edge(availableEdgeLabel, planeListEdgeMap[planePair])
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
        edgeNearP = list(edgeNearP)
        edgeNearP.sort(key=lambda x: x.label)
        edgeNearP = tuple(edgeNearP)
        edgeListVerticesMap[edgeNearP].add(p)
    
    for edgeList in edgeListVerticesMap.keys():
        newVertex = Vertex(availableVertexLabel, edgeListVerticesMap[edgeList])
        availableVertexLabel += 1
        for connectedEdge in edgeList:
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

            

def processGraph(planeSet):
    #edge class set, vertex class set이 주어졌을때 Polygon vertex 리스트 뱉는다.
    def polygonize(polygonEdges, polygonVertices):
        graph = defaultdict(set) #key: vertex, value: 연결된 vertices
        for edge in polygonEdges:
            twosides = list(edge.vertex)
            graph[twosides[0]].add(twosides[1])
            graph[twosides[1]].add(twosides[0])
       
        nowVertex = twosides[0]
        for vertex in graph.keys():
            if len(graph[vertex]) == 1:
                nowVertex = vertex
        visited = [nowVertex]
        while len(visited) < len(polygonVertices):
            nextVertices = list(graph[nowVertex])
            if visited[-1] != nextVertices[0]:
                visited.append(nextVertices[0])
            else:
                visited.append(nextVertices[1])
        return visited

    #planeSet는 업데이트 안됨
    for plane in planeSet:
        for obj in plane.containedObj:
            newPlane = Plane(None, None)
            newPlane.containedObj = obj #원래는 obj들의 set인데 이제부터 그냥 obj 하나
            newPlane.equation = plane.equation
            polygonEdges = set()
            polygonVertices = set()
            for plane2 in obj.planes: # !!!!!!! 여기서 문제 됨 !!!!!!!!!!!!!! 이거 한 edge당 꼭짓점이 하나두개가 아닌것같은데 어떡하지
                if plane2 in plane.planeEdgeDict.keys():
                    newPlane.planeEdgeDict[plane2] = plane.planeEdgeDict[plane2]
                    plane2.planeEdgeDict[newPlane] = plane2.planeEdgeDict[plane]
                    polygonEdges.add(plane.planeEdgeDict[plane2])
                    polygonVertices = polygonVertices.union(plane.planeEdgeDict[plane2].vertex)
                    del plane2.planeEdgeDict[plane]
            newPlane.polygon = polygonize(polygonEdges, polygonVertices)
            obj.planes.remove(plane)
            obj.planes.add(newPlane)



 
def ObjectSegmentation(BoundaryPoints, planeSet, hyperparameter):
    objList = boundaryClustering(BoundaryPoints, hyperparameter)
    availableEdgeLabel = 0
    availableVertexLabel = 0
    for i in range(len(objList)):
        availableEdgeLabel, availableVertexLabel = proccessOneObj(objList[i], availableEdgeLabel, availableVertexLabel)
    
    #processGraph(planeSet)
    return objList, availableEdgeLabel, availableVertexLabel