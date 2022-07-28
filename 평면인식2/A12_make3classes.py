import numpy as np
from A1_classes import *
from A4_findNormal import nearbyRansacPlane 


def makePlaneClass(NewClusterPointMap, hyperparameter):
    PlaneSet = set()
    for i, points in NewClusterPointMap.items():
        plane = Plane(i, points)
        PlaneSet.add(plane)
        for p in points:
            p.planeClass = plane
        plane.equation, maxScore = nearbyRansacPlane(plane.interiorPoints, hyperparameter)
    return PlaneSet


def makeEdgeClass(NewEdgePointMap):
    EdgeSet = set()
    for i, points in NewEdgePointMap.items():
        edge = Edge(i, points)
        EdgeSet.add(edge)
        midPoint = np.array([float(0),float(0),float(0)])
        for p in points:
            p.edgeClass = edge
            midPoint += np.array([p.x, p.y, p.z])
        midPoint /= len(points)
        edge.midPoint = midPoint
    return EdgeSet

def makeVertexClass(vertexPointMap):
    VertexSet = set()
    for i, points in vertexPointMap.items():
        vertex = Vertex(i, points)
        VertexSet.add(vertex)
        mainPoint = np.array([float(0),float(0),float(0)])
        for p in points:
            p.vertexClass = vertex
            mainPoint += np.array([p.x, p.y, p.z])
        mainPoint /= len(points)
        vertex.mainPoint = mainPoint
    return VertexSet


def make3classes(NewClusterPointMap, NewEdgePointMap, vertexPointMap, hyperparameter):
    PlaneSet = makePlaneClass(NewClusterPointMap, hyperparameter)
    EdgeSet = makeEdgeClass(NewEdgePointMap)
    VertexSet = makeVertexClass(vertexPointMap)
    return PlaneSet, EdgeSet, VertexSet
            
        


