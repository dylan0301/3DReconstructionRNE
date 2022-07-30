from collections import defaultdict
from A1_classes import *
import numpy as np

#planeSet는 업데이트 안됨
def disconnectObj(planeSet):
    for planeA in planeSet:
        if len(planeA.containedObj) > 1:
            for obj in planeA.containedObj:
                planeB = Plane(planeA.label, None)
                planeB.containedObj = {obj}
                planeB.equation = planeA.equation
                edgeSet = set()
                for connectedPlane in obj.planes:
                    if connectedPlane in planeA.planeEdgeDict.keys():
                        planeB.planeEdgeDict[connectedPlane] = planeA.planeEdgeDict[connectedPlane]
                        connectedPlane.planeEdgeDict[planeB] = connectedPlane.planeEdgeDict[planeA]
                        edgeSet.add(planeA.planeEdgeDict[connectedPlane])
                        del connectedPlane.planeEdgeDict[planeA]
                holeFill_1(planeB, edgeSet)
                obj.planes.remove(planeA)
                obj.planes.add(planeB)

#art gallery problem 방식
#planeB.interiorPoints들을 채워줌

def holeFill_1(plane, edgeSet):
    def projection(plane, edge):
        normal = np.array([plane.equation.a, plane.equation.b, plane.equation.c])
        norm_projected_linevec = normal*(np.dot(edge.directionVec, normal)/(np.linalg.norm(normal)**2)) 
        pointonline = edge.pointOnLine
        norm_projected_pointvec = normal*(np.dot(pointonline, normal)/(np.linalg.norm(normal)**2)) 
        line_projected = edge.directionVec - norm_projected_linevec
        Newline = Line(line_projected, norm_projected_pointvec)
        return Newline
    
    conditions = []
    
    pass