from collections import defaultdict
from A1_classes import *
import numpy as np

#planeSet는 업데이트 안됨
def disconnectObj(planeSet, hyperparameter):
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
                planeB.interiorPoints = holeFill_1(planeB, edgeSet, hyperparameter)
                obj.planes.remove(planeA)
                obj.planes.add(planeB)

#art gallery problem 방식
#plane.interiorPoints들을 채워줌

def holeFill_1(plane, edgeSet, hyperparameter):
    
    def projection(plane, edge):
        normal = np.array([plane.equation[0], plane.equation[1], plane.equation[2]])
        norm_projected_linevec = normal*(np.dot(edge.directionVec, normal)/(np.linalg.norm(normal)**2)) 
        midpoint_vec = edge.midpoint
        norm_projected_pointvec = normal*(np.dot(midpoint_vec, normal)/(np.linalg.norm(normal)**2)) 
        line_projected = edge.directionVec - norm_projected_linevec
        Newline = Line(line_projected, norm_projected_pointvec)
        return Newline

    def isPositive(line, point):
        return np.dot(line.directionVec, point-line.midpoint) > 0

    lineList = []
    for edge in edgeSet:
        lineList.append(projection(plane, edge))
    
    for line in lineList:
        c = 0
        for otherLine in lineList:
            if line != otherLine:
                pt = otherLine.midpoint
                c += 1 if isPositive(line, pt) else -1
        line.condition = (c > 0)
    
    l = []
    
    if plane.equation[2] != 0:           
        maxcoor = [(-1)*float('inf'), (-1)*float('inf')]
        mincoor = [float('inf'), float('inf')]
        
        for edge in edgeSet:
            for p in edge.linePoints:
                if maxcoor[0] < p.x:
                    maxcoor[0] = p.x
                if maxcoor[1] < p.y:
                    maxcoor[1] = p.y
                if mincoor[0] > p.x:
                    mincoor[0] = p.x
                if mincoor[1] > p.y:
                    mincoor[1] = p.y
                    
        Iteration = [np.arange(mincoor[0], maxcoor[0], hyperparameter.lineardensity), np.arange(mincoor[1], maxcoor[1], hyperparameter.lineardensity)]
        
        for x in Iteration[0]:
            for y in Iteration[1]:
                z = (-1)*(plane.equation[0]*x+plane.equation[1]*y+plane.equation[3])/plane.equation[2]
                coor = np.array([x, y, z])
                for line in lineList:
                    if line.condition:
                        if np.dot(line.directionVec, coor-line.midpoint) < 0:
                            break
                    else:
                        if np.dot(line.directionVec, coor-line.midpoint) > 0:
                            break
                coor = Point(coor[0], coor[1], coor[2], None)
                l.append(coor)
                            
    elif plane.equation[1] != 0:           
        maxcoor = [(-1)*float('inf'), (-1)*float('inf')]
        mincoor = [float('inf'), float('inf')]
        
        for edge in edgeSet:
            for p in edge.linePoints:
                if maxcoor[0] < p.x:
                    maxcoor[0] = p.x
                if maxcoor[1] < p.z:
                    maxcoor[1] = p.z
                if mincoor[0] > p.x:
                    mincoor[0] = p.x
                if mincoor[1] > p.z:
                    mincoor[1] = p.z

        Iteration = [np.arange(mincoor[0], maxcoor[0], hyperparameter.lineardensity), np.arange(mincoor[1], maxcoor[1], hyperparameter.lineardensity)]
                    
        for x in Iteration[0]:
            for z in Iteration[1]:
                y = (-1)*(plane.equation[0]*x+plane.equation[2]*z+plane.equation[3])/plane.equation[1]
                coor = np.array([x, y, z])
                for line in lineList:
                    if line.condition:
                        if np.dot(line.directionVec, coor-line.midpoint) < 0:
                            break
                    else:
                        if np.dot(line.directionVec, coor-line.midpoint) > 0:
                            break
                coor = Point(coor[0], coor[1], coor[2], None)
                l.append(coor)
   
    elif plane.equation[0] != 0:           
        maxcoor = [(-1)*float('inf'), (-1)*float('inf')]
        mincoor = [float('inf'), float('inf')]
        
        for edge in edgeSet:
            for p in edge.linePoints:
                if maxcoor[0] < p.y:
                    maxcoor[0] = p.y
                if maxcoor[1] < p.z:
                    maxcoor[1] = p.z
                if mincoor[0] > p.y:
                    mincoor[0] = p.y
                if mincoor[1] > p.z:
                    mincoor[1] = p.z
                    
        Iteration = [np.arange(mincoor[0], maxcoor[0], hyperparameter.lineardensity), np.arange(mincoor[1], maxcoor[1], hyperparameter.lineardensity)]
                    
        for y in Iteration[0]:
            for z in Iteration[1]:
                x = (-1)*(plane.equation[1]*y+plane.equation[2]*z+plane.equation[3])/plane.equation[0]
                coor = np.array([x, y, z])
                for line in lineList:
                    if line.condition:
                        if np.dot(line.directionVec, coor-line.midpoint) > 0:
                            break
                    else:
                        if np.dot(line.directionVec, coor-line.midpoint) < 0:
                            break
                coor = Point(coor[0], coor[1], coor[2], None)
                l.append(coor)
   
    else: 
        raise(InterruptedError)
    
    return l    