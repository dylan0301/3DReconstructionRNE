#A10_disconnectObj.py

from collections import defaultdict
from A1_classes import *
import numpy as np
import matplotlib.pyplot as plt

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

    #proj_of_u_on_n = (np.dot(u, n)/n_norm**2)*n
    def projection(plane, edge):
        normal = np.array([plane.equation[0], plane.equation[1], plane.equation[2]])
        proj_plane_line = edge.directionVec - normal*(np.dot(edge.directionVec, normal)/(np.linalg.norm(normal)**2))
        
        midpoint_vec = edge.midpoint
        midpoint_distance = (np.dot(midpoint_vec, normal)+plane.equation[3])/np.linalg.norm(normal)
        proj_plane_midpoint = midpoint_vec - normal/np.linalg.norm(normal)*midpoint_distance
        Newline = Line(proj_plane_line, proj_plane_midpoint)
        return Newline

    normal = np.array([plane.equation[0], plane.equation[1], plane.equation[2]])
    
    def isPositive(line, point):
        return np.dot(np.cross(line.directionVec, point-line.midpoint), normal) > 0

    lineList = []
    for edge in edgeSet:
        lineList.append(projection(plane, edge))
    
    ############################visual test용 임시코드#######################################
    # label = []
    # labelNum = 0
    # X = np.linspace(-1.3, 0.7, 100)
    # Z = np.linspace(-2.5, -0.5, 100)
    # X, Z = np.meshgrid(X, Z)
    # Y = -(plane.equation[0]*X + plane.equation[2]*Z + plane.equation[3])/plane.equation[1]

    # label += [0]*len(Y)
    # labelNum+=1
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z)
    # plt.show()

    # X = []
    # Y =[]
    # Z = []
    # label = []
    # added = [0.03*i for i in range(1, 50)]+[-0.03*i for i in range(1, 50)]
    # for testEdge in edgeSet:
    #     X.append(testEdge.midpoint[0])
    #     Y.append(testEdge.midpoint[1])
    #     Z.append(testEdge.midpoint[2])
    #     label.append(labelNum)
    #     labelNum += 1
    #     for step in added:
    #         X.append(testEdge.midpoint[0] + step*testEdge.directionVec[0])
    #         Y.append(testEdge.midpoint[1] + step*testEdge.directionVec[1])
    #         Z.append(testEdge.midpoint[2] + step*testEdge.directionVec[2])
    #         label.append(labelNum)
    #     labelNum += 1
    # X = np.array(X)
    # Y = np.array(Y)
    # Z = np.array(Z)
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X, Y, Z, c=label)
    # plt.show()

    # X = []
    # Y =[]
    # Z = []
    # label = []
    # added = [0.03*i for i in range(1, 50)]+[-0.03*i for i in range(1, 50)]
    # for testEdge in lineList:
    #     X.append(testEdge.midpoint[0])
    #     Y.append(testEdge.midpoint[1])
    #     Z.append(testEdge.midpoint[2])
    #     label.append(labelNum)
    #     labelNum += 1
    #     for step in added:
    #         X.append(testEdge.midpoint[0] + step*testEdge.directionVec[0])
    #         Y.append(testEdge.midpoint[1] + step*testEdge.directionVec[1])
    #         Z.append(testEdge.midpoint[2] + step*testEdge.directionVec[2])
    #         label.append(labelNum)
    #     labelNum += 1
    # X = np.array(X)
    # Y = np.array(Y)
    # Z = np.array(Z)
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X, Y, Z, c=label)
    # plt.show()
    #############################################################################



    for line in lineList:
        c = 0
        for otherLine in lineList:
            if line != otherLine:
                pt = otherLine.midpoint
                c += 1 if isPositive(line, pt) else -1
        line.condition = bool(c > 0)
        #print(c)
    l = []

    if abs(plane.equation[1]) > 0.1:           
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
                flag = True
                for line in lineList:
                    if line.condition:
                        flag = flag and isPositive(line, coor)
                    else:
                        flag = flag and not isPositive(line, coor)
                if flag:
                    coor = Point(coor[0], coor[1], coor[2], None)
                    l.append(coor)

    elif abs(plane.equation[2]) > 0.1:           
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
                flag = True
                for line in lineList:
                    if line.condition:
                        flag = flag and isPositive(line, coor)
                    else:
                        flag = flag and not isPositive(line, coor)
                if flag:
                    coor = Point(coor[0], coor[1], coor[2], None)
                    l.append(coor)
    
    elif abs(plane.equation[0]) > 0.1:           
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
                flag = True
                for line in lineList:
                    if line.condition:
                        flag = flag and isPositive(line, coor)
                    else:
                        flag = flag and not isPositive(line, coor)
                if flag:
                    coor = Point(coor[0], coor[1], coor[2], None)
                    l.append(coor)
   
    else: 
        raise(InterruptedError)
    
    return l    