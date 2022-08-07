from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from A1_classes import *

#polygon 방식
#받는 데이터: objList
#obj들끼리 idx 다 달라서 그걸 label로 하면 될듯
#objList[i] = object class
#object class안에 planes 라는 set 이 있음
#그 set 안에 plane마다 polygon 이라는 list가 있음
#그 list에는 vertex class들이 있음
#각 vertex 안에는 mainpoint 라는 np.array([x,y,z]) 넘파이 배열이 있음.
# def objVisualization_outdated(objList):
#     fig = plt.figure()
#     ax = Axes3D(fig, auto_add_to_figure=False)
#     fig.add_axes(ax)
#     for i in range(len(objList)):
#         obj = objList[i]
#         for plane in obj.planes:
#             pl = [list(v.mainpoint) for v in plane.polygon]
#             polys = Poly3DCollection(pl)
#             polys.set_edgecolor('k')
#             polys.set_facecolor(colors.rgb2hex(np.random.rand(3)))
#             ax.add_collection3d(polys)
#     plt.show()
    

def objVisualization(objList, name):
    finalAllPoints = []
    finalAllLabels = []
    additionalPoints = [Point(0,0,0,None) for i in range(len(objList))]
    additionalLabels = [i for i in range(len(objList))]
    for obj in objList:
        thisObjPoints = []
        thisObjLabels = []
        thisObjPoints.extend(obj.objBoundaryPoints)
        thisObjLabels += [obj.idx]*len(obj.objBoundaryPoints)
        for plane in obj.planes:
            thisObjPoints.extend(plane.interiorPoints)
            thisObjLabels += [obj.idx]*len(plane.interiorPoints)
        finalAllPoints.extend(thisObjPoints)
        finalAllLabels.extend(thisObjLabels)
        
        thisObjPoints.extend(additionalPoints)
        thisObjLabels.extend(additionalLabels)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ap = np.array([[p.x, p.y, p.z] for p in thisObjPoints])
        ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=thisObjLabels, marker='o', s=15, cmap='rainbow')
        plt.title(name+": Object "+str(obj.idx))
        plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ap = np.array([[p.x, p.y, p.z] for p in finalAllPoints])
    ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=finalAllLabels, marker='o', s=15, cmap='rainbow')
    plt.title(name+": All Objects")
    plt.show()
