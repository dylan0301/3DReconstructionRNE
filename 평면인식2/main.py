from A1_classes import *
from A2_data import *
from A3_allFindNearby import *
from A4_findNormal import *
from A5_normalClustering import *
from A6_3DdistanceStairClustering import *
from A7_makePlaneClass import *
from A8_boundaryClustering import *
from A9_processAllObj import *
from A10_disconnectObj import *
from B1_Visualization import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from collections import defaultdict


#2 data
print()
print('#2 bring data start')
t = time.time()
filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
filename = '3boxes.ply'

#AllPoints, hyperparameter, name = importPly(filepath, filename)
AllPoints, hyperparameter, name = FourCleanBoxes()
print('#2 bring data time: ', time.time()-t)
print(len(AllPoints), 'points')
print()
#실제 데이터했으면 여기서 수동으로 hyperparameter 약간 수정 필요

#노이즈제거 전 점들 출력
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in AllPoints.values()])
plt.title(name+": All Points")
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[i for i in range(len(AllPoints))], marker='o', s=15, cmap='rainbow')
plt.show()



#3 allFindNearby
print('#3 allFindNearby start')
t = time.time()
allFindNearby(AllPoints, hyperparameter)
print('#3 allFindNearby time:', time.time()-t)
print()
#여기이후로 AllPoints는 안쓰인다. 만약 AllPoints 쓸려면 vectorDBSCAN 바꾸거라.


#4 findNormal
print('#4 findNormal start')
t = time.time()
BoundaryPoints = []
CenterPoints = []
BoundaryPoints, CenterPoints = findNormal(AllPoints, BoundaryPoints, CenterPoints, hyperparameter)
print(len(BoundaryPoints), 'BoundaryPoints')
print(len(CenterPoints), 'CenterPoints')
print('#4 findNormal time:', time.time()-t)
print()




#법선벡터들 출력
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([list(p.normal) for p in CenterPoints])
plt.title(name+": Normal Vectors of Interior Points")
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[0] * len(CenterPoints), marker='o', s=15, cmap='rainbow')
plt.show()



#5 vectorClustering
print('#5 vectorClustering start')
t = time.time()
CenterPoints, clusterPointMap,  newLabel=normalDBSCAN(CenterPoints, hyperparameter)
print(len(CenterPoints), 'CenterPoints after vectorClustering')
print('#5 vectorClustering time:', time.time()-t)
print()


# print("클러스터 분류 결과:", newLabel)
# vectorClustering 이후 CenterPoints
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in CenterPoints])
plt.title(name+": Normal vector Clustered")
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=newLabel, marker='o', s=15, cmap='rainbow')
plt.show()


#6 3DdistanceStairClustering
print('#6 3DdistanceStairClustering start')
t = time.time()
NewClusterPointMap = defaultdict(list)
NewClusterPointMap = divideAllPlane_DBSCAN(NewClusterPointMap, clusterPointMap, hyperparameter)
print('#6 3DdistanceStairClustering time:', time.time()-t)
print()

NewCenterPoints = [] #centerpoint만 들어있음
for k in NewClusterPointMap.keys():
    NewCenterPoints.extend(NewClusterPointMap[k])

NewCenterLabels = []
for k in NewClusterPointMap.keys():
    NewCenterLabels += [k] * len(NewClusterPointMap[k])

#distanceStairClustering 이후 CenterPoints
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in NewCenterPoints])
plt.title(name+": Final Planes")
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=NewCenterLabels, marker='o', s=15, cmap='rainbow')
plt.show()

#BoundaryPoints
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in BoundaryPoints])
plt.title(name+": Boundary Points")
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[0] * len(BoundaryPoints), marker='o', s=15, cmap='rainbow')
plt.show()


#7 makePlaneClass
print('#7 makePlaneClass start')
t = time.time()
planeSet = makePlaneClass(NewClusterPointMap, hyperparameter)
print(len(planeSet), 'planes')
print('#7 makePlaneClass time:', time.time()-t)
print()


#8 boundaryClustering
print('#8 boundaryClustering start')
t = time.time()
objList, boundaryObjLabels = boundaryClustering(BoundaryPoints, hyperparameter)
print(len(objList), 'objects')
print('#8 boundaryClustering time:', time.time()-t)
print()

#BoundaryPoints after boundaryClustering
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in BoundaryPoints])
plt.title(name+": BoundaryPoints after boundaryClustering")
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=boundaryObjLabels, marker='o', s=15, cmap='rainbow')
plt.show()

#9 processAllObj
print('#9 processAllObj start')
t = time.time()
EdgePoints = processAllObj(objList, hyperparameter)
print(len(EdgePoints), 'EdgePoints')
print('#9 processAllObj time:', time.time()-t)
print()

#EdgePoints
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in EdgePoints])
plt.title(name+": Edge Points")
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[p.edgeClass.label for p in EdgePoints], marker='o', s=15, cmap='rainbow')
plt.show()

#10 disconnectObj
print('#10 disconnectObj start')
t = time.time()
disconnectObj(planeSet, hyperparameter)
print('#10 disconnectObj time:', time.time()-t)
print()

#B1 Visualization
objVisualization(objList, name)