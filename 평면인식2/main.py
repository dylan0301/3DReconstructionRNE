from A1_classes import *
from A2_data import *
from A3_allFindNearby import *
from A4_findNormal import *
from A5_vectorClustering import *
from A6_distanceStairClustering import *
from A7_boundaryFindNearby import *
from A8_findDirection import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from collections import defaultdict


#2 data
print()
print('bring data start')
t = time.time()
filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
filename = 'twoBoxes.ply'

#AllPoints, hyperparameter = importPly(filepath, filename)
AllPoints, hyperparameter = unicorn_sample2()
print('bring data time: ', time.time()-t)
print(len(AllPoints), 'points')
print()
#실제 데이터했으면 여기서 수동으로 hyperparameter 약간 수정 필요

#노이즈제거 전 점들 출력
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in AllPoints.values()])
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[i for i in range(len(AllPoints))], marker='o', s=15, cmap='rainbow')
plt.show()



#3 allFindNearby
print('allFindNearby start')
t = time.time()
allFindNearby(AllPoints, hyperparameter)
print('여기이후로 AllPoints는 안쓰인다. 만약 AllPoints 쓸려면 vectorDBSCAN 바꾸거라.')
print('allFindNearby time:', time.time()-t)
print()



#4 findNormal
print('findNormal start')
t = time.time()
BoundaryPoints = []
CenterPoints = []
BoundaryPoints, CenterPoints = findNormal(AllPoints, BoundaryPoints, CenterPoints, hyperparameter)
print(len(BoundaryPoints), 'BoundaryPoints')
print(len(CenterPoints), 'CenterPoints')
print('findNormal time:', time.time()-t)
print()




#법선벡터들 출력
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([list(p.normal) for p in CenterPoints])
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[0] * len(CenterPoints), marker='o', s=15, cmap='rainbow')
plt.show()



#5 vectorClustering
print('vectorClustering start')
t = time.time()
CenterPoints, clusterPointMap,  newLabel= vectorDBSCAN(CenterPoints, hyperparameter)
print(len(CenterPoints), 'CenterPoints after vectorClustering')
print('vectorClustering time:', time.time()-t)
print()


# print("클러스터 분류 결과:", newLabel)
# vectorClustering 이후 CenterPoints
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in CenterPoints])
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=newLabel, marker='o', s=15, cmap='rainbow')
plt.show()


#6 distanceStairClustering
print('distanceStairClustering start')
t = time.time()
NewClusterPointMap = defaultdict(list)
NewClusterPointMap = divideAllCluster_DBSCAN(NewClusterPointMap, clusterPointMap, hyperparameter)
print('distanceStairClustering time:', time.time()-t)
print()

planeList = []
for i, points in NewClusterPointMap.items():
    plane = Plane(i, points)
    planeList.append(plane)

NewAllPoints = []
for k in NewClusterPointMap.keys():
    NewAllPoints.extend(NewClusterPointMap[k])
# NewAllPoints.extend(BoundaryPoints)

NewLabels = []
for k in NewClusterPointMap.keys():
    NewLabels += [k] * len(NewClusterPointMap[k])
# NewLabels += [max(NewLabels) + 1] * len(BoundaryPoints) 

#distanceStairClustering 이후 CenterPoints
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in NewAllPoints])
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=NewLabels, marker='o', s=15, cmap='rainbow')
plt.show()

#BoundaryPoints
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in BoundaryPoints])
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[0] * len(BoundaryPoints), marker='o', s=15, cmap='rainbow')
plt.show()


#7 boundaryFindNearby
print('boundaryFindNearby start')
print(len(BoundaryPoints), 'BoundaryPoints before boundaryRemoveNoise')
t = time.time()
boundaryFindNearby(BoundaryPoints, hyperparameter)
print('boundaryFindNearby time:', time.time()-t)
print()



#8 findDirection
print('findDirection start')
t = time.time()
EdgePoints = []
VertexPoints = []
VertexPoints, EdgePoints = findDirection(BoundaryPoints, EdgePoints, VertexPoints, hyperparameter)
print(len(VertexPoints), 'VertexPoints')
print(len(EdgePoints), 'EdgePoints')
print('findNormal time:', time.time()-t)
print()


#방향벡터들 출력
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([list(p.direction) for p in EdgePoints])
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[0] * len(EdgePoints), marker='o', s=15, cmap='rainbow')
plt.show()