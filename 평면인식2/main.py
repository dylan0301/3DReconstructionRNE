from A1_classes import *
from A2_data import *
from A3_removeNoise import *
from A4_findNormal import *
from A5_vectorClustering import *
from A6_distanceStairClustering import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

#2 data
print()
print('bring data start')
t = time.time()
filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
#filename = '50000points_2plane.ply'
filename = 'box_5K.ply'

AllPoints, hyperparameter = importPly(filepath+filename)
#AllPoints, hyperparameter = cubeClean()
print('bring data time: ', time.time()-t)
print()
#실제 데이터했으면 여기서 수동으로 hyperparameter 약간 수정 필요


#3 removeNoise
print('removeNoise start')
print(len(AllPoints), 'points before removeNoise')
t = time.time()
AllPoints = removeNoise(AllPoints, hyperparameter)
print(len(AllPoints), 'points after removeNoise')
print('removeNoise time: ', time.time()-t)
print()



#4 findNormal
print('findNormal start')
t = time.time()
BoundaryPoints = []
CenterPoints = []
BoundaryPoints, CenterPoints = findNormalSTD(AllPoints, BoundaryPoints, CenterPoints, hyperparameter)
print(len(BoundaryPoints), 'BoundaryPoints')
print(len(CenterPoints), 'CenterPoints')
print('findNormal time: ', time.time()-t)
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
clusterPointMap, newLabel = vectorClustering(CenterPoints, hyperparameter)
print('vectorClustering time: ', time.time()-t)
print()


# print("클러스터 분류 결과:", newLabel)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in CenterPoints])
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=newLabel, marker='o', s=15, cmap='rainbow')
plt.show()


#6 stairclustering
NewClusterPointMap = defaultdict(list)
NewLabels = []
def divideAllCluster(clusterPointMap):
    for Cluster in clusterPointMap.values():
        divideCluster_stairmethod(Cluster, NewClusterPointMap, NewLabels, hyperparameter)
    
