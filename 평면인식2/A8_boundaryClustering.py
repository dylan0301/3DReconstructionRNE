#A8_boundaryClustering.py

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
    boundaryObjLabels = clustering.fit_predict(boundary_points_np)
    BoundaryCluster = defaultdict(list)
    objList = []
    
    for i in range(len(BoundaryPoints)):
        if boundaryObjLabels[i] != -1:
            BoundaryCluster[boundaryObjLabels[i]].append(BoundaryPoints[i])
    
    for i, points in BoundaryCluster.items():
        obj = Object(i, points)
        objList.append(obj)

    return objList, boundaryObjLabels