#A3_allFindNearby.py

from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy.spatial as spatial

def allFindNearby(AllPoints, hyperparameter):
    size = len(AllPoints)
    # pointxyz = [[p.x, p.y, p.z] for p in AllPoints.values()]
    # distMat = euclidean_distances(pointxyz, pointxyz)
    # for i in range(size):
    #     for j in range(size):
    #         if distMat[i][j] <= hyperparameter.R1:
    #             AllPoints[i].nearby1.append(AllPoints[j]) #자기자신도 포함
    #     numOfpts = len(AllPoints[i].nearby1)
    #     if numOfpts < 4:
    #         print('x: ', AllPoints[i].x, ' y: ', AllPoints[i].y, ' z: ', AllPoints[i].z)
    #         raise Exception('len(pts) < 4')
    
    coorDict = dict()
    for p in AllPoints.values():
        coorDict[(p.x, p.y, p.z)] = p

    avgnearby = 0

    points = np.array([[p.x, p.y, p.z] for p in AllPoints.values()])
    point_tree = spatial.cKDTree(points)
    for i in range(size):
        nearby = point_tree.data[point_tree.query_ball_point([AllPoints[i].x,AllPoints[i].y, AllPoints[i].z],hyperparameter.R1)]
        if len(nearby) < 4:
            print('x: ', AllPoints[i].x, ' y: ', AllPoints[i].y, ' z: ', AllPoints[i].z)
            raise Exception('len(pts) < 4')
        avgnearby += len(nearby)
        for q in nearby:
            AllPoints[i].nearby1.append(coorDict[(q[0], q[1], q[2])])
    
    avgnearby /= size
    print(avgnearby)