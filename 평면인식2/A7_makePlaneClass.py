import numpy as np
from A1_classes import *
from A4_findNormal import nearbyRansacPlane 


def makePlaneClass(NewClusterPointMap, hyperparameter):
    planeSet = set()
    for i, points in NewClusterPointMap.items():
        plane = Plane(i, points)
        planeSet.add(plane)
        for p in points:
            p.planeClass = plane
        plane.equation, maxScore = nearbyRansacPlane(plane.interiorPoints, hyperparameter)
    return planeSet