import numpy as np
from A1_classes import *
from A4_findNormal import ODRplane


def makePlaneClass(NewClusterPointMap, hyperparameter):
    planeSet = set()
    for i, points in NewClusterPointMap.items():
        plane = Plane(i, points)
        planeSet.add(plane)
        for p in points:
            p.planeClass = plane
        plane.equation = ODRplane(plane.interiorPoints)
    return planeSet