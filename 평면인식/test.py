import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

a = np.array([1,0,0])
b = np.array([1,0,0])
print(np.cross(a,b))



class Point:
    def __init__(self, X, Y, Z, idx):
        self.x = X 
        self.y = Y 
        self.z = Z
        self.idx = idx

def generatePoints4(size):
    points = defaultdict(Point)
    for i in range(size-100):
        if i%6 == 0:
            x = 30*random.random()
            y = 30*random.random()
            z = 0
        if i%6 == 1:
            x = 30*random.random()
            y = 0
            z = 30*random.random()
        if i%6 == 2:
            x = 0
            y = 30*random.random()
            z = 30*random.random()
        if i%6 == 3:
            x = 30*random.random()
            y = 30*random.random()
            z = 30
        if i%6 == 4:
            x = 30*random.random()
            y = 30
            z = 30*random.random()
        if i%6 == 5:
            x = 30
            y = 30*random.random()
            z = 30*random.random()                  
        p = Point(x,y,z,i)
        points[i] = p
    for i in range(100):
        p = Point(30*random.random(), 30*random.random(), 30*random.random(), size-100+i)
        points[size-100+i] = p
    return points

random.seed(8)
NewAllPoints = generatePoints4(3000)
NewLabels = [0] * 3000

ap = np.array([[p.x, p.y, p.z] for p in NewAllPoints.values()])
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ap[:,0], ap[:,1], ap[:,2], c=NewLabels, marker='o', s=15, cmap='rainbow')
plt.show()