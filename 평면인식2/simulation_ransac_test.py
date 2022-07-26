#하나의 nearby에서 그냥 생으로 랜색을 돌려서 뭐가 나오는지 확인해보기
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from A2_data import butterfly


AllPoints, hyperparameter = butterfly(R = 7, alpha = np.pi/2, size = 2000)

hyperparameter.vectorRansacTrial=1000
hyperparameter.vectorRansacThreshold=1
hyperparameter.normalLeastNorm = 0



# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ap = np.array([[p.x, p.y, p.z] for p in AllPoints.values()])
# ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[0 for i in range(len(AllPoints))], marker='o', s=15, cmap='rainbow')
# plt.show()



def P_RansacPlane(AllPoints, hyperparameter):
    random.seed(0)
    
    #점 p랑 ax+by+cz+d=0 수직거리
    def sujikDistance(p, plane):
        a, b, c, d = plane[0], plane[1], plane[2], plane[3]
        x, y, z = p.x, p.y, p.z
        res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
        return res

    #점 3개지나는 평면의 방정식 abcd 튜플로 리턴
    def findPlane(p1, p2, p3):
        v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
        v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
        normal = np.cross(v12,v13)
        if np.linalg.norm(normal) < hyperparameter.normalLeastNorm:
            return None
        d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
        return (normal[0], normal[1], normal[2], d)
    
    pts = list(AllPoints.values())

    numOfpts = len(pts)
    maxScore = 0
    bestPlane = None
    bestSatisfied = set()
    for trial in range(hyperparameter.vectorRansacTrial):
        plane = None
        while plane == None:
            i1 = random.randrange(0,numOfpts)
            i2 = random.randrange(0,numOfpts)
            while i1 == i2:
                i2 = random.randrange(0,numOfpts)
            i3 = random.randrange(0,numOfpts)
            while i1 == i3 or i2 == i3:
                i3 = random.randrange(0,numOfpts)
            plane = findPlane(pts[i1], pts[i2], pts[i3])
        score = 0

        satisfied = set()
        for p in pts:
            d = sujikDistance(p, plane)
            if d < hyperparameter.vectorRansacThreshold:
                score +=1
                satisfied.add(p)
        if score > maxScore:
            maxScore = score #이거 넣는걸 깜빡해서 문제가 좀 생겼음. 근데 정육면체데이터는 괜찮았네? 아무거나 평면 잡아도 다 거기서거기라 그런듯
            bestPlane = plane
            bestSatisfied = satisfied
    return bestPlane, maxScore, bestSatisfied

bestPlane, maxScore, bestSatisfied = P_RansacPlane(AllPoints, hyperparameter)



def isSatisfied(p):
    if p in bestSatisfied:
        return 1
    return 0

print(maxScore, maxScore/len(AllPoints))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ap = np.array([[p.x, p.y, p.z] for p in AllPoints.values()])
ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[isSatisfied(p) for p in AllPoints.values()], marker='o', s=15, cmap='rainbow')
plt.show()
