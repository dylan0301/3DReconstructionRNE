import numpy as np
import random
from sklearn.linear_model import LinearRegression

#Input: point, Output: nearby를 포함하는 평면, Method: Linear Regression
def nearbyLinearRegressionPlane(point, hyperparameter):
    pts = point.nearby
    XY = np.array([[p.x, p.y] for p in pts])
    Z = np.array([p.z for p in pts])
    reg = LinearRegression().fit(XY, Z) 
    coef = reg.coef_
    a, b, d= coef[0], coef[1], reg.intercept_
    #z = ax+by+d
    c = -1
    return (a, b, c, d)


#Input: point, Output: nearby를 포함하는 평면, Method: RANSAC
def nearbyRansacPlane(point, hyperparameter):
    random.seed(0)
    
    #점 3개지나는 평면의 방정식 abcd 튜플로 리턴
    def findPlane(p1, p2, p3):
        v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
        v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
        normal = np.cross(v12,v13)
        if np.linalg.norm(normal) < hyperparameter.normalLeastNorm:
            return None
        d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
        return (normal[0], normal[1], normal[2], d)
    

    #nearby에 자기자신도 있음
    pts = point.nearby

    numOfpts = len(pts)
    maxScore = 0
    bestPlane = None
    for trial in range(hyperparameter.vectorRansacTrial):
        i1 = random.randrange(0,numOfpts)
        i2 = random.randrange(0,numOfpts)
        while i1 == i2:
            i2 = random.randrange(0,numOfpts)
        i3 = random.randrange(0,numOfpts)
        while i1 == i3 or i2 == i3:
            i3 = random.randrange(0,numOfpts)
        plane = findPlane(pts[i1], pts[i2], pts[i3])
        if plane == None:
            continue
        score = 0
        for p in pts:
            d = sujikDistance(p, plane)
            if d < hyperparameter.vectorRansacThreshold: #r/3이 여기서 랜색 threshold임<<이렇게하니까 문제생김
                score +=1
        if score > maxScore:
            maxScore = score #이거 넣는걸 깜빡해서 문제가 좀 생겼음. 근데 정육면체데이터는 괜찮았네? 아무거나 평면 잡아도 다 거기서거기라 그런듯
            bestPlane = plane
    return bestPlane

#점 p랑 ax+by+cz+d=0 수직거리
def sujikDistance(p, plane):
    a, b, c, d = plane[0], plane[1], plane[2], plane[3]
    x, y, z = p.x, p.y, p.z
    res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
    return res

