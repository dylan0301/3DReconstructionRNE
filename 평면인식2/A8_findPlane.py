import numpy as np
import random


def RANSACPlane2(planeList, hyperparameter):
    random.seed(0)
    
    #점 3개지나는 평면의 방정식 abcd 튜플로 리턴
    def findPlane(p1, p2, p3):
        v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
        v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
        normal = np.cross(v12,v13)
        d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
        return (normal[0], normal[1], normal[2], d)


    #점 p랑 ax+by+cz+d=0 수직거리
    def sujikDistance(p, plane):
        a, b, c, d = plane[0], plane[1], plane[2], plane[3]
        x, y, z = p.x, p.y, p.z
        res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
        return res


    for classPlane in planeList:
        pts = classPlane.interiorPoints

        numOfpts = len(pts)
        maxScore = 0
        bestPlane = None
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
            for p in pts:
                d = sujikDistance(p, plane)
                if d < hyperparameter.vectorRansacThreshold:
                    score +=1
            if score > maxScore:
                maxScore = score
                bestPlane = plane
        classPlane.equation = bestPlane

