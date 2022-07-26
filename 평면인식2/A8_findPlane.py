import numpy as np

#점 3개지나는 평면의 방정식 abcd 튜플로 리턴
def findPlane(p1, p2, p3):
    v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
    v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
    normal = np.cross(v12,v13)
    d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
    return (normal[0], normal[1], normal[2], d)