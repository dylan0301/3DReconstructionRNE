import numpy as np
import random
from sklearn.decomposition import PCA


#Input: pts, Output: pts를 포함하는 평면, Method: RANSAC,
def nearbyRansacPlane(pts, hyperparameter):
    random.seed(0)
    
    #점 3개지나는 평면의 방정식 abcd 튜플로 리턴
    def findPlane(p1, p2, p3):
        v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
        v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
        normal = np.cross(v12,v13)
        if np.linalg.norm(normal) < 0.0000001:
            return None
        d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
        return (normal[0], normal[1], normal[2], d)
    
    #점 p랑 ax+by+cz+d=0 수직거리
    def pointPlaneDistance(p, plane):
        a, b, c, d = plane[0], plane[1], plane[2], plane[3]
        x, y, z = p.x, p.y, p.z
        res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
        return res


    numOfpts = len(pts)
    if numOfpts < 3:
        raise Exception('len(pts) < 3')
    maxScore = 0
    bestPlane = None
    for trial in range(50):
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
            d = pointPlaneDistance(p, plane)
            if d < hyperparameter.H1: 
                score += 1
        if score > maxScore:
            maxScore = score
            bestPlane = plane
    return bestPlane, maxScore



#Input: pts, mainpoint = 이 점을 평면이 지남. Output: pts를 포함하는 평면, Method: PCA,
def PCAplane(pts, mainpoint = None):
    
    Y = np.array([[p.x, p.y, p.z] for p in pts])
    pca = PCA(n_components=3)
    pca.fit(Y)
    V = pca.components_.T
    x_pca_axis, y_pca_axis, z_pca_axis = V
    
    if not mainpoint:
        mainpointArr = np.array([float(0),float(0),float(0)])
        for p in Y:
            mainpointArr += np.array([p[0], p[1], p[2]])
        mainpointArr /= len(Y)
    else:
        mainpointArr = np.array([mainpoint.x, mainpoint.y, mainpoint.z])
    
    d = -np.dot(z_pca_axis, mainpointArr)
    return (z_pca_axis[0], z_pca_axis[1], z_pca_axis[2], d)
    

def normalVectorizeRatioNew(point, BoundaryPoints, CenterPoints, hyperparameter, BoundaryRatio, CenterRatio):
    plane = PCAplane(point.nearby1, point)
    plane_normal = np.array([plane[0], plane[1], plane[2]])
    plane_normal /= np.linalg.norm(plane_normal)

    #점 p랑 ax+by+cz+d=0 수직거리
    def pointPlaneDistance(p, plane):
        a, b, c, d = plane[0], plane[1], plane[2], plane[3]
        x, y, z = p.x, p.y, p.z
        res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
        return res
    
    score = 0
    for p in point.nearby1:
            d = pointPlaneDistance(p, plane)
            if d < hyperparameter.H1: 
                score += 1
    planeRatio = score / len(point.nearby1)

    #planeRatio = PCAscore #이거 잘 안된다. log라서 음수나오고 그럼.
    
    if planeRatio > hyperparameter.ratioThreshold1: #경계인 경우는 그대로 None
        point.normal = plane_normal
        CenterPoints.append(point)
        CenterRatio.append(planeRatio)
    else: 
        BoundaryPoints.append(point)
        BoundaryRatio.append(planeRatio)
    return BoundaryPoints, CenterPoints, BoundaryRatio, CenterRatio




def normalVectorizeRatio(point, BoundaryPoints, CenterPoints, hyperparameter, BoundaryRatio, CenterRatio):
    plane, maxScore = nearbyRansacPlane(point.nearby1, hyperparameter)
    plane_normal = np.array([plane[0], plane[1], plane[2]])
    plane_normal /= np.linalg.norm(plane_normal)

    planeRatio = maxScore / len(point.nearby1)

    
    if planeRatio > hyperparameter.ratioThreshold1: #경계인 경우는 그대로 None
        point.normal = plane_normal
        CenterPoints.append(point)
        CenterRatio.append(planeRatio)
    else: 
        BoundaryPoints.append(point)
        BoundaryRatio.append(planeRatio)
    return BoundaryPoints, CenterPoints, BoundaryRatio, CenterRatio

    



def findNormal(AllPoints, BoundaryPoints, CenterPoints, hyperparameter):
    BoundaryRatio = []
    CenterRatio = []
    for p in AllPoints.values():
        BoundaryPoints, CenterPoints, BoundaryRatio, CenterRatio = normalVectorizeRatioNew(p, BoundaryPoints, CenterPoints, hyperparameter, BoundaryRatio, CenterRatio)
    print('BoundaryRatio')
    print(BoundaryRatio[:200])
    print()
    print('CenterRatio')
    print(CenterRatio[:200])
    return BoundaryPoints, CenterPoints
