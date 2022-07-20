import numpy as np
from A4_1_nearbyPlane import *


#Input: point, Output: point의 Normal Vector 설정해줌. 리턴값은 없음, 표준편차방식        
#BoundaryPoints, CenterPoints를 업데이트해줌
def normalVectorizeSTD(point, hyperparameter, BoundaryPoints, CenterPoints):
    neighborVectors = []
    #nearby에는 (distance, point) 가 들어있음
    for tup in point.nearby:
        neighbor = tup[1]
        v = np.array([neighbor.x-point.x, neighbor.y-point.y, neighbor.z-point.z])
        neighborVectors.append(v)   
   
    plane = nearbyRansacPlane(point, hyperparameter)
    plane_normal = np.array([plane[0], plane[1], plane[2]])

    normalCandidates = []
    for i in range(len(neighborVectors)):
        for j in range(i+1, len(neighborVectors)):
            neighborNormal = np.cross(neighborVectors[i], neighborVectors[j])
            neighborNormal /= np.linalg.norm(neighborNormal)

            #주어진 평면 벡터와 비슷한 벡터를 골라줄거임 (directionFix)
            cosine = np.dot(neighborNormal, plane_normal)/np.linalg.norm(neighborNormal)/np.linalg.norm(plane_normal)
            if cosine < 0: 
                neighborNormal *= -1            
            normalCandidates.append(neighborNormal)

    avg = np.array([0,0,0])
    for neighborNormal in normalCandidates:
        avg += neighborNormal
    avg /= np.linalg.norm(avg)
    

    standardDeviation = 0
    for neighborNormal in normalCandidates:
        standardDeviation += (np.linalg.norm(avg-neighborNormal))**2
    standardDeviation = np.sqrt(standardDeviation/len(normalCandidates))

    if standardDeviation < hyperparameter.stdThreshold: #경계인 경우는 그대로 None
        #point.normal = avg
        point.normal = plane_normal #울퉁불퉁한거 하고나서 새로운 아이디어
        CenterPoints.append(point)
    else: 
        BoundaryPoints.append(point)
    return BoundaryPoints, CenterPoints


def normalVectorizeErrorRate(point, hyperparameter, BoundaryPoints, CenterPoints):
    neighborVectors = []