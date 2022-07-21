import numpy as np
from A4_1_nearbyPlane import *

#Input: point, Output: point의 Normal Vector 설정해줌. 리턴값은 없음, 표준편차방식        
#BoundaryPoints, CenterPoints를 업데이트해줌
def normalVectorizeSTD(point, BoundaryPoints, CenterPoints, hyperparameter):
    neighborVectors = []
    #nearby에는 (distance, point) 가 들어있음
    for tup in point.nearby:
        neighbor = tup[1]
        v = np.array([neighbor.x-point.x, neighbor.y-point.y, neighbor.z-point.z])
        neighborVectors.append(v)   
   
    plane = nearbyRansacPlane(point, hyperparameter)
    plane_normal = np.array([plane[0], plane[1], plane[2]])
    plane_normal /= np.linalg.norm(plane_normal)

    normalCandidates = []
    for i in range(len(neighborVectors)):
        for j in range(i+1, len(neighborVectors)):
            neighborNormal = np.cross(neighborVectors[i], neighborVectors[j])
            if np.linalg.norm(neighborNormal) < hyperparameter.normalLeastNorm: #외적한거 너무 작으면 취급 x
                continue

            neighborNormal /= np.linalg.norm(neighborNormal)

            #주어진 평면 벡터와 비슷한 벡터를 골라줄거임 (directionFix)
            cosine = np.dot(neighborNormal, plane_normal)/np.linalg.norm(neighborNormal)/np.linalg.norm(plane_normal)
            if cosine < 0: 
                neighborNormal *= -1            
            normalCandidates.append(neighborNormal)

    avg = np.array([float(0),float(0),float(0)])
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



#Input: point, Output: point의 Normal Vector 설정해줌. 리턴값은 없음, Error가 일정 이상 넘으면 경계로 처리방식        
#BoundaryPoints, CenterPoints를 업데이트해줌
def normalVectorizeError(point, BoundaryPoints, CenterPoints, hyperparameter, BoundaryError, CenterError):
    plane = nearbyRansacPlane(point, hyperparameter)
    plane_normal = np.array([plane[0], plane[1], plane[2]])
    plane_normal /= np.linalg.norm(plane_normal)

    planeError = 0
    for tup in point.nearby:
        neighbor = tup[1]
        planeError += sujikDistance(neighbor, plane) ** 2
    planeError = np.sqrt(planeError/len(point.nearby))
    #print(planeError)

    
    if planeError < hyperparameter.ransacErrorThreshold: #경계인 경우는 그대로 None
        #point.normal = avg
        point.normal = plane_normal #울퉁불퉁한거 하고나서 새로운 아이디어
        CenterPoints.append(point)
        CenterError.append(planeError)
    else: 
        BoundaryPoints.append(point)
        BoundaryError.append(planeError)
    return BoundaryPoints, CenterPoints, BoundaryError, CenterError


#Input: point, Output: point의 Normal Vector 설정해줌. 리턴값은 없음, 삼중곱 크기 방식      
#BoundaryPoints, CenterPoints를 업데이트해줌
def normalVectorizeTriple(point, BoundaryPoints, CenterPoints, hyperparameter):
    neighborVectors = []
    #nearby에는 (distance, point) 가 들어있음
    for tup in point.nearby:
        neighbor = tup[1]
        v = np.array([neighbor.x-point.x, neighbor.y-point.y, neighbor.z-point.z])
        neighborVectors.append(v)   
   
    plane = nearbyRansacPlane(point, hyperparameter)
    plane_normal = np.array([plane[0], plane[1], plane[2]])
    plane_normal /= np.linalg.norm(plane_normal)

    
    for i in range(len(neighborVectors)):
        for j in range(i+1, len(neighborVectors)):
            for k in range(j+1, len(neighborVectors)):
                tripled = np.dot(neighborVectors[i], np.cross(neighborVectors[j], neighborVectors[k]))
         

    avg = np.array([float(0),float(0),float(0)])
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