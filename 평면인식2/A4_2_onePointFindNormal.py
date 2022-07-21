import numpy as np
from A4_1_nearbyPlane import *
from sklearn.linear_model import LinearRegression

#Input: point, Output: point의 Normal Vector 설정해줌. 리턴값은 없음, Error가 일정 이상 넘으면 경계로 처리방식        
#BoundaryPoints, CenterPoints를 업데이트해줌
def normalVectorizeError(point, BoundaryPoints, CenterPoints, hyperparameter, BoundaryError, CenterError):
    plane = nearbyRansacPlane(point, hyperparameter)
    plane_normal = np.array([plane[0], plane[1], plane[2]])
    plane_normal /= np.linalg.norm(plane_normal)

    planeError = 0
    for neighbor in point.nearby:
        planeError += sujikDistance(neighbor, plane) ** 2
    planeError = np.sqrt(planeError/len(point.nearby))
    #print(planeError)

    
    if planeError < hyperparameter.ransacErrorThreshold: #경계인 경우는 그대로 None
        point.normal = plane_normal #울퉁불퉁한거 하고나서 새로운 아이디어
        CenterPoints.append(point)
        CenterError.append(planeError)
    else: 
        BoundaryPoints.append(point)
        BoundaryError.append(planeError)
    return BoundaryPoints, CenterPoints, BoundaryError, CenterError


def normalVectorizeR2score(point, BoundaryPoints, CenterPoints, hyperparameter, BoundaryError, CenterError):
    pts = point.nearby
    XY = np.array([[p.x, p.y] for p in pts])
    Z = np.array([p.z for p in pts])
    reg = LinearRegression().fit(XY, Z) 
    coef = reg.coef_
    a, b, c, d = coef[0], coef[1], -1, reg.intercept_
    #z = ax+by+d

    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)

    score = reg.score(XY, Z)
    #print(score)

    
    if score > 0.3:
        point.normal = plane_normal 
        CenterPoints.append(point)
        CenterError.append(score)
    else: 
        BoundaryPoints.append(point)
        BoundaryError.append(score)
    return BoundaryPoints, CenterPoints, BoundaryError, CenterError