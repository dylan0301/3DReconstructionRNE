import numpy as np

class Point:
    def __init__(self, X, Y, Z, idx, R = None, G = None, B = None):
        self.x = X 
        self.y = Y 
        self.z = Z
        self.idx = idx
        self.R = R
        self.G = G
        self.B = B
        self.nearby = list() #가까운 점들의 (distance, index)가 들어감
        self.normal = None
        self.cluster = None
       
    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z)
    
    def distance(self, p):
        def sqsbt(a,b):
            return (a-b)**2
        return (sqsbt(self.x,p.x)+sqsbt(self.y,p.y)+sqsbt(self.z,p.z))**0.5


    # def distance(self, p):
    #     a = np.array([self.x, self.y, self.z])
    #     b = np.array([p.x, p.y, p.z])
    #     return np.sqrt(np.dot(a, a) - 2 * np.dot(a, b) + np.dot(b, b))


class Hyperparameter:
    #여기있는건 realdata 기준 값들, 거리단위 m
    def __init__(self, pointLeastDifference = 0.0001, numOfPoints = 5000, OutlierThreshold = 12, 
                R = 0.03, vectorRansacTrial = 50, vectorRansacThreshold = 0.15, normalLeastNorm = 0.001, R2ScoreThreshold = 0.5, ransacScoreThreshold = -0.5,
                ransacErrorThreshold = 0.01, eps_vector = 0.04, min_samples_vector = 12,
                eps_point = 0.05, min_samples_point = 12,  planeRansacTrial = 50,
                planeRansacThreshold = 0.15, boundaryR = 0.06, boundaryOutlierThreshold = 9):

        #2 data
        self.pointLeastDifference = pointLeastDifference #각 좌표값 차이가 이거보다 가까이 있는 점 쌍은 하나로 취급
        self.numOfPoints = numOfPoints #generatepoint 점개수

        #3 removeNoise
        self.OutlierThreshold = OutlierThreshold #noiseR 이내에 outlier 보다 적게있으면 이상점
        self.R = R #이상점걸러내기용 R, nearbyR

        
        #4 findNormal
        self.vectorRansacTrial = vectorRansacTrial #법선벡터구할때 랜색 시행횟수
        self.vectorRansacThreshold = vectorRansacThreshold #법선벡터구할때 랜색 오차허용범위
        self.normalLeastNorm = normalLeastNorm #법선벡터 최소 노름

        self.R2ScoreThreshold = R2ScoreThreshold
        self.ransacScoreThreshold = ransacScoreThreshold #ransac Score 이거보다 크면 내부점
        self.ransacErrorThreshold = ransacErrorThreshold #Error 방법으로 했을때 Error 이거보다 크면 경계점
        
        #5 vectorClustering
        self.eps_vector = eps_vector #vector DBSCAN eps
        self.min_samples_vector = min_samples_vector #vector DBSCAN min_samples

        #6 distanceStairClustering
        self.eps_point = eps_point #point DBSCAN eps
        self.min_samples_point = min_samples_point #point DBSCAN min_samples
        
        #8 findPlane
        self.planeRansacTrial = planeRansacTrial #최종 평면 구할때 랜색 시행횟수
        self.planeRansacThreshold = planeRansacThreshold #최종적으로 평면 만들때 쓰는 랜색 오차허용범위
        
        #9 boundaryRemoveNoise
        #여기서 OutlierThreshold랑 noiseR 다르게 설정 필요할듯
        self.boundaryR = boundaryR
        self.boundaryOutlierThreshold = boundaryOutlierThreshold
        


class plane:
    def __init__(self, polygon, equation):
        self.equation = equation #(a,b,c,d)
        self.polygon = polygon
        
    def __str__(self):
        return self.polygon
        