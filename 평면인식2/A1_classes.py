import numpy as np
from collections import defaultdict

class Point:
    def __init__(self, X, Y, Z, idx, R = None, G = None, B = None):
        self.x = X 
        self.y = Y 
        self.z = Z
        self.idx = idx
        self.R = R
        self.G = G
        self.B = B
        self.nearby1 = [] #AllPoints에서 가까운 point들이 들어감
        self.nearby2 = [] #BoundaryPoints에서 가까운 point들이 들어감
        self.normal = None #평면 법선벡터
        self.direction = None #직선 방향벡터
        # self.planeCluster = None
        # self.edgeClutser = None
        # self.vertexCluster = None
       
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
    def __init__(self, numOfPoints = 5000, OutlierThreshold1 = 40, 
                R1 = 0.05, H1 = 0.01,
                ratioThreshold1 = 0.8, eps_vector = 0.05, min_samples_vector = 10,
                eps_point = 0.05, min_samples_point = 10, R2 = 0.06, OutlierThreshold2 = 15,
                H2 = 0.01, ratioThreshold2 = 0.8,
                planeRansacThreshold = 0.01, eps_point2 = 0.05, min_samples_point2 = 8,
                edgeRansacThreshold = 0.01):

        #2 data
        self.numOfPoints = numOfPoints #generatepoint 점개수

        #3 allFindNearby
        self.R1 = R1 #AllPoints nearby

        
        #4 findNormal
        self.OutlierThreshold1 = OutlierThreshold1 #noiseR 이내에 outlier 보다 적게있으면 이상점
        self.H1 = H1 #법선벡터구할때 랜색 오차허용범위 = H
        
        self.ratioThreshold1 = ratioThreshold1 #ratio 방법으로 했을때 ratio 이거보다 크면 내부점

        #5 vectorClustering
        self.eps_vector = eps_vector #vector DBSCAN eps
        self.min_samples_vector = min_samples_vector #vector DBSCAN min_samples

        #6 distanceStairClustering
        self.eps_point = eps_point #centerpoint DBSCAN eps
        self.min_samples_point = min_samples_point #centerpoint DBSCAN min_samples

        #7 boundaryFindNearby
        self.R2 = R2

        #8 findDirection
        self.OutlierThreshold2 = OutlierThreshold2
        self.H2 = H2 #법선벡터구할때 랜색 오차허용범위 = H
        self.ratioThreshold2 = ratioThreshold2 #ratio 방법으로 했을때 ratio 이거보다 크면 내부점
        
        
        #10 findEquations
        self.planeRansacThreshold = planeRansacThreshold #최종적으로 평면 만들때 쓰는 랜색 오차허용범위
        self.edgeRansacThreshold = edgeRansacThreshold #edge 지나는 직선 만들때 쓰는 랜색 오차허용범위

        #11 objectSegmentation
        self.eps_point2 = eps_point2 #boundarypoint DBSCAN eps
        self.min_samples_point2 = min_samples_point2 #boundarypoint DBSCAN min_samples
    
        
        



class Plane:
    def __init__(self, label, interiorPoints):
        self.label = label
        self.interiorPoints = interiorPoints
        self.planeEdgeDict = defaultdict(list) 
        #key는 다른 plane, (연결된 plane만 있음)
        #value는 그 plane과 사이에 있는 boundarypoints. 
        self.containedObj = set()
        self.equation = None #(a,b,c,d)


class Edge:
    def __init__(self, label, edgePoints):
        self.label = label


class Vertex:
    def __init__(self, label, vertexPoints):
        pass

        
        
class Object:
    def __init__(self, idx, BoundaryPoints):
        self.idx = idx
        self.BoundaryPoints = BoundaryPoints  
        self.planes = set() # Plane형
        
        
    