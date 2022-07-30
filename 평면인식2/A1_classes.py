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
        self.nearby2 = []
        self.normal = None #평면 법선벡터
        self.planeClass = None
        self.edgeClass = None
        self.vertexClass = None
       
    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z)
    
    def distance(self, p):
        def sqsbt(a,b):
            return (a-b)**2
        return (sqsbt(self.x,p.x)+sqsbt(self.y,p.y)+sqsbt(self.z,p.z))**0.5



class Hyperparameter:
    #여기있는건 realdata 기준 값들, 거리단위 m
    def __init__(self, numOfPoints = 5000, 
                R1 = 0.05, R2 = 0.08, OutlierThreshold1 = 40, H1 = 0.01, ratioThreshold1 = 0.8,
                eps_normal = 0.05, min_samples_normal = 10,
                eps_centerPoint = 0.05, min_samples_centerPoint = 10,
                eps_finalBoundaryPoint = 0.05, min_samples_finalBoundaryPoint = 8):

        #2 data
        self.numOfPoints = numOfPoints #generatepoint 점개수

        #3 allFindNearby
        self.R1 = R1 #AllPoints nearby
        self.R2 = R2

        #4 findNormal
        self.OutlierThreshold1 = OutlierThreshold1 #noiseR 이내에 outlier 보다 적게있으면 이상점
        self.H1 = H1 #법선벡터구할때 랜색 오차허용범위 = H 
        self.ratioThreshold1 = ratioThreshold1 #ratio 방법으로 했을때 ratio 이거보다 크면 내부점

        #5 vectorClustering
        self.eps_normal = eps_normal #vector DBSCAN eps
        self.min_samples_normal = min_samples_normal #vector DBSCAN min_samples

        #6 3DdistanceStairClustering
        self.eps_centerPoint = eps_centerPoint #centerpoint DBSCAN eps
        self.min_samples_centerPoint = min_samples_centerPoint #centerpoint DBSCAN min_samples
        
        #7 findEquations
        #self.planeRansacThreshold = planeRansacThreshold
        #최종적으로 평면 만들때 쓰는 랜색 오차허용범위
        #이건 그냥 H1 쓰자

        #8 objectSegmentation
        self.eps_finalBoundaryPoint = eps_finalBoundaryPoint #boundarypoint DBSCAN eps
        self.min_samples_finalBoundaryPoint= min_samples_finalBoundaryPoint #boundarypoint DBSCAN min_samples
    
        
class Plane:
    def __init__(self, label, interiorPoints):
        self.label = label
        self.interiorPoints = interiorPoints
        self.planeEdgeDict = defaultdict(Edge) 
        #key는 다른 plane, (연결된 plane만 있음)
        #value는 그 plane과 사이에 있는 edge class. 
        self.containedObj = set()
        self.equation = None #(a,b,c,d)
        self.polygon = [] #vertices를 순서에 맞게 배열한것

class Edge:
    def __init__(self, label, linePoints):
        self.label = label
        self.linePoints = linePoints
        self.vertex = set() #len = 2일거임

class Vertex:
    def __init__(self, label, dotPoints):
        self.label = label
        self.dotPoints = dotPoints
        self.edges = set()
        self.mainPoint = None #point class 아니라 그냥 길이 3짜리 Numpy array임
  
class Object:
    def __init__(self, idx, BoundaryPoints):
        self.idx = idx
        self.BoundaryPoints = BoundaryPoints  
        self.planes = set() # Plane형
        self.edges = set()
        self.vertices = set()