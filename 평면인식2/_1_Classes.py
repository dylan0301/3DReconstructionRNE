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
        return ((sqsbt(self.x, p.x)+sqsbt(self.y, p.y)+sqsbt(self.z, p.z)))**(0.5)

class Hyperparameter:
    #여기있는건 realdata 기준 값들, 거리단위 m
    def __init__(self, pointLeastDifference = 0.0001, numOfPoints = 5000, OutlierThreshold = 8, 
                noiseR = 0.2, friend = 12, vectorRansacTrial = 20, vectorRansacThreshold = 0.15,
                stdThreshold = 0.5, numOfClusters = 5, step_threshold = 0.3, planeRansacTrial = 20,
                planeRansacThreshold = 0.15):

        #2 data
        self.pointLeastDifference = pointLeastDifference #각 좌표값 차이가 이거보다 가까이 있는 점 쌍은 하나로 취급
        self.numOfPoints = numOfPoints #generatepoint 점개수

        #3 removeNoise
        self.OutlierThreshold = OutlierThreshold #noiseR 이내에 outlier 보다 적게있으면 이상점
        self.noiseR = noiseR #이상점걸러내기용 R
        
        #4 nearby
        self.friend = friend #nearby 크기
        
        #5 findNormal
        self.vectorRansacTrial = vectorRansacTrial #법선벡터구할때 랜색 시행횟수
        self.vectorRansacThreshold = vectorRansacThreshold#법선벡터구할때 랜색 오차허용범위
        self.stdThreshold = stdThreshold #외적 벡터드표준편차 이거보다 크면 경계점
        
        #6 vectorClustering
        self.numOfClusters = numOfClusters #벡터클러스터링할때 클러스터 개수 (곱하기2 안한것)

        #7 distanceStairClustering
        self.step_threshold = step_threshold #stair 클러스터링에서 이값보다 더많이 점프하면 다른평면
        
        #9 findPlane
        self.planeRansacTrial = planeRansacTrial #최종 평면 구할때 랜색 시행횟수
        self.planeRansacThreshold = planeRansacThreshold #최종적으로 평면 만들때 쓰는 랜색 오차허용범위


        #10 boundaryRemoveNoise
        #여기서 OutlierThreshold랑 noiseR 다르게 설정 필요할듯
        


class plane:
    def __init__(self, polygon, equation):
        self.equation = equation
        self.polygon = polygon
        
    def __str__(self):
        return self.polygon
        