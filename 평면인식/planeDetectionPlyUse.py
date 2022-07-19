from turtle import resetscreen
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import time
from collections import defaultdict
import pointCloudImport



#############-realpoint*100 hyperparameter-#################
friend = 12 #nearby 크기
nearbyLeastDistance = 0.1 #1차랜색에서 지나치게 가까이 있는점은 잘못된데이터
r = 20 #이상점걸러내기용
OutlierThreshold = 8 #r 이내에 outlier 보다 적게있으면 이상점
normalLeastNorm = 0.1 #유의미한 벡터쌍들에 대해서만 법선벡터 계산
stdThreshold = 0.7 #표준편차 이거보다 크면 경계점
step_threshold = 20 #2차클러스터링에서 이값보다 더많이 점프하면 다른평면
numOfPoints = 5000 #generatepoint 점개수
ransacTrial1 = 20 #법선벡터구할때 랜색 시행횟수
ransacTrial2 = 20 #최종 평면 구할때 랜색 시행횟수
ransacThreshold1 = 20 #법선벡터구할때 랜색 오차허용범위
ransacThreshold2 = 10 #최종적으로 평면 만들때 쓰는 랜색 오차허용범위
howmanyclusters = 5


#############-cube hyperparameter-#################
friend = 12 #nearby 크기
nearbyLeastDistance = 0.01 #1차랜색에서 지나치게 가까이 있는점은 잘못된데이터
r = 5 #이상점걸러내기용
OutlierThreshold = 8 #r 이내에 outlier 보다 적게있으면 이상점
normalLeastNorm = 0.01 #유의미한 벡터쌍들에 대해서만 법선벡터 계산
stdThreshold = 0.6 #표준편차 이거보다 크면 경계점
step_threshold = 20 #2차클러스터링에서 이값보다 더많이 점프하면 다른평면
numOfPoints = 3000 #generatepoint 점개수
ransacTrial1 = 20 #법선벡터구할때 랜색 시행횟수
ransacTrial2 = 20 #최종 평면 구할때 랜색 시행횟수
ransacThreshold1 = 2 #법선벡터구할때 랜색 오차허용범위
ransacThreshold2 = 2 #최종적으로 평면 만들때 쓰는 랜색 오차허용범위
howmanyclusters = 3

#############-realpoint*100 hyperparameter-#################
friend = 12 #nearby 크기
nearbyLeastDistance = 0.1 #1차랜색에서 지나치게 가까이 있는점은 잘못된데이터
r = 20 #이상점걸러내기용
OutlierThreshold = 8 #r 이내에 outlier 보다 적게있으면 이상점
normalLeastNorm = 0.1 #유의미한 벡터쌍들에 대해서만 법선벡터 계산
stdThreshold = 0.7 #표준편차 이거보다 크면 경계점
step_threshold = 20 #2차클러스터링에서 이값보다 더많이 점프하면 다른평면
numOfPoints = 5000 #generatepoint 점개수
ransacTrial1 = 20 #법선벡터구할때 랜색 시행횟수
ransacTrial2 = 20 #최종 평면 구할때 랜색 시행횟수
ransacThreshold1 = 20 #법선벡터구할때 랜색 오차허용범위
ransacThreshold2 = 10 #최종적으로 평면 만들때 쓰는 랜색 오차허용범위
howmanyclusters = 5





dist_stdard_threshold = 2 #안씀

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
    
    #다른 모든 점들과의 거리들을 크기순(오름차순)으로 모두 찾아준다. (self.nearby를 업데이트 해주는 함수)
    def updateNearby(self, distMat):
        global friend
        l = distMat[self.idx]
        res = sorted(l.values(), key = lambda x: x[0])
        self.nearby = res[1:friend+1]
        # delete_candidates = set() #res에서의 index가 들어감
        # global nearbyLeastDistance
        # for i in range(2, len(res)): #자기자신 빼고하게 2부터
        #     nowpoint = AllPoints[res[i][1]]
        #     beforepoint = AllPoints[res[i-1][1]]
        #     if nowpoint.distance(beforepoint) < nearbyLeastDistance: #nearby에서 지나치게 가까이 있는건 잘못된데이터
        #         delete_candidates.add(i-1)
        # for i in range(1, len(res)):
        #     if i not in delete_candidates:
        #         self.nearby.append(res[i])
        #     if len(self.nearby) >= friend:
        #         break        

    #자신과 nearby점들로 선형회귀해서 평면찾음, return값은 그 평면의 법선벡터
    #이 법선벡터를 바로 법선벡터로 하는것도 고려해보자. 경계값때문에 그렇게 하진 않을거같긴함
    #Ransac 쓸수도 있을거같은데 그럼 경계점에서 문제생길거같음
    #선형회귀 구현 힘들어서 그냥 랜색쓴다 n번 반복함
    def ransacVector(self, n):
        global r

        #점 3개지나는 평면의 방정식 abcd 튜플로 리턴
        def findPlane(p1, p2, p3):
            v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
            v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
            normal = np.cross(v12,v13)
            d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
            return normal[0], normal[1], normal[2], d

        #점 p랑 ax+by+cz+d=0 수직거리
        def sujikDistance(p, a,b,c,d):
            x = p.x
            y = p.y
            z = p.z
            res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
            return res
        
        pts = []
        for p in self.nearby:
            q = AllPoints[p[1]]
            pts.append(q)
        pts.append(self)

        numOfPoints = len(pts)
        maxScore = 0
        bestNormal = None
        global nearbyLeastDistance
        for i in range(n):
            p1 = pts[random.randrange(0,numOfPoints)]
            p2 = pts[random.randrange(0,numOfPoints)]
            while p1.distance(p2) < nearbyLeastDistance:
                p2 = pts[random.randrange(0,numOfPoints)]
            p3 = pts[random.randrange(0,numOfPoints)]
            while p3.distance(p1) < nearbyLeastDistance or p3.distance(p2) < nearbyLeastDistance:
                p3 = pts[random.randrange(0,numOfPoints)]
            a, b, c, d = findPlane(p1, p2, p3)
            score = 0
            global ransacThreshold1
            for p in pts:
                sujikk = sujikDistance(p,a,b,c,d)
                if sujikk > 100:
                    print(sujikk)
                if sujikk < ransacThreshold1: #r/3이 여기서 랜색 threshold임<<이렇게하니까 문제생김
                    score +=1
            if score > maxScore:
                maxScore = score #이거 넣는걸 깜빡해서 문제가 좀 생겼음. 근데 정육면체데이터는 괜찮았네?
                planeNormal = Vector(np.array([a,b,c]))
                planeNormal.normalize()
                bestNormal = planeNormal
        return bestNormal


    #Input: Point, Output: Normal Vector        
    def normalVectorize(self):
        vectors = []
        for otherpoint in self.nearby:
            q = AllPoints[otherpoint[1]]
            v = np.array([q.x-self.x, q.y-self.y, q.z-self.z])
            v = Vector(v)
            vectors.append(v)
        
        global ransacTrial1, normalLeastNorm
        normals = []
        plane_normal = self.ransacVector(ransacTrial1)
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                nor = vectors[i].cross(vectors[j])
                if np.linalg.norm(nor.vector) > normalLeastNorm: #유의미한 벡터쌍들에 대해서만 법선벡터 계산
                    nor.normalize()
                    nor.directionFix(plane_normal) #여기에 plane의 Normal vector 들어간다.
                    normals.append(nor)
        
        

        avg = Vector(np.array([0,0,0]))
        for nor in normals:
            avg.vector = avg.vector + nor.vector
        #avg.vector = avg.vector/len(normals)
        avg.normalize()
        
        
        if self.idx == 0:
            for noooo in normals:
                print(noooo)
            print(avg, 'avg')

        standardDeviation = 0
        for nor in normals:
            standardDeviation += (avg.distance(nor))**2
        standardDeviation = np.sqrt(standardDeviation/len(normals))


        global stdThreshold
        avg.normalize()
        if standardDeviation < stdThreshold: #경계인 경우는 그대로 None
            self.normal = avg
            CenterPoints.append(self)
        else: 
            BoundaryPoints.append(self)
        

class Vector:
    def __init__(self, arr3):
        self.vector = arr3 #np.arr([dx, dy, dz])

    def __str__(self):
        return "dx: " + str(self.vector[0]) + ", dy: " + str(self.vector[1]) + ", dz: " + str(self.vector[2])

    #self벡터와 u벡터간의 norm2 거리
    def distance(self, u):
        w = self.vector - u.vector
        return np.linalg.norm(w)
    
    #self벡터와 u벡터간의 각도 cosine, directionfix에서 사용됨
    def cosine_angle(self, u):
        cosine = np.dot(self.vector, u.vector)/np.linalg.norm(self.vector)/np.linalg.norm(u.vector)
        return cosine

    #self벡터와 u벡터간의 각도인데 무조건 예각임, 1차 클러스터링에서 사용됨
    def angleDistance(self, u):
        cosine = np.dot(self.vector, u.vector)/np.linalg.norm(self.vector)/np.linalg.norm(u.vector)
        cosine = abs(cosine)
        theta = np.arccos(cosine)
        return theta
    
    #self X u
    def cross(self, u):
        return Vector(np.cross(self.vector, u.vector))

    def normalize(self):
        self.vector /= np.linalg.norm(self.vector)

    #주어진 평면 벡터와 비슷한 벡터를 골라줄거임
    def directionFix(self, normalVector):
        if self.cosine_angle(normalVector) < 0: 
            self.vector = self.vector*(-1)




#####################-point cloud data-#####################
def importPly():
    points = defaultdict(Point)
    pointcloud = pointCloudImport.pointcloud
    for i in range(len(pointcloud)):
        p = Point(pointcloud[i][0]*100, pointcloud[i][1]*100, pointcloud[i][2]*100, i, pointcloud[i][6], pointcloud[i][7], pointcloud[i][8])
        p.normal = Vector(np.array([pointcloud[i][3], pointcloud[i][4], pointcloud[i][5]]))
        #100 곱하는건 cm로 바꿀려고
        points[i] = p
        CenterPoints.append(p)
    return points

    
def generatePoints1(size):
    points = defaultdict(Point)
    for i in range(size-100):
        if i%3 == 0:
            x = 100*random.random()
            y = 100*random.random()
            z = 0
        if i%3 == 1:
            x = 100*random.random()
            y = 0
            z = 100*random.random()
        if i%3 == 2:
            x = 0
            y = 100*random.random()
            z = 100*random.random()              
        p = Point(x,y,z,i)
        points[i] = p
    for i in range(100):
        p = Point(100*random.random(), 100*random.random(), 100*random.random(), size-100+i)
        points[size-100+i] = p
    return points

def generatePoints2(size):
    points = defaultdict(Point)
    for i in range(size-100):
        if i%3 == 0:
            x = 100*random.random()
            y = 100*random.random()
            z = 0
        if i%3 == 1:
            x = 100*random.random()
            y = 0
            z = 100*random.random()
        if i%3 == 2:
            x = 100*random.random()
            y = 100*random.random()
            z = 100-y            
        p = Point(x,y,z,i)
        points[i] = p
    for i in range(100):
        p = Point(100*random.random(), 100*random.random(), 100*random.random(), size-100+i)
        points[size-100+i] = p
    return points

def generatePoints3(size):
    points = defaultdict(Point)
    for i in range(size-100):
        x,y,z = 100, 100, 100
        if i%4 == 0:
            while x + y > 100:
                x = 100*random.random()
                y = 100*random.random()
                z = 0
        if i%4 == 1:
            while x + z > 100:
                x = 100*random.random()
                y = 0
                z = 100*random.random()
        if i%4 == 2:
            while y + z > 100:
                x = 0
                y = 100*random.random()
                z = 100*random.random()  
        if i%4 == 3:
            while x + y > 100:
                x = 100*random.random()
                y = 100*random.random()
                z = 100 - x - y
        p = Point(x,y,z,i)
        points[i] = p
    for i in range(100):
        p = Point(100*random.random(), 100*random.random(), 100*random.random(), size-100+i)
        points[size-100+i] = p
    return points

def generatePoints4(size):
    points = defaultdict(Point)
    for i in range(size-100):
        if i%6 == 0:
            x = 30*random.random()
            y = 30*random.random()
            z = 0
        if i%6 == 1:
            x = 30*random.random()
            y = 0
            z = 30*random.random()
        if i%6 == 2:
            x = 0
            y = 30*random.random()
            z = 30*random.random()
        if i%6 == 3:
            x = 30*random.random()
            y = 30*random.random()
            z = 30
        if i%6 == 4:
            x = 30*random.random()
            y = 30
            z = 30*random.random()
        if i%6 == 5:
            x = 30
            y = 30*random.random()
            z = 30*random.random()                  
        p = Point(x,y,z,i)
        points[i] = p
    for i in range(100):
        p = Point(30*random.random(), 30*random.random(), 30*random.random(), size-100+i)
        points[size-100+i] = p
    return points

def generatePoints5(size):
    points = defaultdict(Point)
    for i in range(size-100):
        if i%8 == 0:
            x = 30*random.random()
            y = 30*random.random()
            z = 0
                 
        p = Point(x,y,z,i)
        points[i] = p
    for i in range(100):
        p = Point(30*random.random(), 30*random.random(), 30*random.random(), size-100+i)
        points[size-100+i] = p
    return points

#울퉁불퉁 정육면체
def generatePoints6(size):
    points = defaultdict(Point)
    for i in range(size-100):
        diff = 4*random.random()-2
        if i%6 == 0:
            x = 30*random.random()
            y = 30*random.random()
            z = diff
        if i%6 == 1:
            x = 30*random.random()
            y = diff
            z = 30*random.random()
        if i%6 == 2:
            x = diff
            y = 30*random.random()
            z = 30*random.random()
        if i%6 == 3:
            x = 30*random.random()
            y = 30*random.random()
            z = 30+diff
        if i%6 == 4:
            x = 30*random.random()
            y = 30+diff
            z = 30*random.random()
        if i%6 == 5:
            x = 30+diff
            y = 30*random.random()
            z = 30*random.random()                  
        p = Point(x,y,z,i)
        points[i] = p
    for i in range(100):
        p = Point(36*random.random()-3, 36*random.random()-3, 36*random.random()-3, size-100+i)
        points[size-100+i] = p
    return points

random.seed(8)
#AllPoints = generatePoints4(numOfPoints)
CenterPoints = []
BoundaryPoints = []
AllPoints = importPly()

###################### -find normal vector-  ############################3

def removeNoise():
    t = time.time()
    global friend, r, OutlierThreshold
    size = len(AllPoints)
    distMat = defaultdict(dict)
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (AllPoints[i].distance(AllPoints[j]),j) 

    for i in range(size):
        AllPoints[i].updateNearby(distMat)

    #노이즈 제거
    del_candidate = [] #index들 리스트
    for i in range(size):
        if AllPoints[i].nearby[OutlierThreshold-1][0] > r:
            del_candidate.append(i)
            del distMat[i]
            
    for j in del_candidate:
        del AllPoints[j]
        
    for i in AllPoints.keys():
        for j in del_candidate:
            del distMat[i][j]
            
    for i in AllPoints.keys():
        AllPoints[i].updateNearby(distMat)

    print("리무브노이스+메이크니어바이 수행시간: ", time.time()-t)


def makeNormal():
    t = time.time()
    size = len(AllPoints)
    for i in AllPoints.keys():
        AllPoints[i].normalVectorize()
    print("메이크노멀 수행시간: ", time.time()-t)


####################### 1차 클러스터링##################


def vectorClustering(numOfCluster):
    print('vectorClustering start')
    t = time.time()
    Duplicatedvectors = np.array([p.normal.vector for p in CenterPoints] + [(-1)*p.normal.vector for p in CenterPoints])
    ac = AgglomerativeClustering(n_clusters=numOfCluster, affinity="euclidean", linkage="complete")
    labels = ac.fit_predict(Duplicatedvectors)
    
    print('vectorclustering step1')

    oppositeVector = []
    for i in range(len(CenterPoints)):
        if sorted([labels[i], labels[i+len(CenterPoints)]]) not in oppositeVector:
            oppositeVector.append(sorted([labels[i], labels[i+len(CenterPoints)]]))

    print(oppositeVector)
        
    del_target_vec = [oppositeVector[i][0] for i in range(len(oppositeVector))]
    
    print(del_target_vec)
    
    newLabel = [None] * len(CenterPoints)
    for i in range(len(labels)):
        if labels[i] not in del_target_vec:
            newLabel[i%len(CenterPoints)] = labels[i]

    newLabel += [numOfCluster] * len(BoundaryPoints)
    plotAll = CenterPoints + BoundaryPoints

    # print("클러스터 분류 결과:", newLabel)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Allvectors[:,0],Allvectors[:,1], Allvectors[:,2], c=ac.labels_, marker='o', s=15, cmap='rainbow')
    ap = np.array([[p.x, p.y, p.z] for p in plotAll])
    ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=newLabel, marker='o', s=15, cmap='rainbow')
    print("클러스터링 수행시간: ", time.time()-t)
    # plt.show()
    clusterPointMap = defaultdict(list) #index = cluster 번호, index 안에는 포인트
    for i in range(len(CenterPoints)):
        clusterPointMap[newLabel[i]].append(CenterPoints[i])
    return clusterPointMap

#https://wikidocs.net/92111

#duplicate 안하고 그냥하기    
def vectorClustering2(numOfCluster):
    print('vectorClustering2 start')
    t = time.time()
    Centervectors = np.array([p.normal.vector for p in CenterPoints])
    ac = AgglomerativeClustering(n_clusters=numOfCluster, affinity="euclidean", linkage="complete")
    labels = ac.fit_predict(Centervectors)
    
    print('vectorclustering2 step1')

    labels = list(labels)

    labels += [numOfCluster] * len(BoundaryPoints)
    plotAll = CenterPoints + BoundaryPoints

    # print("클러스터 분류 결과:", newLabel)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Allvectors[:,0],Allvectors[:,1], Allvectors[:,2], c=ac.labels_, marker='o', s=15, cmap='rainbow')
    ap = np.array([[p.x, p.y, p.z] for p in plotAll])
    ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=labels, marker='o', s=15, cmap='rainbow')
    print("클러스터링 수행시간: ", time.time()-t)
    # plt.show()
    clusterPointMap = defaultdict(list) #index = cluster 번호, index 안에는 포인트
    for i in range(len(CenterPoints)):
        clusterPointMap[labels[i]].append(CenterPoints[i])
    return clusterPointMap    
    

####################### 2차 클러스터링##################
NewClusterPointMap = defaultdict(list)
FinalPlanes = []

def divideCluster_stairmethod(Cluster):
    print("계단식 디바이드클러스터 실행됨")
    avg = Vector(np.array([0,0,0]))
    for p in Cluster:
        avg.vector = avg.vector + p.normal.vector
    avg.normalize()
    print(avg)

    #점 p랑 ax+by+cz+d=0 수직거리. a,b,c는 avg벡터고 d=0
    def shortestDistance(p):
        x = p.x
        y = p.y
        z = p.z
        a = avg.vector[0]
        b = avg.vector[1]
        c = avg.vector[2]
        d = 0
        res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
        return res

    short_dis = [(shortestDistance(p),p) for p in Cluster]
    short_dis.sort(key = lambda x: x[0])
    labels = [None] * len(Cluster)
    
    # #거리가 어떤지 출력해보기
    # for teststst in short_dis:
    #     print(teststst[0])
    # print()
    
    global step_threshold
    
    label = 0
    for i in range(len(short_dis)):
        if i == 0: labels[Cluster.index(short_dis[i][1])] = label
        else:
            stepGap = abs(short_dis[i][0] - short_dis[i-1][0])
            if stepGap > step_threshold:
                label += 1
                labels[Cluster.index(short_dis[i][1])] = label
            else:
                labels[Cluster.index(short_dis[i][1])] = label
    
    clusterNow = len(NewClusterPointMap)
    for i in range(len(Cluster)):
        NewClusterPointMap[labels[i]+clusterNow].append(Cluster[i])

def divideCluster(Cluster):
    print("디바이드클러스터 실행됨")
    avg = Vector(np.array([0,0,0]))
    for p in Cluster:
        avg.vector = avg.vector + p.normal.vector
    avg.normalize()
    print(avg)

    #점 p랑 ax+by+cz+d=0 수직거리. a,b,c는 avg벡터고 d=0
    def shortestDistance(p):
        x = p.x
        y = p.y
        z = p.z
        a = avg.vector[0]
        b = avg.vector[1]
        c = avg.vector[2]
        d = 0
        res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
        return res
    
    def stdDev(data, means):
        size = len(data)
        s = 0
        for i in range(size):
            s += (data[i][0]-means[i])**2
        s /= size
        return s**0.5
    
    def smallcluster(numOfCluster):
        ac = AgglomerativeClustering(n_clusters=numOfCluster, affinity="euclidean", linkage="complete")
        labels = ac.fit_predict(short_dis)
        cluster_mean_status = [[0,0] for _ in range(max(labels)+1)] 
        for i in range(len(short_dis)):
            cluster_mean_status[labels[i]][0] += short_dis[i][0]
            cluster_mean_status[labels[i]][1] += 1
        for i in range(max(labels)+1):
            cluster_mean_status[i][0] /= cluster_mean_status[i][1]
        means = []
        for i in range(len(short_dis)):
            means.append(cluster_mean_status[labels[i]][0])
        stdard_devi = stdDev(short_dis, means)
        return stdard_devi
    
    short_dis = np.array([[shortestDistance(p),0] for p in Cluster])

    numOfCluster = 1
    stdard_devi = smallcluster(numOfCluster)
    print("표준편차 구하기 완료", numOfCluster, stdard_devi)
    
    while stdard_devi > dist_stdard_threshold:
        numOfCluster += 1
        stdard_devi = smallcluster(numOfCluster)
        print("표준편차 구하기 완료", numOfCluster, stdard_devi)
    
    print("종료")
    ac = AgglomerativeClustering(n_clusters=numOfCluster, affinity="euclidean", linkage="complete")
    labels = ac.fit_predict(short_dis)
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ap = np.array([[p.x, p.y, p.z] for p in Cluster])
    # ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=labels, marker='o', s=15, cmap='rainbow')
    # plt.show()
    
    
    clusterNow = len(NewClusterPointMap)
    for i in range(len(Cluster)):
        NewClusterPointMap[labels[i]+clusterNow].append(Cluster[i])
    
def divideAllCluster(clusterPointMap):
    for Cluster in clusterPointMap.values():
        divideCluster_stairmethod(Cluster)


#cluster는 한 클러스터 안에 있는 점들의 리스트임. 베스트 평면방정식을 찾아줌
def ransacPlane(Cluster):
    print("ransacPlane 가동")
    #점 3개지나는 평면의 방정식 abcd 튜플로 리턴
    def findPlane(p1, p2, p3):
        v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
        v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
        normal = np.cross(v12,v13)
        d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
        return normal[0], normal[1], normal[2], d

    #점 p랑 ax+by+cz+d=0 수직거리
    def sujikDistance(p, a,b,c,d):
        x = p.x
        y = p.y
        z = p.z
        res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
        return res
    
    pts = Cluster

    global ransacTrial2
    global ransacThreshold2
    numOfPoints = len(pts)
    maxScore = 0
    bestPlane = None
    for i in range(ransacTrial2):
        p1 = pts[random.randrange(0,numOfPoints)]
        p2 = pts[random.randrange(0,numOfPoints)]
        while p1.distance(p2) < nearbyLeastDistance:
            p2 = pts[random.randrange(0,numOfPoints)]
        p3 = pts[random.randrange(0,numOfPoints)]
        while p3.distance(p1) < nearbyLeastDistance or p3.distance(p2) < nearbyLeastDistance:
            p3 = pts[random.randrange(0,numOfPoints)]
        a, b, c, d = findPlane(p1, p2, p3)
        score = 0
        for p in pts:
            if sujikDistance(p, a, b, c, d) < ransacThreshold2:
                score +=1
        if score > maxScore:
            bestPlane = (a,b,c,d)
    return bestPlane


#NewClusterPointMap에서 ransac으로 실제 평면 방정식을 뽑아냄
def makeAllPlane():
    for Cluster in NewClusterPointMap.values():
        FinalPlanes.append(ransacPlane(Cluster))



#####################경계값 처리 ####################

def boundaryRemoveNoise():
    boundaryFriend = 10
    boundaryR = 2
    boundaryOutlierThreshold = 8
    size = len(BoundaryPoints)
    distMat = defaultdict(dict)
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (BoundaryPoints[i].distance(BoundaryPoints[j]),j) 

    for i in range(size):
        l = distMat[i]
        res = sorted(l.values(), key = lambda x: x[0])
        BoundaryPoints[i].nearby = res[1:boundaryFriend+1]

    #노이즈 제거
    del_candidate = [] #index들 리스트
    for i in range(size):
        if BoundaryPoints[i].nearby[boundaryOutlierThreshold-1][0] > boundaryR:
            del_candidate.append(i)
    
    for p in del_candidate:
        BoundaryPoints.remove(p)
    

#####################-실행코드-#####################

#removeNoise()

print(len(AllPoints))
    
# ap = np.array([[p.x, p.y, p.z] for p in AllPoints])
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ap[:,0], ap[:,1], ap[:,2], c=[0 for i in range(len(AllPoints))], marker='o', s=15, cmap='rainbow')
# plt.show()

#makeNormal()

print(len(BoundaryPoints))

# Allvec = np.array([p.normal.vector for p in AllPoints])
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(Allvec[:,0],Allvec[:,1],Allvec[:,2],c=[0 for i in range(len(Allvec))], marker='o', cmap='rainbow')
# plt.show()

#plt.ion()

#clusterPointMap = vectorClustering(2*howmanyclusters)
clusterPointMap = vectorClustering2(howmanyclusters)
#divideAllCluster(clusterPointMap)
NewClusterPointMap = clusterPointMap

NewAllPoints = []
for k in NewClusterPointMap.keys():
    NewAllPoints.extend(NewClusterPointMap[k])

# NewAllPoints.extend(BoundaryPoints)

NewLabels = []
for k in NewClusterPointMap.keys():
    NewLabels += [k] * len(NewClusterPointMap[k])
        
# NewLabels += [max(NewLabels) + 1] * len(BoundaryPoints)        


ap = np.array([[p.x, p.y, p.z] for p in NewAllPoints])
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ap[:,0], ap[:,1], ap[:,2], c=NewLabels, marker='o', s=15, cmap='rainbow')
plt.show()




#makeAllPlane()
#print(FinalPlanes) # 최종 평면들


