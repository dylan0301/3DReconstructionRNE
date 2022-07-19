#옛날방식이다. 느려서 버림


import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time



#############-hyperparameter-#################
friend = 10
r = 10
OutlierThreshold = 9
stdThreshold = 0.5



class Point:
    def __init__(self, X, Y, Z, idx, RGBA = None):
        self.x = X 
        self.y = Y 
        self.z = Z
        self.idx = idx
        self.rgba = RGBA
        self.nearby = list() #가까운 점들의 index가 들어감
        self.normal = None
        
    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z)
    
    def distance(self, p):
        def sqsbt(a,b):
            return (a-b)**2
        return ((sqsbt(self.x, p.x)+sqsbt(self.y, p.y)+sqsbt(self.z, p.z)))**(0.5)
    
    #다른 모든 점들과의 거리들을 크기순(오름차순)으로 모두 찾아준다. (self.nearby를 업데이트 해주는 함수)
    def updateNearby(self, distMat):
        l = distMat[self.idx]
        l.sort(key = lambda x : x[0])
        self.nearby = l[1:friend+1]
    

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
        for i in range(n):
            p1 = random.randrange(0,numOfPoints)
            p2 = random.randrange(0,numOfPoints)
            while p2 == p1:
                p2 = random.randrange(0,numOfPoints)
            p3 = random.randrange(0,numOfPoints)
            while p3 == p2 or p3 == p1:
                p3 = random.randrange(0,numOfPoints)
            p1 = pts[p1]
            p2 = pts[p2]
            p3 = pts[p3]
            a, b, c, d = findPlane(p1, p2, p3)
            score = 0
            for p in pts:
                if sujikDistance(p, a, b, c, d) < r/3:
                    score +=1
            if score > maxScore:
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
        
        normals = []
        plane_normal = self.ransacVector(10)
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                nor = vectors[i].cross(vectors[j])
                nor.normalize()
                nor.directionFix(plane_normal) #여기에 plane의 Normal vector 들어간다.
                normals.append(nor)
        
        avg = Vector(np.array([0,0,0]))
        for nor in normals:
            avg.vector = avg.vector + nor.vector
        avg.vector = avg.vector/len(normals)
        avg.normalize()
        
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
def generatePoints1(size):
    points = []
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
        points.append(p)
    for i in range(100):
        p = Point(100*random.random(), 100*random.random(), 100*random.random(), size-100+i)
        points.append(p)
    return points

def generatePoints2(size):
    points = []
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
        points.append(p)
    for i in range(100):
        p = Point(100*random.random(), 100*random.random(), 100*random.random(), size-100+i)
        points.append(p)
    return points

def generatePoints3(size):
    points = []
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
        points.append(p)
    for i in range(100):
        p = Point(100*random.random(), 100*random.random(), 100*random.random(), size-100+i)
        points.append(p)
    return points


random.seed(8)
AllPoints = generatePoints3(5000)
BoundaryPoints = []
CenterPoints = []

###################### -find normal vector-  ############################3



def removeNoise():
    t = time.time()
    global friend, r, OutlierThreshold
    
    size = len(AllPoints)
    distMat = [[0 for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            distMat[i][j] = (AllPoints[i].distance(AllPoints[j]),j) 

    for i in range(size):
        AllPoints[i].updateNearby(distMat)

    #노이즈 제거
    del_candidate = [] #index들 리스트
    for i in range(size):
        if AllPoints[i].nearby[OutlierThreshold-1][0] > r:
            del_candidate.append(AllPoints[i])

    for p in del_candidate:
        AllPoints.remove(p)
    print("리무브노이스 수행시간: ", time.time()-t)

def makeNearby():
    t = time.time()
    size = len(AllPoints)
    distMat = [[0 for j in range(size)] for i in range(size)]

    for i in range(size):
        for j in range(size):
            distMat[i][j] = (AllPoints[i].distance(AllPoints[j]),j)

    for i in range(size):
        AllPoints[i].idx = i
        AllPoints[i].updateNearby(distMat)
    print("메이크니어바이 수행시간: ", time.time()-t)

def makeNormal():
    t = time.time()
    size = len(AllPoints)
    for i in range(size):
        AllPoints[i].normalVectorize()
    print("메이크노멀 수행시간: ", time.time()-t)


####################### 클러스터링 시작 ##################33


def vectorClustering(numOfCluster):
    t = time.time()
    Duplicatedvectors = np.array([p.normal.vector for p in CenterPoints] + [(-1)*p.normal.vector for p in CenterPoints])
    ac = AgglomerativeClustering(n_clusters=numOfCluster, affinity="euclidean", linkage="complete")
    labels = ac.fit_predict(Duplicatedvectors)

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
    plt.show()
    return newLabel

#https://wikidocs.net/92111

    
    


#####################-실행코드-#####################

removeNoise()

print(len(AllPoints))
    
# ap = np.array([[p.x, p.y, p.z] for p in AllPoints])
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ap[:,0], ap[:,1], ap[:,2], c=[0 for i in range(len(AllPoints))], marker='o', s=15, cmap='rainbow')
# plt.show()

makeNearby()
makeNormal()

print(len(BoundaryPoints))

# Allvec = np.array([p.normal.vector for p in AllPoints])
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(Allvec[:,0],Allvec[:,1],Allvec[:,2],c=[0 for i in range(len(Allvec))], marker='o', cmap='rainbow')
# plt.show()

vectorClustering(8)


