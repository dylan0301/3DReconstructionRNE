#boundarypoint들을 위치를 기준으로 DBSCAN함.
#같은 클러스터로 분류된 boundarypoint들을 같은 물체로 분류
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from A1_classes import *
import random

def boundaryClustering(BoundaryPoints, hyperparameter):
    boundary_points_np = np.array([[p.x, p.y, p.z] for p in BoundaryPoints])
    clustering = DBSCAN(eps = hyperparameter.eps_point2, min_samples = hyperparameter.min_samples_point2)
    labels = clustering.fit_predict(boundary_points_np)
    BoundaryCluster = defaultdict(list())
    objList = []
    
    for i in range(len(BoundaryPoints)):
        BoundaryCluster[labels[i]].append(BoundaryPoints[i])
    
    for i, points in BoundaryCluster.item():
        obj = Object(i, points)
        objList.append(obj)
            
    return objList

def findOneObjPlanes(obj, planeList, hyperparameter):
    numOfPlanes = len(planeList)

    for p in obj.BoundaryPoints:
        planeNearP = []
        for i in range(numOfPlanes):
            for q in planeList[i].interiorPoints:
                if p.distance(q) < hyperparameter.eps_point:
                    if planeList[i] not in planeNearP:
                        planeNearP.append(planeList[i])
    
        if len(planeNearP) == 2:
            planeNearP[0].connected[planeNearP[1]].append(p)
            planeNearP[1].connected[planeNearP[0]].append(p)
            planeNearP[0].containedObj.add(obj)
            planeNearP[1].containedObj.add(obj)
            obj.planes.add(planeNearP[0])
            obj.planes.add(planeNearP[1])
        if len(planeNearP) >= 2:
            pass


            

#input: graph, 2차원 defaultdict(dict)
#D = Object_Planes: list, D[i] = list of planes. (newCluster indexes로 표현됨.)

#output: graph를 잘 조작해서 구멍을 채워서 점을 생성한다.

def processGraph(planeList, hyperparameter):
    #각 평면이 어느 오브젝트들에 들어있는지 표시
    
    for plane in planeList:
        if len(plane.containedObj) > 1:
            disconnect(plane)
    
    def disconnect(plane):
        #plane과 연결된 obj들에 대해서
        for obj in plane.containedObj:
            holeFill(plane, obj)
        
        def holeFill(plane, obj):
            #edgePoints 주어지면 그거 이루는 ransac 직선 나옴
            def edgeRansac(edgePoints):
                def findLine(p1, p2):
                    direction = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
                    return (direction, p2)
                
                #p에서 line까지 거리
                def pointLineDistance(p, direction, p2):
                    PA = np.array([p.x-p2.x, p.y-p2.y, p.z-p2.z])
                    res = np.linalg.norm(np.cross(PA, direction))/np.linalg.norm(direction)
                    return res

                pts = edgePoints

                numOfpts = len(pts)
                maxScore = 0
                bestLine = None
                for trial in range(hyperparameter.edgeRansacTrial):
                    line = None
                    while line == None:
                        i1 = random.randrange(0,numOfpts)
                        i2 = random.randrange(0,numOfpts)
                        while i1 == i2:
                            i2 = random.randrange(0,numOfpts)
                        line = findLine(pts[i1], pts[i2])
                    score = 0
                    for p in pts:
                        d = pointLineDistance(p, line)
                        if d < hyperparameter.edgeRansacThreshold:
                            score +=1
                    if score > maxScore:
                        maxScore = score
                        bestLine = line
                return bestLine[0], bestLine[1] #방향벡터, 그위의 점   
            
            projectedLines = []

            for plane2 in obj:
                if plane2 not in plane.connected.keys():
                    continue
                edgePoints = plane.connected[plane2] # Boundary btwn plane and plane2
                edgeVector, startPoint = edgeRansac(edgePoints)
 
def ObjectSegmentation(BoundaryPoints, planeList, hyperparameter):
    objList = boundaryClustering(BoundaryPoints, hyperparameter)
    for i in range(len(objList)):
        findOneObjPlanes(objList[i], planeList, hyperparameter)

    processGraph(planeList, hyperparameter)

#각각의 평면 클러스터를 버텍스로 하고, 그 연결 관계를 edge로 하는 그래프를 그린다
#만약 버텍스(v)가 여러 D_i에 속한다면 v를 제거한다. 얘네가 바닥이나 벽, 책상
#v를 원소로 가지는 모든 D_i들에 대해서
    #D_i의 원소들 중 v와 adjacent했던 클러스터들에 대해서
        #구멍을 메꾸는 평면 cluster vertex u를 생성하고 u와 각각의 클러스터들을 이어 준다.
        #구멍 메꾸는법:
            #v제거할때 v와 연결된 edge들이 구멍을 이루는 boundarypoints다. 
            #한 edge에 속한 boundarypoints들은 직선 하나를 이룬다. 이거 RANSAC으로 찾음.
            #v 평면에 그 직선을 projection 시킴
            #이렇게 모두 projection시키면 v평면에 내부 구역이 생긴다. 그게 구멍임.
            #점으로 메꾼다.

            #아래 상자 찾는법:
                #v가 속한 각 D_i에 대해서 centerpoint들이 D_i boundary 내부에 있는지 본다.
                #구멍들은 D_i boundary 내부에 centerpoint 거의 없을거고,
                #아래 상자는 거의다 있을거다.

#이러면 이제 component들이 object가 될 예정

#문제점: 물체가 하나밖에 없이 바닥 위에 놓여있다면 제거가 안된다.
#ㄴ근데 그럴일은 없음. 물체 여러개다.

#문제점: 물체끼리 겹쳐 있는 경우 그 물체끼리 구분하기 아직 어려움. -답이 없음-

#몇번 바운더리 클러스터에 어떤 바운더리 점들 속하는지 필요
#(평면A, 평면B) -> 연결 가능성
#(평면A, 평면B) -> 연결되어있다면 거기 속하는 boundary 점들
#boundary cluster -> 거기 속하는 평면
#평면 -> boundary cluster
#평면들의 그래프