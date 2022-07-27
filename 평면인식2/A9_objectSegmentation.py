#boundarypoint들을 위치를 기준으로 DBSCAN함.
#같은 클러스터로 분류된 boundarypoint들을 같은 물체로 분류
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN

def boundaryClustering(BoundaryPoints, hyperparameter):
    boundary_points_np = np.array([[p.x, p.y, p.z] for p in BoundaryPoints])
    clustering = DBSCAN(eps = hyperparameter.eps_point, min_samples = hyperparameter.min_samples_point)
    labels = clustering.fit_predict(boundary_points_np)
    BoundaryCluster = defaultdict(list)
    
    for i in range(len(BoundaryPoints)):
        BoundaryCluster[labels[i]].append(BoundaryPoints[i])
    
    return BoundaryCluster

def oneObjBoundary(ObjBoundaryCluster, NewAllPoints, NewLabel, hyperparameter):
    ObjectPlanes = []
    ObjectBoundary = defaultdict(list)
    size = max(NewLabel) + 1
    planeClusterConnectMap = defaultdict(dict())
    for i in range(size):
        for j in range(size):
            planeClusterConnectMap[i][j] = 0

    for p in ObjBoundaryCluster:
        planeNearP = []
        for i in range(len(NewAllPoints)):
            q = NewAllPoints[i]
            if p.distance(q) < hyperparameter.eps_point:
                if NewLabel[i] not in planeNearP:
                    planeNearP.append(NewLabel[i])
    
        if len(planeNearP) == 2:
            planeNearP.sort()
            planeClusterConnectMap[planeNearP[0]][planeNearP[1]] += 1
            ObjectBoundary[(planeNearP[0], planeNearP[1])].append(p)
           
    requiredConnectnum = 10
    for i in range(size):
        for j in range(size):
            if planeClusterConnectMap[i][j] / requiredConnectnum >= 1:
                planeClusterConnectMap[i][j] = 1
                if i not in ObjectPlanes:
                    ObjectPlanes.append(i)
                if j not in ObjectPlanes:
                    ObjectPlanes.append(j)
            else: 
                planeClusterConnectMap[i][j] = 0
    return planeClusterConnectMap, ObjectPlanes, ObjectBoundary

#input: graph, 2차원 defaultdict(dict)
#D = Object_Planes: list, D[i] = list of planes. (newCluster indexes로 표현됨.)

#output: graph를 잘 조작해서 구멍을 채워서 점을 생성한다.

def processGraph(graph, Object_Info, hyperparameter):
    #각 평면이 어느 오브젝트들에 들어있는지 표시
    Plane_Objects = defaultdict(list)
    Object_Planes, ObjectBoundary = Object_Info
    for obj in Object_Planes:
        for plane in obj:
            Plane_Objects[plane].append(obj)
    
    for plane in Plane_Objects.keys():
        if len(Plane_Objects[plane]) > 1:
            disconnect(graph, plane)
    
    def disconnect(graph, plane):
        #plane과 연결된 D_i
        contained_obj = Plane_Objects[plane]
        for D_i in contained_obj.values():
            holeFill(D_i, plane)
        
        def holeFill(D_i, plane):
            
            
            def isSpecial()
            pass           
    
def ObjectSegmentation(BoundaryPoints, NewClusterPointMap, NewAllPoints, NewLabel, hyperparameter):
    BoundaryCluster = boundaryClustering(BoundaryPoints, hyperparameter)
    Object_Info = []
    size = max(NewLabel) + 1
    Graph = defaultdict(dict)
    for i in range(size):
        for j in range(size):
            Graph[i][j] = 0 
    for k in range(len(BoundaryCluster)):
        planeClusterConnectMap, ObjectPlanes, ObjectBoundary = oneObjBoundary(BoundaryCluster[k], NewAllPoints, NewLabel, hyperparameter)
        for i in range(size):
            for j in range(size):
                Graph[i][j] = Graph[i][j] or planeClusterConnectMap[i][j]
        Object_Info.append((ObjectPlanes, ObjectBoundary))
    
    NewGraph = processGraph(Graph, Object_Info, hyperparameter)
    
            
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