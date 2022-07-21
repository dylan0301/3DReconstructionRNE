import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN


def divideCluster_stairmethod(Cluster, NewClusterPointMap, hyperparameter):
    Newclusterpm = NewClusterPointMap
    avg = np.array([0,0,0])
    for p in Cluster:
        avg = avg + p.normal
    avg /= np.linalg.norm(avg)
    print('Avg Before:', avg)

    #점 p랑 ax+by+cz+d=0 수직거리. a,b,c는 avg벡터고 d=0
    def shortestDistance(p):
        x = p.x
        y = p.y
        z = p.z
        a = avg[0]
        b = avg[1]
        c = avg[2]
        d = 0
        res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
        return res

    short_dis = [(shortestDistance(p),p) for p in Cluster]
    short_dis.sort(key = lambda x: x[0])
    #print(short_dis)
    labels = [None] * len(Cluster)
    
    
    label = 0
    for i in range(len(short_dis)):
        if i == 0: labels[Cluster.index(short_dis[i][1])] = label
        else:
            stepGap = abs(short_dis[i][0] - short_dis[i-1][0])
            if stepGap > hyperparameter.step_threshold:
                label += 1
                labels[Cluster.index(short_dis[i][1])] = label
            else:
                labels[Cluster.index(short_dis[i][1])] = label        
           
    clusterNow = len(Newclusterpm)
    for i in range(len(Cluster)):
        Newclusterpm[labels[i]+clusterNow].append(Cluster[i]) 
    #print(Newclusterpm)
    return Newclusterpm
    

def divideAllCluster_stair(NewClusterPointMap, clusterPointMap, hyperparameter):
    for Cluster in clusterPointMap.values():
        temp = divideCluster_stairmethod(Cluster, NewClusterPointMap, hyperparameter)
        NewClusterPointMap = temp
    return NewClusterPointMap




def divideCluster_DBSCAN(cluster, NewClusterPointMap, hyperparameter):
    Newclusterpm = NewClusterPointMap
    cluster_Points = np.array([[p.x, p.y, p.z] for p in cluster])
    clustering = DBSCAN(eps = hyperparameter.eps_point, min_samples = hyperparameter.min_samples_point)
    labels = clustering.fit_predict(cluster_Points)

    clusterNow = len(Newclusterpm)
    for i in range(len(cluster)):
        Newclusterpm[labels[i]+clusterNow].append(cluster[i]) 

    return Newclusterpm




def divideAllCluster_DBSCAN(NewClusterPointMap, clusterPointMap, hyperparameter):
    for Cluster in clusterPointMap.values():
        temp = divideCluster_DBSCAN(Cluster, NewClusterPointMap, hyperparameter)
        NewClusterPointMap = temp
    return NewClusterPointMap