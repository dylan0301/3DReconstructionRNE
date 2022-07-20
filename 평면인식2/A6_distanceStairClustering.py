import numpy as np

def divideCluster_stairmethod(Cluster, NewClusterPointMap, NewLabels,  hyperparameter):
    avg = np.array([0,0,0])
    for p in Cluster:
        avg = avg + p.normal
    avg.normalize()
    print(avg)

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
    labels = [None] * len(Cluster)
    step_threshold = hyperparameter.step_threshold
    
    
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
        NewLabels.append(labels[i]+clusterNow)
    