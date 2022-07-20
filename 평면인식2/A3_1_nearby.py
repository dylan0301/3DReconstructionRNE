from collections import defaultdict

def updateNearby(point, distMat, del_set = set()):
    l = distMat[point.idx]
    if del_set == set():
        sorteddisMat = sorted(l.values(), key=lambda x: x[0])
    else:
        sorteddisMat = l.values()
    res = []
    for i in sorteddisMat:
        if i[1] not in del_set and point != i[1]:
            res.append(i)
    
    point.nearby = res #nearby에는 (dist, point) 가 들어있음
    res = dict()
    for i in point.nearby:
        res[i[1].idx] = i
    return res
