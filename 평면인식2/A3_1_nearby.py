from collections import defaultdict

#실제 데이터는 del candidate가 없어서 비는 문제 발생해서 noisy 넣음
def updateNearby(point, distMat, noisy, hyperparameter = None, del_set = set()):
    l = distMat[point.idx]
    if noisy:
        sorteddisMat = sorted(l.values(), key=lambda x: x[0])
    else:
        sorteddisMat = l.values()
    res = []
    if noisy:
        for i in sorteddisMat:
            if i[1] not in del_set and point != i[1]:
                res.append(i)
    else:
        for i in sorteddisMat:
            if i[1] not in del_set and point != i[1]:
                res.append(i)
            if len(res) >= hyperparameter.friend:
                break

    point.nearby = res #nearby에는 (dist, point) 가 들어있음
    res = dict()
    for i in point.nearby:
        res[i[1].idx] = i
    return res
