from A1_classes import Point, Hyperparameter
from A2_1_importPointCloud import read_ply_xyzrgb
from collections import defaultdict
import random

#현재 hyperparameter 설정이 덜됐음


def importPly(absoluteFile):
    hyperparameter = Hyperparameter()
    
    rawPoints = read_ply_xyzrgb(absoluteFile)
    sortedPoints = sorted(rawPoints, key = lambda x: (x[0],x[1],x[2]))

    points = defaultdict(Point)
    numOfPoints = 0

    #중복점제거하면서 포인트 추가
    for i in range(len(sortedPoints)-1):
        if sortedPoints[i+1][0] - sortedPoints[i][0] < hyperparameter.pointLeastDifference:
            if sortedPoints[i+1][1] - sortedPoints[i][1] < hyperparameter.pointLeastDifference:
                if sortedPoints[i+1][2] - sortedPoints[i][2] < hyperparameter.pointLeastDifference:
                    continue
        if absoluteFile == '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/Box25K.ply':
            if sortedPoints[i][0] > 0.45 or sortedPoints[i][0] < -0.2 or sortedPoints[i][1] > -0.2:
                continue
        p = Point(sortedPoints[i][0], sortedPoints[i][1], sortedPoints[i][2],
                numOfPoints, sortedPoints[i][3], sortedPoints[i][4], sortedPoints[i][5]) #단위 m
        points[numOfPoints] = p
        numOfPoints += 1
    p = Point(sortedPoints[-1][0], sortedPoints[-1][1], sortedPoints[-1][2],
                numOfPoints, sortedPoints[-1][3], sortedPoints[-1][4], sortedPoints[-1][5])
    points[numOfPoints] = p #마지막점 추가해주기
    
    hyperparameter.numOfPoints = numOfPoints

    return points, hyperparameter


    

def _test_importPly():
    filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
    filename = 'Box25K.ply'
    AllPoints, hyperparameter = importPly(filepath+filename)
    print(len(AllPoints))
    for p in AllPoints.values():
        print(p)

if __name__ == "__main__":
    _test_importPly()

#100*100*100 큐브의 반쪽, clean
def halfCubeClean(size):
    random.seed(0)
    points = defaultdict(Point)
    hyperparameter = Hyperparameter(



    )
    size = hyperparameter.numOfPoints
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
    return points, hyperparameter


#100*100*100 직각삼각뿔, clean
def triPyramidClean():
    random.seed(0)
    points = defaultdict(Point)
    
    hyperparameter = Hyperparameter(

    )
    size = hyperparameter.numOfPoints
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
    return points, hyperparameter


#30*30*30 정육면체, clean
def cubeClean():
    random.seed(0)
    points = defaultdict(Point)
    hyperparameter = Hyperparameter()
    size = hyperparameter.numOfPoints
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
    return points, hyperparameter



#30*30*30 정육면체, dirty
def cubeDirty():
    random.seed(0)
    points = defaultdict(Point)
    hyperparameter = Hyperparameter()
    size = hyperparameter.numOfPoints
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
    return points, hyperparameter



    
def unicorn_sample():
    random.seed(132808)
    points = defaultdict(Point)
    hyperparameter = Hyperparameter()

    size = hyperparameter.numOfPoints
    for i in range(size):
        diff = 2*random.random()-1
        if i%12 == 0:
            x = 30*random.random()
            y = 30*random.random()
            z = diff
        if i%12 == 1:
            x = 30*random.random()
            y = diff
            z = 30*random.random()
        if i%12 == 2:
            x = diff
            y = 30*random.random()
            z = 30*random.random()
        if i%12 == 3:
            x = 30*random.random()
            y = 30*random.random()
            z = 30+diff
        if i%12 == 4:
            x = 30*random.random()
            y = 30+diff
            z = 30*random.random()
        if i%12 == 5:
            x = 30+diff
            y = 30*random.random()
            z = 30*random.random()   
        if i%12 == 6:
            x = 30*random.random() + 40
            y = diff
            z = 30*random.random()
        if i%12 == 7:
            x = diff + 40
            y = 30*random.random()
            z = 30*random.random()
        if i%12 == 8:
            x = 30*random.random() + 40
            y = 30*random.random()
            z = 30+diff
        if i%12 == 9:
            x = 30*random.random() + 40
            y = 30+diff
            z = 30*random.random()
        if i%12 == 10:
            x = 30+diff + 40
            y = 30*random.random()
            z = 30*random.random()  
        if i%12 == 11:
            x = 30*random.random() + 40
            y = 30*random.random()
            z = diff    
        p = Point(x,y,z,i)
        points[i] = p
    return points, hyperparameter

def unicorn_sample2():
    random.seed(132808)
    points = defaultdict(Point)
    hyperparameter = Hyperparameter(pointLeastDifference = 0.001, numOfPoints = 3000,
    OutlierThreshold = 10, R = 5, vectorRansacTrial = 50, vectorRansacThreshold = 0.15,
    normalLeastNorm = 0.001, ransacErrorThreshold = 1, eps_vector = 0.1, min_samples_vector = 9,
    eps_point = 4, min_samples_point = 10, planeRansacTrial = 50, planeRansacThreshold = 1,
    boundaryR = 4, boundaryOutlierThreshold = 10)

    size = hyperparameter.numOfPoints
    for i in range(size):
        diff = 2*random.random()-1
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
            x = 30*random.random() + 40
            y = 30*random.random()
            z = diff
        if i%6 == 4:
            x = 30*random.random() + 40
            y = diff
            z = 30*random.random()
        if i%6 == 5:
            x = diff + 40
            y = 30*random.random()
            z = 30*random.random()   
        p = Point(x,y,z,i)
        points[i] = p
    return points, hyperparameter