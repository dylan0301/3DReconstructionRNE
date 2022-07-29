from A1_classes import Point, Hyperparameter
from A2_1_importPointCloud import read_ply_xyzrgb
from collections import defaultdict
import random
import numpy as np

#현재 hyperparameter 설정이 덜됐음


def importPly(filepath, filename):
    hyperparameter = Hyperparameter()
    
    rawPoints = read_ply_xyzrgb(filepath+filename)
    sortedPoints = sorted(rawPoints, key = lambda x: (x[0],x[1],x[2]))

    points = defaultdict(Point)
    numOfPoints = 0
    pointLeastDifference = 0.00001

    #중복점제거하면서 포인트 추가
    for i in range(len(sortedPoints)-1):
        if sortedPoints[i+1][0] - sortedPoints[i][0] < pointLeastDifference:
            if sortedPoints[i+1][1] - sortedPoints[i][1] < pointLeastDifference:
                if sortedPoints[i+1][2] - sortedPoints[i][2] < pointLeastDifference:
                    continue
        
        x = sortedPoints[i][0]
        y = sortedPoints[i][1]
        z = sortedPoints[i][2]

        if filename == 'Box25K.ply':
            if x > 0.45 or x < -0.2 or y > -0.2:
                continue
        if filename == 'movable_desk.ply':
            if x < -0.8 or y > -0.5 or z < -3.5:
                continue
        if filename == 'highBox.ply':
            if x < -0.5 or x > 0.5 or z < -0.55 or z > -0.12:
                continue
        if filename == 'twoBooksAndBox.ply':
            if y > -0.6 or y < -1 or (x < -0.3 and y < -0.96) or (x<0.2 and y<-0.98) or z < -1.5 or z > -0.55 or x < -0.8 or x > 0.5:
                continue
            
        if filename == 'twoBoxes.ply':
            if x < -0.5 or x > 0.25 or z > -0.25 or z < -1.35:
                continue

        p = Point(x, y, z,
                numOfPoints, sortedPoints[i][3], sortedPoints[i][4], sortedPoints[i][5]) #단위 m
        points[numOfPoints] = p
        numOfPoints += 1
    # p = Point(sortedPoints[-1][0], sortedPoints[-1][1], sortedPoints[-1][2],
    #             numOfPoints, sortedPoints[-1][3], sortedPoints[-1][4], sortedPoints[-1][5])
    # points[numOfPoints] = p #마지막점 추가해주기
    
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
    random.seed(0)
    points = defaultdict(Point)
    hyperparameter = Hyperparameter(numOfPoints = 3000,
    R1 = 5, OutlierThreshold1 = 35, H1 = 0.5, ratioThreshold1 = 0.7,
    eps_normal = 0.1, min_samples_normal = 9,
    eps_centerPoint = 4, min_samples_centerPoint = 8,
    eps_finalBoundaryPoint = 5, min_samples_finalBoundaryPoint = 5
    )

    size = hyperparameter.numOfPoints
    for i in range(size):
        diff = 0.2*(random.random()-0.5)
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

def Octahedron():
    random.seed(8)
    points = defaultdict(Point)
    
    hyperparameter = Hyperparameter()

    size = hyperparameter.numOfPoints
    for i in range(size):
        diff = 0.5*(random.random()-0.5)
        x = 60*(random.random()-0.5)
        y = 60*(random.random()-0.5)
        z = random.choice([-1,1])*(30-abs(x)-abs(y)) + diff
        while abs(x) + abs(y) > 30:
            diff = 2*random.random()-1
            x = 60*(random.random()-0.5)
            y = 60*(random.random()-0.5)
            z = random.choice([-1,1])*(30-abs(x)-abs(y)) + diff
        p = Point(x,y,z,i)
        points[i] = p
    return points, hyperparameter

def Sphere():
    import math as mt
    random.seed(8)
    points = defaultdict(Point)

    hyperparameter = Hyperparameter()

    size = hyperparameter.numOfPoints
    r = 30

    for i in range(size):
        theta = 4*mt.pi*(random.random()-0.5)
        phi =  2*mt.pi*(random.random()-0.5)
        x = r * mt.cos(phi) * mt.cos(theta)
        y = r * mt.cos(phi) * mt.sin(theta)
        z = r * mt.sin(phi)
        p = Point(x, y, z, i)
        points[i] = p
    return points, hyperparameter




def butterfly(R = 20, alpha = np.pi/3, size=2000):
    random.seed(0)
    points = defaultdict(Point)

    hyperparameter = Hyperparameter(numOfPoints=size)

    for i in range(size):
        if i % 2 == 0:
            x = 2*R*(random.random()-0.5)
            y = -np.sqrt(R**2-x**2)*random.random()
            z = 0
        if i % 2 == 1:
            x = 2*R*(random.random()-0.5)
            y = np.sqrt((R**2-x**2)*np.cos(alpha)**2)*random.random()
            z = np.tan(alpha)*y
        p = Point(x, y, z, i)
        points[i] = p
    return points, hyperparameter


def bang_simple():
    import math as mt
    random.seed(8)
    points = defaultdict(Point)

    hyperparameter = Hyperparameter(pointLeastDifference = 0.001, numOfPoints = 5000,
    OutlierThreshold = 10, R = 7, vectorRansacTrial = 100, vectorRansacThreshold = 0.4, normalLeastNorm = 0.001,
    ratioThreshold = 0.6, eps_vector = 0.1, min_samples_vector = 9,
    eps_point = 4, min_samples_point = 10, planeRansacTrial = 50, planeRansacThreshold = 0.15,
    boundaryR = 4, boundaryOutlierThreshold = 9)

    size = hyperparameter.numOfPoints
    
    r = 30

    for i in range(size):
        diff = 0.5*(random.random()-0.5)
        if i < 3500:
            x = r * random.random()
            y = r * random.random()
            z = 0
            if 10 <= x and x <= 20 and 10 <= y and y <= 20:
                z = 10
            if 13 <= x and x <= 16 and 13 <= y and y <= 16:
                z = 13
        elif i < 4800:
            if i % 4 == 0:
                x = 10 + 10 * random.random()
                y = 10
                z = 10 * random.random()
            elif i % 4 == 1:
                x = 10 
                y = 10 + 10 * random.random()
                z = 10 * random.random()
            elif i % 4 == 2:
                x = 10 + 10 * random.random()
                y = 20
                z = 10 * random.random()
            elif i % 4 == 3:
                x = 20
                y = 10 + 10 * random.random()
                z = 10 * random.random()
        else:
            if i % 4 == 0:
                x = 13 + 3 * random.random()
                y = 13
                z = 10 + 3 * random.random()
            elif i % 4 == 1:
                x = 13
                y = 13 + 3 * random.random()
                z = 10 + 3 * random.random()
            elif i % 4 == 2:
                x = 13 + 3 * random.random()
                y = 16
                z = 10 + 3 * random.random()
            elif i % 4 == 3:
                x = 16 
                y = 13 + 3 * random.random()
                z = 10 + 3 * random.random()  
        z += diff
        p = Point(x, y, z, i)
        points[i] = p
    return points, hyperparameter