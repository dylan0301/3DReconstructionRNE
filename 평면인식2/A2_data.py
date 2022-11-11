from A1_classes import Point, Hyperparameter
from A2_1_importPointCloud import read_ply_xyzrgb, read_txt_xyz
from collections import defaultdict
import random
import numpy as np

#현재 hyperparameter 설정이 덜됐음

#density = 점사이 간격
def fillRect(p1,p4, points, density): #p1이 제일 원점에 가깝고 p4가 멀다
    i = 0
    size = len(points)
    if p1.x == p4.x:
        Iteration = [np.arange(p1.y, p4.y, density), np.arange(p1.z, p4.z, density)]
        for y in Iteration[0]:
            for z in Iteration[1]:
                p = Point(p1.x, y, z, size + i)
                points[size + i] = p
                i += 1
    elif p1.y == p4.y:
        Iteration = [np.arange(p1.x, p4.x, density), np.arange(p1.z, p4.z, density)]
        for x in Iteration[0]:
            for z in Iteration[1]:
                p = Point(x, p1.y, z, size + i)
                points[size + i] = p
                i += 1
    elif p1.z == p4.z:
        Iteration = [np.arange(p1.x, p4.x, density), np.arange(p1.y, p4.y, density)]
        for x in Iteration[0]:
            for y in Iteration[1]:
                p = Point(x, y, p1.z, size + i)
                points[size + i] = p
                i += 1
    return points

def importTxt(filepath, filename):
    hyperparameter = Hyperparameter()
    if filename == 'Cuboid.txt':
        hyperparameter = Hyperparameter(10, 1, 0.75, 0.07, 200, 10, 20, 10, 6, 1, 2)
    if filename == 'Curtin314.txt':
        hyperparameter = Hyperparameter(0.5, 0.05, 0.6, 0.03, 300, 4, 20, 5, 25, 1, 2)
    if filename == 'House.txt':
        hyperparameter = Hyperparameter(0.02, 0.002, 0.75, 0.03, 20, 0.04, 20, 0.05, 25, 0.001, 0.002)


    rawPoints = read_txt_xyz(filepath+filename)
    sortedPoints = sorted(rawPoints, key = lambda x: (x[0],x[1],x[2]))

    points = defaultdict(Point)
    numOfPoints = 0
    pointLeastDifference = 0.00001
    

    #중복점제거하면서 포인트 추가
    for i in range(1, len(sortedPoints)):
        if sortedPoints[i][0] - sortedPoints[i-1][0] < pointLeastDifference:
            if sortedPoints[i][1] - sortedPoints[i-1][1] < pointLeastDifference:
                if sortedPoints[i][2] - sortedPoints[i-1][2] < pointLeastDifference:
                    continue
        
        x = sortedPoints[i][0]
        y = sortedPoints[i][1]
        z = sortedPoints[i][2]


        p = Point(x, y, z, numOfPoints)
        points[numOfPoints] = p
        numOfPoints += 1

    return points, hyperparameter, filename


    

def _test_importPly():
    filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
    filename = 'Box25K.ply'
    AllPoints, hyperparameter, filename = importPly(filepath+filename)
    print(len(AllPoints))
    for p in AllPoints.values():
        print(p)



    

def importPly(filepath, filename):
    hyperparameter = Hyperparameter()
    
    rawPoints = read_ply_xyzrgb(filepath+filename)
    sortedPoints = sorted(rawPoints, key = lambda x: (x[0],x[2],x[1]))

    points = defaultdict(Point)
    numOfPoints = 0
    pointLeastDifference = 0.0001 #final1에서 바뀜

    floorPointIndex = [] #바닥에 있는 점 일괄적으로 만들거임
    floorLevel = 0

    #중복점제거하면서 포인트 추가
    for i in range(1, len(sortedPoints)):
        if sortedPoints[i][0] - sortedPoints[i-1][0] < pointLeastDifference:
            if sortedPoints[i][2] - sortedPoints[i-1][2] < pointLeastDifference:
                if sortedPoints[i][1] - sortedPoints[i-1][1] < pointLeastDifference:
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
            if x < -0.5 or x > 0.3 or z > -0.25 or z < -1.35:
                continue

        if filename == '3boxes.ply':
            if x < -0.45 or x > 0.57 or z > -0.4 or z < -1.45 or (x<0.2 and z<-1.1) or (x>0.1 and z>-0.8) or y <-1.01 or y>-0.8:
                continue

        if filename == 'twobox1.ply':
            hyperparameter = Hyperparameter(0.05, 0.005, 0.75, 0.07, 100, 0.1, 30, 0.1, 20, 0.003, 0.01)
            if x < -0.47 or x > 0.3 or z > -0.55 or z < -1.05 or y<-1.05 or y>-0.35 or (y > -0.75 and x < -0.4 and z<-0.75) or (y > -0.55 and x>0.175) or (x < -0.26 and y> -0.564):
                continue

        if filename == 'final1.ply':
            hyperparameter = Hyperparameter(0.04, 0.004, 0.7, 0.1, 300, 0.05, 20, 0.008, 10, 0.003, 0.001)
            if z<-2.05 or x>0.7 or x<-0.6 or z>-0.45 or (z > -0.95 and  y<-0.86) or (x>0.1 and z > -1) or (x<-0.2 and z<-1.4):
                continue
            if y<-0.75:
                y = -0.75

        if filename == 'realfinal1.ply':
            hyperparameter = Hyperparameter(0.035, 0.0035, 0.8, 0.1, 300, 0.05, 20, 0.008, 10, 0.003, 0.001)
            floorLevel = -0.9
            if x < -0.63 or x > 0.3 or z > -0.71 or z < -2.5 or y<-1.05 or (x > 0.2 and z > -1.2) or  (x < -0.43 and z > -1.3):
                 continue
            if x < -0.435 and z<-1.5 and y >-0.66:
                y = -0.66 
            if x<-0.55 and z < -1.5 and y>-0.9:
                x = -0.55
            if y<floorLevel:
                floorPointIndex.append(i)
                continue
                

        if filename == 'realfinal2.ply':
            hyperparameter = Hyperparameter(0.04, 0.004, 0.7, 0.1, 300, 0.05, 20, 0.008, 10, 0.003, 0.001)
            if x > 0.7 or x < -0.2 or z < -1.55 or z > -0.25 or y<-1.2 :
                continue

        p = Point(x, y, z,
                numOfPoints, sortedPoints[i][3], sortedPoints[i][4], sortedPoints[i][5]) #단위 m
        points[numOfPoints] = p
        numOfPoints += 1

    floorPointLeastDifference = 0.001

    for i in range(1, len(floorPointIndex)):
        if sortedPoints[floorPointIndex[i]][0] - sortedPoints[floorPointIndex[i-1]][0] < floorPointLeastDifference:
            if sortedPoints[floorPointIndex[i]][2] - sortedPoints[floorPointIndex[i-1]][2] < floorPointLeastDifference:
                continue
        x = sortedPoints[floorPointIndex[i]][0]
        y = floorLevel
        z = sortedPoints[floorPointIndex[i]][2]
        p = Point(x, y, z,
                numOfPoints, sortedPoints[floorPointIndex[i]][3], sortedPoints[floorPointIndex[i]][4], sortedPoints[floorPointIndex[i]][5]) #단위 m
        points[numOfPoints] = p
        numOfPoints += 1
    return points, hyperparameter, filename


    

def _test_importPly():
    filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
    filename = 'Box25K.ply'
    AllPoints, hyperparameter, filename = importPly(filepath+filename)
    print(len(AllPoints))
    for p in AllPoints.values():
        print(p)

if __name__ == "__main__":
    _test_importPly()



def butterfly(R = 20, alpha = np.pi/3, size=3000):
    random.seed(0)
    points = defaultdict(Point)

    hyperparameter = Hyperparameter()

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


def butterfly_uniform(R = 20, alpha = np.pi/3, lineardensity = 0.1):
    points = defaultdict(Point)

    hyperparameter = Hyperparameter(lineardensity=lineardensity)

    cnt = 0  
    z = 0
    for x in np.arange(-R, R, lineardensity):
        for y in np.arange(-R, 0, lineardensity):
            if x**2 + y**2 > R**2:
                continue   
            p = Point(x, y, z, cnt)
            points[cnt] = p
            cnt += 1

    if abs(alpha-np.pi/2) > 0.5:
        for x in np.arange(-R, R, lineardensity):
            for y in np.arange(0, R*np.cos(alpha), lineardensity):
                z = np.tan(alpha)*y
                if x**2 + y**2 + z**2 > R**2:
                    continue   
                p = Point(x, y, z, cnt)
                points[cnt] = p
                cnt += 1
    else:
        for x in np.arange(-R, R, lineardensity):
            for z in np.arange(0, R*np.sin(alpha), lineardensity):
                y = z/np.tan(alpha)
                if x**2 + y**2 + z**2 > R**2:
                    continue   
                p = Point(x, y, z, cnt)
                points[cnt] = p
                cnt += 1
    return points, hyperparameter




def bang_muchsimple():
    random.seed(0)
    points = defaultdict(Point)
    name = 'bang_muchsimple'

    hyperparameter = Hyperparameter(
    R1 = 3, H1 = 0.4, ratioThreshold1 = 0.7,
    eps_normal = 0.1, min_samples_normal = 9,
    eps_centerPoint = 3, min_samples_centerPoint = 30,
    eps_finalBoundaryPoint = 2, min_samples_finalBoundaryPoint = 10,
    edgeRansacH = 0.3)
    
    r = 30
    cnt = -1    
    for i in range(50):
        for j in range(50):
            diff = 0.5*(random.random()-0.5)
            cnt += 1
            x = r * i/50
            y = r * j/50
            z = 0
            if 11 <= x and x <= 20 and 11 <= y and y <= 20:
                z = 9
            p = Point(x, y, z+diff, cnt)
            points[cnt] = p
    
    for i in range(900):
        diff = 0.5*(random.random()-0.5)
        cnt += 1
        if i < 225:
            x = 11 + (i % 15)*3/5
            y = 11
            z = (i // 15)*3/5
        elif i < 450:
            x = 11 
            y = 11 + ((i-225) % 15)*3/5
            z = ((i-225) // 15)*3/5
        elif i < 675:
            x = 11 + ((i-450) % 15)*3/5
            y = 20
            z = ((i-450) // 15)*3/5
        else:
            x = 20
            y = 11 + ((i-675) % 15)*3/5
            z = ((i-675) // 15)*3/5
        p = Point(x, y, z+diff, cnt)
        points[cnt] = p
    
    for i in range(50):
        for j in range(50):
            diff = 0.5*(random.random()-0.5)
            cnt += 1
            x = r * i/50
            y = 0
            z = r * j/50
            p = Point(x, y+diff, z, cnt)
            points[cnt] = p
            
    for i in range(50):
        for j in range(50):
            diff = 0.5*(random.random()-0.5)
            cnt += 1
            x = 0
            y = r * i/50
            z = r * j/50
            p = Point(x+diff, y, z, cnt)
            points[cnt] = p

    return points, hyperparameter, name

def bang_verysimple():
    points = defaultdict(Point)
    name = 'bang_verysimple'

    hyperparameter = Hyperparameter(
    R1 = 2, H1 = 0.2, ratioThreshold1 = 0.7,
    eps_normal = 0.1, min_samples_normal = 9,
    eps_centerPoint = 4, min_samples_centerPoint = 30,
    eps_finalBoundaryPoint = 3, min_samples_finalBoundaryPoint = 10,
    edgeRansacH = 0.1)
    
    r = 30
    cnt = -1    
    for i in range(40):
        for j in range(40):
            cnt += 1
            x = r * i/40
            y = r * j/40
            z = 0
            if 11 <= x and x <= 20 and 11 <= y and y <= 20:
                z = 9
            p = Point(x, y, z, cnt)
            points[cnt] = p
    
    for i in range(900):
        cnt += 1
        if i < 225:
            x = 11 + (i % 15)*3/5
            y = 11
            z = (i // 15)*3/5
        elif i < 450:
            x = 11 
            y = 11 + ((i-225) % 15)*3/5
            z = ((i-225) // 15)*3/5
        elif i < 675:
            x = 11 + ((i-450) % 15)*3/5
            y = 20
            z = ((i-450) // 15)*3/5
        else:
            x = 20
            y = 11 + ((i-675) % 15)*3/5
            z = ((i-675) // 15)*3/5
        p = Point(x, y, z, cnt)
        points[cnt] = p
    
    for i in range(40):
        for j in range(40):
            cnt += 1
            x = r * i/40
            y = 0
            z = r * j/40
            p = Point(x, y, z, cnt)
            points[cnt] = p
            
    for i in range(40):
        for j in range(40):
            cnt += 1
            x = 0
            y = r * i/40
            z = r * j/40
            p = Point(x, y, z, cnt)
            points[cnt] = p
     
    for i in range(40):
        for j in range(40):
            cnt += 1
            x = 30
            y = r * i/40
            z = r * j/40
            p = Point(x, y, z, cnt)
            points[cnt] = p 
                   
    for i in range(40):
        for j in range(40):
            cnt += 1
            x = r * i/40
            y = 30
            z = r * j/40
            p = Point(x, y, z, cnt)
            points[cnt] = p

    for i in range(40):
        for j in range(40):
            cnt += 1
            x = r * i/40
            y = r * j/40
            z = 30
            p = Point(x, y, z, cnt)
            points[cnt] = p

    return points, hyperparameter, name




def cube_sameDensity():
    random.seed(0)
    points = defaultdict(Point)
    name = 'cube_sameDensity'

    hyperparameter = Hyperparameter(
    R1 = 2, H1 = 0.2, ratioThreshold1 = 0.85,
    eps_normal = 0.1, min_samples_normal = 9,
    eps_centerPoint = 3, min_samples_centerPoint = 10,
    eps_finalBoundaryPoint = 2, min_samples_finalBoundaryPoint = 10,
    edgeRansacH = 0.3)
    
    r = 30
    cnt = -1    
    for i in range(50):
        for j in range(50):
            x = r * i/50
            y = r * j/50
            if 11 <= x and x <= 20 and 11 <= y and y <= 20:
                cnt += 1
                z = 0
                p = Point(x, y, z, cnt)
                points[cnt] = p

    for i in range(50):
        for j in range(50):
            x = r * i/50
            y = r * j/50
            if 11 <= x and x <= 20 and 11 <= y and y <= 20:
                cnt += 1
                z = 9
                p = Point(x, y, z, cnt)
                points[cnt] = p
    
    for i in range(900):
        cnt += 1
        if i < 225:
            x = 11 + (i % 15)*3/5
            y = 11
            z = (i // 15)*3/5
        elif i < 450:
            x = 11 
            y = 11 + ((i-225) % 15)*3/5
            z = ((i-225) // 15)*3/5
        elif i < 675:
            x = 11 + ((i-450) % 15)*3/5
            y = 20
            z = ((i-450) // 15)*3/5
        else:
            x = 20
            y = 11 + ((i-675) % 15)*3/5
            z = ((i-675) // 15)*3/5
        p = Point(x, y, z, cnt)
        points[cnt] = p

    return points, hyperparameter, name




def VertexDense():
    random.seed(0)
    points = defaultdict(Point)
    name = 'VertexDense'

    hyperparameter = Hyperparameter(
    R1 = 2, H1 = 0.2, ratioThreshold1 = 0.8,
    eps_normal = 0.1, min_samples_normal = 9,
    eps_centerPoint = 3, min_samples_centerPoint = 30,
    eps_finalBoundaryPoint = 2, min_samples_finalBoundaryPoint = 10,
    edgeRansacH = 0.3)
    
    r = 30
    cnt = -1    
    for i in range(50):
        for j in range(50):
            diff = 0
            cnt += 1
            x = r * i/50
            y = r * j/50
            z = 0
            if 11 <= x and x <= 20 and 11 <= y and y <= 20:
                z = 9
            p = Point(x, y, z+diff, cnt)
            points[cnt] = p
    
    for i in range(900):
        diff = 0
        cnt += 1
        if i < 225:
            x = 11 + (i % 15)*3/5
            y = 11
            z = (i // 15)*3/5
        elif i < 450:
            x = 11 
            y = 11 + ((i-225) % 15)*3/5
            z = ((i-225) // 15)*3/5
        elif i < 675:
            x = 11 + ((i-450) % 15)*3/5
            y = 20
            z = ((i-450) // 15)*3/5
        else:
            x = 20
            y = 11 + ((i-675) % 15)*3/5
            z = ((i-675) // 15)*3/5
        p = Point(x, y, z+diff, cnt)
        points[cnt] = p
    
    for i in range(50):
        for j in range(50):
            diff = 0
            cnt += 1
            x = r * i/50
            y = 0
            z = r * j/50
            p = Point(x, y+diff, z, cnt)
            points[cnt] = p
            
    for i in range(50):
        for j in range(50):
            diff = 0
            cnt += 1
            x = 0
            y = r * i/50
            z = r * j/50
            p = Point(x+diff, y, z, cnt)
            points[cnt] = p

    return points, hyperparameter, name





def NonUniformCube():
    random.seed(0)
    points = defaultdict(Point)
    name = 'NonUniformCube'

    hyperparameter = Hyperparameter(
    R1 = 2, H1 = 0.2, ratioThreshold1 = 0.85,
    eps_normal = 0.1, min_samples_normal = 9,
    eps_centerPoint = 3, min_samples_centerPoint = 10,
    eps_finalBoundaryPoint = 2, min_samples_finalBoundaryPoint = 10,
    edgeRansacH = 0.3)
    
    r = 30
    cnt = -1    
    for i in range(150):
        for j in range(150):
            x = r * i/150
            y = r * j/150
            if 11 <= x and x <= 20 and 11 <= y and y <= 20:
                cnt += 1
                z = 0
                p = Point(x, y, z, cnt)
                points[cnt] = p

    for i in range(150):
        for j in range(150):
            x = r * i/150
            y = r * j/150
            if 11 <= x and x <= 20 and 11 <= y and y <= 20:
                cnt += 1
                z = 9
                p = Point(x, y, z, cnt)
                points[cnt] = p
    
    for i in range(900):
        cnt += 1
        if i < 225:
            x = 11 + (i % 15)*3/5
            y = 11
            z = (i // 15)*3/5
        elif i < 450:
            x = 11 
            y = 11 + ((i-225) % 15)*3/5
            z = ((i-225) // 15)*3/5
        elif i < 675:
            x = 11 + ((i-450) % 15)*3/5
            y = 20
            z = ((i-450) // 15)*3/5
        else:
            x = 20
            y = 11 + ((i-675) % 15)*3/5
            z = ((i-675) // 15)*3/5
        p = Point(x, y, z, cnt)
        points[cnt] = p

    return points, hyperparameter, name




def FourCleanBoxes():
    random.seed(0)
    points = defaultdict(Point)
    name = 'FourCleanBoxes'

    hyperparameter = Hyperparameter(
    R1 = 0.8, H1 = 0.05, ratioThreshold1 = 0.93,
    eps_normal = 0.15, min_samples_normal = 20,
    eps_centerPoint = 1, min_samples_centerPoint = 20,
    eps_finalBoundaryPoint = 1, min_samples_finalBoundaryPoint = 10,
    edgeRansacH = 0.05, lineardensity = 0.25)
    
    cnt = 0    
    density = 0.25
    #planes
    for i in range(120):
        for j in range(120):
            x = 30 * i/120
            y = 30 * j/120
            z = 0
            if 3 <= x and x <= 14 and 3 <= y and y <= 14:
                z = 11
                if 6 <= x and x <= 11 and 6 <= y and y <= 11:
                    z = 16
            if 16 <= x and x <= 27 and 16 <= y and y <= 27:
                z = 11
            p = Point(x, y, z, cnt)
            points[cnt] = p
            cnt += 1
    
    points = fillRect(Point(0,0,0,None), Point(0,30,20,None), points, density)
    points = fillRect(Point(0,30,0,None), Point(30,30,20,None), points, density)

    #cube1.1
    points = fillRect(Point(3,3,0,None), Point(3,14,11,None), points, density)
    points = fillRect(Point(3,3,0,None), Point(14,3,11,None), points, density)
    points = fillRect(Point(3,14,0,None), Point(14,14,11,None), points, density)
    points = fillRect(Point(14,3,0,None), Point(14,14,11,None), points, density)

    #cube1.2
    points = fillRect(Point(6,6,11,None), Point(6,11,16,None), points, density)
    points = fillRect(Point(6,6,11,None), Point(11,6,16,None), points, density)
    points = fillRect(Point(6,11,11,None), Point(11,11,16,None), points, density)
    points = fillRect(Point(11,6,11,None), Point(11,11,16,None), points, density)

    cnt = len(points)
    #cube2.1
    for i in range(44):
        for j in range(44):
            x = 16 + 11 * i/44
            y = 16
            z = 11 * j/44
            if 19 <= x and x <= 24 and 3 <= z and z <= 8:
                y = 11
            p = Point(x, y, z, cnt)
            points[cnt] = p
            cnt += 1
    
    points = fillRect(Point(16,16,0,None), Point(16,27,11,None), points, density)
    points = fillRect(Point(27,16,0,None), Point(27,27,11,None), points, density)
    points = fillRect(Point(16,27,0,None), Point(27,27,11,None), points, density)

    #cube2.2
    points = fillRect(Point(19,11,3,None), Point(19,16,8,None), points, density)
    points = fillRect(Point(19,11,3,None), Point(24,16,3,None), points, density)
    points = fillRect(Point(24,11,3,None), Point(24,16,8,None), points, density)
    points = fillRect(Point(19,11,8,None), Point(24,16,8,None), points, density)

    return points, hyperparameter, name







def OpenPlane():
    random.seed(0)
    points = defaultdict(Point)
    name = 'OpenPlane'

    hyperparameter = Hyperparameter(
    R1 = 3, H1 = 0.3, ratioThreshold1 = 0.8,
    eps_normal = 1, min_samples_normal = 15,
    eps_centerPoint = 4, min_samples_centerPoint = 30,
    eps_finalBoundaryPoint = 3, min_samples_finalBoundaryPoint = 7,
    edgeRansacH = 0.1)
    
    ultung = 0.3
    r = 30
    cnt = -1    
    for i in range(50):
        for j in range(50):
            diff = ultung*(random.random()-0.5)
            cnt += 1
            x = r * i/50
            y = r * j/50
            z = 0
            p = Point(x, y, z+diff, cnt)
            points[cnt] = p
    
    for i in range(50):
        for j in range(50):
            diff = ultung*(random.random()-0.5)
            cnt += 1
            x = 0
            y = r * i/50
            z = r * j/50
            p = Point(x+diff, y, z, cnt)
            points[cnt] = p

    for i in range(20):
        for j in range(20):
            diff = ultung*(random.random()-0.5)
            cnt += 1
            x = 15
            y = 10 + r*2/5 * i/20
            z = r*2/5 * j/20
            p = Point(x+diff, y, z, cnt)
            points[cnt] = p
        



    return points, hyperparameter, name







def FloorWall():
    random.seed(0)
    points = defaultdict(Point)
    name = 'FloorWall'

    hyperparameter = Hyperparameter(
    R1 = 2, H1 = 0.05, ratioThreshold1 = 0.8,
    eps_normal = 0.15, min_samples_normal = 10,
    eps_centerPoint = 0.8, min_samples_centerPoint = 5,
    eps_finalBoundaryPoint = 1.3, min_samples_finalBoundaryPoint = 10,
    edgeRansacH = 0.05)
    
    r = 30
    cnt = -1    
    for i in range(50):
        for j in range(50):
            diff = 0
            cnt += 1
            x = r * i/50
            y = r * j/50
            z = 0
            if 4 <= x and x <= 13 and 4 <= y and y <= 13:
                z = 9
            if 17 <= x and x <= 26 and 21 <= y and y <= 30:
                z = 9
            p = Point(x, y, z+diff, cnt)
            points[cnt] = p

    for i in range(50):
        for j in range(50):
            diff = 0
            cnt += 1
            x = r * i/50
            y = r
            z = r * j/50
            if 4 <= x and x <= 13 and 17 <= z and z <= 26:
                y = r - 9
            if 17 <= x and x <= 26 and 0 <= z and z <= 9:
                y = r - 9
            p = Point(x, y+diff, z, cnt)
            points[cnt] = p
    
    for i in range(900):
        diff = 0
        cnt += 1
        if i < 225:
            x = 4 + (i % 15)*3/5
            y = 4
            z = (i // 15)*3/5
        elif i < 450:
            x = 4 
            y = 4 + ((i-225) % 15)*3/5
            z = ((i-225) // 15)*3/5
        elif i < 675:
            x = 4 + ((i-450) % 15)*3/5
            y = 13
            z = ((i-450) // 15)*3/5
        else:
            x = 13
            y = 4 + ((i-675) % 15)*3/5
            z = ((i-675) // 15)*3/5
        p = Point(x, y, z+diff, cnt)
        points[cnt] = p

    for i in range(900):
        diff = 0
        cnt += 1
        if i < 225:
            x = 4 + (i % 15)*3/5
            y = r - (i // 15)*3/5
            z = 17
        elif i < 450:
            x = 4 
            y = r - ((i-225) // 15)*3/5
            z = 17 + ((i-225) % 15)*3/5
        elif i < 675:
            x = 4 + ((i-450) % 15)*3/5
            y = r - ((i-450) // 15)*3/5
            z = 26
        else:
            x = 13
            y = r - ((i-675) // 15)*3/5
            z = 17 + ((i-675) % 15)*3/5
        p = Point(x, y, z+diff, cnt)
        points[cnt] = p



    for i in range(450):
        diff = 0
        cnt += 1
        if i < 225:
            x = 17 
            y = 21 + ((i) % 15)*3/5
            z = ((i) // 15)*3/5
        else:
            x = 26
            y = 21 + ((i-225) % 15)*3/5
            z = ((i-225) // 15)*3/5
        p = Point(x, y, z+diff, cnt)
        points[cnt] = p
    




    
    for i in range(50):
        for j in range(50):
            diff = 0
            cnt += 1
            x = 0
            y = r * i/50
            z = r * j/50
            p = Point(x, y+diff, z, cnt)
            points[cnt] = p

    return points, hyperparameter, name