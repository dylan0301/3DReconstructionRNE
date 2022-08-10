#하나의 nearby에서 그냥 생으로 랜색을 돌려서 뭐가 나오는지 확인해보기
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from A2_data import butterfly, butterfly_uniform
from A1_classes import *
import pandas as pd
from IPython.display import display

def simulation_ransac_test_func(R = 5, alpha = np.pi/3, lineardensity = 0.05, H = 0.4):


    #AllPoints, hyperparameter = butterfly(R = R, alpha = alpha)
    AllPoints, hyperparameter = butterfly_uniform(R = R, alpha = alpha, lineardensity = lineardensity)

    hyperparameter.H1 = H




    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ap = np.array([[p.x, p.y, p.z] for p in AllPoints.values()])
    # ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=[0 for i in range(len(AllPoints))], marker='o', s=15, cmap='rainbow')
    # plt.show()



    def P_RansacPlane(AllPoints, hyperparameter):
        random.seed(0)
        
        #점 p랑 ax+by+cz+d=0 수직거리
        def sujikDistance(p, plane):
            a, b, c, d = plane[0], plane[1], plane[2], plane[3]
            x, y, z = p.x, p.y, p.z
            res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
            return res

        #점 3개지나는 평면의 방정식 abcd 튜플로 리턴
        def findPlane(p1, p2, p3):
            v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
            v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
            normal = np.cross(v12,v13)
            if np.linalg.norm(normal) < 0:
                return None
            d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
            return (normal[0], normal[1], normal[2], d)
        
        pts = list(AllPoints.values())

        numOfpts = len(pts)
        maxScore = 0
        bestPlane = None
        bestSatisfied = set()
        for trial in range(100):
            plane = None
            while plane == None:
                i1 = random.randrange(0,numOfpts)
                i2 = random.randrange(0,numOfpts)
                while i1 == i2:
                    i2 = random.randrange(0,numOfpts)
                i3 = random.randrange(0,numOfpts)
                while i1 == i3 or i2 == i3:
                    i3 = random.randrange(0,numOfpts)
                plane = findPlane(pts[i1], pts[i2], pts[i3])
            score = 0

            satisfied = set()
            for p in pts:
                d = sujikDistance(p, plane)
                if d < hyperparameter.H1:
                    score +=1
                    satisfied.add(p)
            if score > maxScore:
                maxScore = score #이거 넣는걸 깜빡해서 문제가 좀 생겼음. 근데 정육면체데이터는 괜찮았네? 아무거나 평면 잡아도 다 거기서거기라 그런듯
                bestPlane = plane
                bestSatisfied = satisfied
        return bestPlane, maxScore, bestSatisfied

    bestPlane, maxScore, bestSatisfied = P_RansacPlane(AllPoints, hyperparameter)



    

    print()
    print()
    print('R =', R, '/ H =', H, '/ alpha =', alpha)
    print('alpha in degrees:', alpha*180/np.pi)
    print('2H/R:', 2*H/R)
    print()

    print(maxScore,'out of', len(AllPoints))
    print('ratio:', maxScore/len(AllPoints))
    normal = np.array([bestPlane[0], bestPlane[1], bestPlane[2]])
    normal /= np.linalg.norm(normal)
    print('normal vector:',normal)
    print()

    beta_experimental = np.arctan(-normal[1]/normal[2])
    print('beta_experimental:', beta_experimental)
    print('beta_experimental in degrees:', beta_experimental*180/np.pi)
    print()

    print('beta_calculated1 = 0')
    beta_calculated1 = 0
    print('angle difference in degrees:', (beta_calculated1-beta_experimental)*180/np.pi)
    def sik(insideSin):
        repeated = 2*H/R/np.sin(insideSin)
        return (repeated*np.sqrt(1-repeated**2)+np.arcsin(repeated))/np.pi
    calculated1_ratio = 0.5+sik(alpha)
    print('calculated1_ratio:', calculated1_ratio)
    print()



    print('beta_calculated2 = arcsin(2H/R)')
    beta_calculated2 = np.arcsin(2*H/R)
    print('beta_calculated2:', beta_calculated2)
    print('beta_calculated2 in degrees:', beta_calculated2*180/np.pi)
    print('angle difference in degrees:', (beta_calculated2-beta_experimental)*180/np.pi)
    calculated2_ratio = sik(alpha-beta_calculated2)+sik(beta_calculated2)
    print('calculated2_ratio:', calculated2_ratio)
    print()

    print('beta_calculated3 = alpha/2')
    beta_calculated3 = alpha/2
    print('beta_calculated3:', beta_calculated3)
    print('beta_calculated3 in degrees:', beta_calculated3*180/np.pi)
    print('angle difference in degrees:', (beta_calculated3-beta_experimental)*180/np.pi)
    repeated = 2*np.arcsin(2*H/R/np.sin(beta_calculated3))
    calculated3_ratio = (np.sin(repeated)+repeated)/np.pi
    print('calculated3_ratio:', calculated3_ratio)
    print()

    #오차율 = (이론값-측정값)/이론값*100
    #각도에서는 오차율 따지면 안됨. 똑같은 1도차이여도 오차율 너무 달라짐.
    #그래서 그냥 각도차이 따진다.
    #error_rate = (beta_calculated-beta_experimental)/beta_calculated*100
    #print('error_rate:',error_rate)


    print()
    print()

    return bestPlane, AllPoints, bestSatisfied

def simulation_ransac_test_visual(bestPlane, AllPoints, bestSatisfied, R, alpha, lineardensity):
    planePoints = []
    for x in np.arange(-R-1, R+1, lineardensity*10):
        for y in np.arange(-R-1, R+1, lineardensity*10):
            z = -(bestPlane[0]*x + bestPlane[1]*y + bestPlane[3])/bestPlane[2]
            if y > R*np.cos(alpha)+R/5:
                continue
            if z < -R/5:
                continue
            if z > R*np.sin(alpha)+R/5:
                continue
            p = Point(x, y, z, None)        
            planePoints.append(p)

    def isSatisfied(p):
        if p in bestSatisfied:
            return 1
        return 0

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    #planePoints = []

    labels = [isSatisfied(p) for p in AllPoints.values()]
    labels.extend([2 for i in range(len(planePoints))])

    ap = np.array([[p.x, p.y, p.z] for p in AllPoints.values()]+[[p.x, p.y, p.z] for p in planePoints])
    ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=labels, marker='o', s=15, cmap='rainbow')
    # plt.title('R = '+ str(R)+' / H = '+ str(H)+ ' / alpha = '+str(alpha*180/np.pi)+' degrees'
    # +'\n'+'beta = '+str(beta_experimental*180/np.pi)+' degrees'+'\n'+'ratio = '+ str(maxScore/len(AllPoints)))
    plt.show()






def makeimages():

    R = 5
    alpha = np.pi/3
    lineardensity = 0.025
    H = 0.4

    bestPlane, AllPoints, bestSatisfied = simulation_ransac_test_func(R, alpha, lineardensity, H)
    simulation_ransac_test_visual(bestPlane, AllPoints, bestSatisfied, R, alpha, lineardensity)


    R = 5
    alpha = np.pi/4
    lineardensity = 0.025
    H = 0.4

    bestPlane, AllPoints, bestSatisfied = simulation_ransac_test_func(R, alpha, lineardensity, H)
    simulation_ransac_test_visual(bestPlane, AllPoints, bestSatisfied, R, alpha, lineardensity)



def simulation_ransac_test_func_for_table(R = 5, alpha = np.pi/3, lineardensity = 0.025, H = 0.4):


    AllPoints, hyperparameter = butterfly_uniform(R = R, alpha = alpha, lineardensity = lineardensity)

    hyperparameter.H1 = H


    def P_RansacPlane(AllPoints, hyperparameter):
        random.seed(0)
        
        #점 p랑 ax+by+cz+d=0 수직거리
        def sujikDistance(p, plane):
            a, b, c, d = plane[0], plane[1], plane[2], plane[3]
            x, y, z = p.x, p.y, p.z
            res = abs(a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
            return res

        #점 3개지나는 평면의 방정식 abcd 튜플로 리턴
        def findPlane(p1, p2, p3):
            v12 = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
            v13 = np.array([p1.x-p3.x, p1.y-p3.y, p1.z-p3.z])
            normal = np.cross(v12,v13)
            if np.linalg.norm(normal) < 0:
                return None
            d = -(normal[0]*p1.x + normal[1]*p1.y + normal[2]*p1.z)
            return (normal[0], normal[1], normal[2], d)
        
        pts = list(AllPoints.values())

        numOfpts = len(pts)
        maxScore = 0
        bestPlane = None
        bestSatisfied = set()
        for trial in range(100):
            plane = None
            while plane == None:
                i1 = random.randrange(0,numOfpts)
                i2 = random.randrange(0,numOfpts)
                while i1 == i2:
                    i2 = random.randrange(0,numOfpts)
                i3 = random.randrange(0,numOfpts)
                while i1 == i3 or i2 == i3:
                    i3 = random.randrange(0,numOfpts)
                plane = findPlane(pts[i1], pts[i2], pts[i3])
            score = 0

            satisfied = set()
            for p in pts:
                d = sujikDistance(p, plane)
                if d < hyperparameter.H1:
                    score +=1
                    satisfied.add(p)
            if score > maxScore:
                maxScore = score #이거 넣는걸 깜빡해서 문제가 좀 생겼음. 근데 정육면체데이터는 괜찮았네? 아무거나 평면 잡아도 다 거기서거기라 그런듯
                bestPlane = plane
                bestSatisfied = satisfied
        return bestPlane, maxScore, bestSatisfied

    bestPlane, maxScore, bestSatisfied = P_RansacPlane(AllPoints, hyperparameter)



    


    ratio_experimental = maxScore/len(AllPoints)

    normal = np.array([bestPlane[0], bestPlane[1], bestPlane[2]])
    normal /= np.linalg.norm(normal)

    beta_experimental = np.arctan(-normal[1]/normal[2])


    beta_calculated1 = 0
    def sik(insideSin):
        repeated = 2*H/R/np.sin(insideSin)
        return (repeated*np.sqrt(1-repeated**2)+np.arcsin(repeated))/np.pi
    calculated1_ratio = 0.5+sik(alpha)


    beta_calculated2 = np.arcsin(2*H/R)  
    calculated2_ratio = sik(alpha-beta_calculated2)+sik(beta_calculated2)
 

    beta_calculated3 = alpha/2
    repeated = 2*np.arcsin(2*H/R/np.sin(beta_calculated3))
    calculated3_ratio = (np.sin(repeated)+repeated)/np.pi

    max_ratio_beta_equation = ''
    maximum_calculated_ratio = max(calculated1_ratio, calculated2_ratio, calculated3_ratio)
    if maximum_calculated_ratio == calculated1_ratio:
        max_ratio_beta_equation = 'Beta = 0'
        calculated_beta = beta_calculated1
    elif maximum_calculated_ratio == calculated2_ratio:
        max_ratio_beta_equation = 'Beta = arcsin(2H/R)'
        calculated_beta = beta_calculated2
    elif maximum_calculated_ratio == calculated3_ratio:
        max_ratio_beta_equation = 'Beta = Alpha/2'
        calculated_beta = beta_calculated3
    else:
        raise Exception('float error?')

    return [2*H/R, alpha*180/np.pi, max_ratio_beta_equation, calculated_beta*180/np.pi, maximum_calculated_ratio, beta_experimental*180/np.pi, ratio_experimental]





def maketable():
    #2H/R, alpha, 계산한거 어떤 각도일때 최대되는지, 그 각도, 계산한거 ratio, 실험 각도, 실험 ratio
    col = ['2H/R', 'Alpha', 'Max_Ratio_Beta_Equation', 'Calculated_Beta', 'Calculated_Ratio', 'Experimental_Beta', 'Experimental_Ratio']
    lineardensity = 0.025
    
    #R, alpha, H 튜플
    samples  = [(5, np.pi/2.5, 0.4),
                (5, np.pi/3, 0.4),
                (5, np.pi/4, 0.4),
                (5, np.pi/5, 0.4),
                (5, np.pi/2.5, 0.8),
                (5, np.pi/3, 0.8),
                (5, np.pi/4, 0.8),
                (5, np.pi/5, 0.8)]
    
    content = []

    for i in range(len(samples)):
        R = samples[i][0]
        alpha = samples[i][1]
        H = samples[i][2]
        newline = simulation_ransac_test_func_for_table(R = R, alpha = alpha, lineardensity = lineardensity, H = H)
        content.append(newline)
    
    
    df = pd.DataFrame(data = content, columns=col)
    display(df)
    df.to_excel("justification.xlsx")
    


maketable()