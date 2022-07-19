class Point:
    def __init__(self, X, Y, Z, idx, R = None, G = None, B = None):
        self.x = X 
        self.y = Y 
        self.z = Z
        self.idx = idx
        self.R = R
        self.G = G
        self.B = B
        self.nearby = list() #가까운 점들의 (distance, index)가 들어감
        self.normal = None
        self.cluster = None
       
        
    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z)
    
    def distance(self, p):
        def sqsbt(a,b):
            return (a-b)**2
        return ((sqsbt(self.x, p.x)+sqsbt(self.y, p.y)+sqsbt(self.z, p.z)))**(0.5)

class plane:
    def __init__(self, polygon, equation):
        self.equation = equation
        self.polygon = polygon
    def __str__(self):
        return self.polygon
        