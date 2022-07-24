import matplotlib.pyplot as plt
import math as mt

x = [0.001*i for i in range(1000)]
y = []
r = 1
h = 0.05
alpha = mt.pi/3


def integration(x):
    return x*mt.sqrt(r**2-x**2)+mt.atan(x/mt.sqrt(r**2-x**2))

for i in x:
    if mt.sin(i) != 0 and mt.sin(i) != alpha:
        A = 2*h/mt.sin(alpha-i)
        B = -2*h/mt.sin(i)
        res = 0
        if r**2 - A**2 < 0:
            res += mt.pi*r**2/2
            res += (-1)*integration(B)
        elif r**2 - B**2 < 0:
            res += mt.pi*r**2/2
            res += integration(A)
        else:
            res = integration(A) - integration(B)
        res /= mt.pi*r**2
        y.append(res)
    else:
        y.append(0)
        
plt.plot(x,y,'ro')
plt.show()