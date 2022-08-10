from math import sin, cos, pi, asin
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

x = []
y = []
z = []
for angle in range(30, 151):
    alpha = angle*pi/180
    # for i in range(1, 40):
    r_h_ratio =  0.05
    try:
        w1 = asin(r_h_ratio/sin(alpha-asin(2*r_h_ratio)))
    except:
        w1 = 0
    try:
        w2 = asin(r_h_ratio/sin(alpha))
    except:
        w2 = 0
    try:
        w3 = asin(r_h_ratio/sin(alpha/2))
    except:
        w3 = 0
    res = [0.5+(cos(w1)*sin(w1)+w1)/pi, 0.5+(cos(w2)*sin(w2)+w2)/pi, (sin(2*w3)+2*w3)/pi]
    result = max(res)
    x.append(r_h_ratio)
    y.append(alpha)
    z.append(result)
        
# fig = plt.figure(figsize=(9,6))
# ax = fig.add_subplot(111, projection="3d")
# ap = np.array([[x[i], y[i], z[i]] for i in range(len(x))])
# print(len(x), len(y), len(z))
# ax.plot_surface(ap[:0], ap[:1], ap[:2], c=[0]*len(x), cmap="rainbow")
plt.show()
print(x,y,z, max(z))
    
