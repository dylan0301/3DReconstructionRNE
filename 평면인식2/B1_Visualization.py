from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def objVisualization(objList):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    for i in range(len(objList)):
        obj = objList[i]
        for plane in obj.planes:
            pl = [list(v.mainpoint) for v in plane.polygon]
            polys = Poly3DCollection(pl)
            polys.set_edgecolor('k')
            polys.set_facecolor(colors.rgb2hex(np.random.rand(3)))
            ax.add_collection3d(polys)
    plt.show()
    
#받는 데이터: objList
#obj들끼리 idx 다 달라서 그걸 label로 하면 될듯
#objList[i] = object class
#object class안에 planes 라는 set 이 있음
#그 set 안에 plane마다 polygon 이라는 list가 있음
#그 list에는 vertex class들이 있음
#각 vertex 안에는 mainpoint 라는 np.array([x,y,z]) 넘파이 배열이 있음.



