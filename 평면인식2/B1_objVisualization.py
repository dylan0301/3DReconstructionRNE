from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

x = [0,0,0,0]
y = [0,0,1,1]
z = [0,1,1,0]
verts = [list(zip(x,y,z))]
polys = Poly3DCollection(verts)
polys.set_edgecolor('k')
polys.set_facecolor(colors.rgb2hex(np.random.rand(3)))
ax.add_collection3d(polys)
x = [0,0,1,1]
y = [0,0,0,0]
z = [0,1,1,0]
verts = [list(zip(x,y,z))]
polys = Poly3DCollection(verts)
polys.set_edgecolor('k')
polys.set_facecolor(colors.rgb2hex(np.random.rand(3)))
ax.add_collection3d(polys)
plt.show()
# for i in range(100):
#     vtx = np.random.rand(3,3)
#     tri = a3.art3d.Poly3DCollection([vtx])
#     tri.set_color(colors.rgb2hex(np.random.rand(3)))
#     tri.set_edgecolor('k')
#     ax.add_collection3d(tri)
