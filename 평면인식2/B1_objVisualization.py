from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(0, 10, 0.1)
y = np.sin(x)
x_m, y_m = np.meshgrid(x, y)
z = x_m + 5 * y_m

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z, cmap="brg_r")
plt.show()
