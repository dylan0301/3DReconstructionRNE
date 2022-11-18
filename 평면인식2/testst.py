import numpy as np
import scipy.spatial as spatial

points = np.array([(1,2,3),(1,2,4),(1,2,5)])
point_tree = spatial.cKDTree(points)

# print(point_tree.data[point_tree.query_ball_point([1,2,2],1.5)])
print(point_tree.data[point_tree.query_ball_point([1,2,2],2)])