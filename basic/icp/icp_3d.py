from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mset = np.array([
				  [ 0.0,  0.0, 5.0],
				  [ 5.0,  0.0, 0.0],
				  [-5.0,  0.0, 0.0],
				  [ 0.0,  0.0, 5.0],
				  [ 5.0,  0.0, 0.0],
				  [-5.0,  0.0, 0.0],
				  [ 0.0,  0.0, 5.0],
				  [ 0.0,  5.0, 0.0],
				  [ 0.0, -5.0, 0.0]
				], np.float32)

angle = -45
theta = angle * np.pi / 180.0
rot_matrix = np.array([
                       [np.cos(theta), -np.sin(theta), 0], 
                       [np.sin(theta),  np.cos(theta), 0],
                       [0            ,  0            , 1]
                      ])

sset = np.array(mset)

for i, point in enumerate(sset):
  sset[i] = np.matmul(rot_matrix, point)

colors = cm.viridis(np.array(mset[:, 2]))


'''
ICP here
'''

ax.plot_wireframe(np.array([mset[:, 0]]), np.array([mset[:, 1]]), np.array([mset[:, 2]]), rstride=1, cstride=1, color="green")
ax.plot_wireframe(np.array([sset[:, 0]]), np.array([sset[:, 1]]), np.array([sset[:, 2]]), rstride=1, cstride=1, color="red")

plt.show()
