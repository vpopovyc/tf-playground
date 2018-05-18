from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import icp

mset = np.array([
				  [ 5.0,  0.0, 0.0],
				  [-5.0,  0.0, 0.0],
				  [ 0.0,  0.0, 5.0],
				  [ 0.0,  0.0, -5.0],
				  [ 0.0,  5.0, 0.0],
				  [ 0.0, -5.0, 0.0]
				], np.float32)

angle = 45
theta = angle * np.pi / 180.0
rot_matrix = np.array([
                       [np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta),  np.cos(theta), 0],
                       [0            ,  0            , 1]
                      ])

sset = np.array(mset)

for i, point in enumerate(sset):
  sset[i] = np.dot(point, rot_matrix) + np.array([10.0, 0.0, 0.0])

new_sset = np.array(sset)

mset_centroid = np.mean(mset, axis=0)
sset_centroid = np.mean(sset, axis=0)
new_sset_centroid = np.mean(new_sset, axis=0)

'''
<ICP>
'''

def translate_my_shit():
	global new_sset
	print("~~~~~~~~~~~~~~~ Translating ~~~~~~~~~~~~~~~~~~~~")
	new_sset, loss = icp.icp_loop(mset, new_sset, translation_only=True)
	print("Loss: ", loss)

def rotate_my_shit():
	global new_sset
	print("~~~~~~~~~~~~~~~ Rotating ~~~~~~~~~~~~~~~~~~~~~~~")
	new_sset, loss = icp.icp_loop(mset, new_sset, rotation_only=True)
	print("Loss: ", loss)

def do_shit():
	global new_sset
	print("~~~~~~~~~~~~~~~ Translating & Rotating ~~~~~~~~~~~~~~~~~~~~~~~")
	new_sset, loss = icp.icp_loop(mset, new_sset)
	print("Loss: ", loss)

'''
</ICP>
'''

def press(event):
	if event.key == 'w':
		quit()

	if event.key == 'r':
		rotate_my_shit()

	if event.key == 't':
		translate_my_shit()

	if event.key == 'a':
		do_shit()

	plt.clf()
	plt.cla()
	plt.close()
	draw_my_shit(True if event.key == 'p' else False)

def draw_my_shit(mode):
	fig = plt.figure(figsize=(20,10))
	ax = fig.add_subplot(111, projection='3d')

	colors = cm.viridis(np.array(mset[:, 2]))
	
	ax.plot_wireframe(np.array([mset[:, 0]]), np.array([mset[:, 1]]), np.array([mset[:, 2]]), rstride=1, cstride=1, color="green")
	ax.plot_wireframe(np.array([sset[:, 0]]), np.array([sset[:, 1]]), np.array([sset[:, 2]]), rstride=1, cstride=1, color="red")
	ax.plot_wireframe(np.array([new_sset[:, 0]]), np.array([new_sset[:, 1]]), np.array([new_sset[:, 2]]), rstride=1, cstride=1, color="blue", linestyle="dashed")

	ax.scatter(mset_centroid[0], mset_centroid[1], mset_centroid[2])
	ax.scatter(sset_centroid[0], sset_centroid[1], sset_centroid[2])
	ax.scatter(new_sset_centroid[0], new_sset_centroid[1], new_sset_centroid[2])
	
	if mode:
		closest_points = []

		for model_point in new_sset:
			closest_points.append(icp.closest_point(model_point, mset))

		closest_points = np.array(closest_points)

		line = []

		for i in range(new_sset.shape[0]):
			line.append(closest_points[i, :])
			line.append(new_sset[i, :])

		line = np.array(line)

		for i in range(new_sset.size - 1):
			ax.plot(line[:, 0], line[:, 1], line[:, 2], color="grey", linestyle=":", linewidth=0.5)

	fig.canvas.mpl_connect('key_press_event', press)
	
	plt.show()

draw_my_shit(False)
