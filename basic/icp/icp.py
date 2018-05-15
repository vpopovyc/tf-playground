import numpy as np
import matplotlib.pyplot as plt

base_set = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0], [9.0, 9.0], [10.0, 10.0]], np.float32)

angle = 30.
theta = (angle/180.) * np.pi
rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                       [np.sin(theta),  np.cos(theta)]])

for point in base_set:
    point[1] += np.random.randint(0, high=5)

rotated_set = np.dot(base_set, rot_matrix)

base_centroid = np.mean(base_set, axis=0)
rotated_centroid = np.mean(rotated_set, axis=0)

variance = np.sum(np.square((base_set-base_centroid)), axis=0)/(base_set.size-1)

relative_variance = variance/base_centroid

covariance = np.mean((base_set-base_centroid)*(rotated_set-rotated_centroid), axis=0)

def closest_point(point, scene_points):
    min_distance = np.inf
    closest_point_index = -1
    i = 0
    for scene_point in scene_points:
        distance = np.sqrt(np.square(np.sum(point - scene_point)))
        if min_distance > distance:
            min_distance = distance
            closest_point_index = i
        i += 1
    assert np.isinf(min_distance) == False
    return closest_point_index

def plot_closest_points(base_set, rotated_set, indices_of_closest_points):
    for index in range(base_set.shape[0]):
        wIndex = indices_of_closest_points[index]
        plt.plot([rotated_set[index, 0], base_set[wIndex, 0]],
                 [rotated_set[index, 1], base_set[wIndex, 1]],
                 linestyle=":")

def icp_loop(scene_points, model_points):

    indices_of_closest_points = []

    for model_point in model_points:
        # Find closest point in scene_points
        indices_of_closest_points.append(closest_point(model_point, scene_points))

    plot_closest_points(base_set, rotated_set, indices_of_closest_points)

icp_loop(base_set, rotated_set)

plt.text(-10.0, 10.0, "relative_variance: " + str(relative_variance), ha="left")
plt.text(-10.0, 9.0, "covariance: " + str(covariance), ha="left")
plt.plot(base_set[:, 0], base_set[:, 1])
plt.plot(rotated_set[:, 0], rotated_set[:, 1])
plt.plot(base_centroid[0], base_centroid[1], 'bo')
plt.plot(rotated_centroid[0], rotated_centroid[1], 'bo')
plt.grid(True)
plt.show()