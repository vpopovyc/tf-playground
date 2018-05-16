import numpy as np
import matplotlib.pyplot as plt

base_set = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0], [4.0, 4.0, 0.0], [5.0, 5.0, 0.0], [6.0, 6.0, 0.0], [7.0, 7.0, 0.0], [8.0, 8.0, 0.0], [9.0, 9.0, 0.0], [10.0, 10.0, 0.0]], np.float32)

angle = 45
theta = angle * np.pi / 180.
rot_matrix = np.array([
                       [np.cos(theta), -np.sin(theta), 0], 
                       [np.sin(theta),  np.cos(theta), 0],
                       [0            ,  0            , 1]
                      ])

for point in base_set:
    point[1] += np.random.randint(4, high=8)

rotated_set = np.dot(base_set, rot_matrix)

# variance = np.sum(np.square((base_set-base_centroid)), axis=0)/(base_set.size-1)

# relative_variance = variance/(base_centroid+np.finfo(float).eps)

# covariance = np.mean((base_set-base_centroid)*(rotated_set-rotated_centroid), axis=0)

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

def l2_norm(x):
    x = np.sqrt(np.sum(np.square(x)))

def compute_mse(set):
    np.mean(set)

def align(scene_points, model_points):

    # Find centroids of two point sets
    scene_centroid = np.mean(base_set, axis=0)
    model_centroid = np.mean(rotated_set, axis=0)
    # Compute point coords relative to their centroids
    scene_prime_points = scene_points - scene_centroid
    model_prime_points = model_points - model_centroid
    # Compute optimal quternion
    Sx = scene_prime_points[:, 0]
    Sy = scene_prime_points[:, 1]
    Sz = scene_prime_points[:, 2]
    Mx = model_prime_points[:, 0]
    My = model_prime_points[:, 1]
    Mz = model_prime_points[:, 2]
    sum_xx = np.sum(Mx * Sx)
    sum_xy = np.sum(Mx * Sy)
    sum_xz = np.sum(Mx * Sz)
    sum_yx = np.sum(My * Sx)
    sum_yy = np.sum(My * Sy)
    sum_yz = np.sum(My * Sz)
    sum_zx = np.sum(Mz * Sx)
    sum_zy = np.sum(Mz * Sy)
    sum_zz = np.sum(Mz * Sz)

    N = np.array(
    [
     [ sum_xx + sum_yy + sum_zz, sum_yz - sum_zy         , -sum_xz + sum_zx         , sum_xy - sum_yz         ],
     [-sum_zy + sum_yz         , sum_xx - sum_zz - sum_yy,  sum_xy + sum_yx         , sum_xz + sum_zx         ],
     [ sum_zx - sum_xz         , sum_yx + sum_xy         ,  sum_yy - sum_zz - sum_xx, sum_yz + sum_zy         ],
     [-sum_yz + sum_xy         , sum_zx + sum_xz         ,  sum_zy + sum_yz         , sum_zz - sum_yy - sum_xx]
    ])

    ei_values, ei_vector = np.linalg.eig(N)

    '''
    Stop here
    Find largest positive eigen value(it's) -> eigen vector
    ''' 

    # print (N, "\n", ei_values, "\n", ei_vector)


def icp_loop(scene_points, model_points):

  indices_of_closest_points = []

  for model_point in model_points:
    # Find closest point in scene_points
    indices_of_closest_points.append(closest_point(model_point, scene_points))


  align(scene_points, model_points)
  plot_closest_points(base_set, rotated_set, indices_of_closest_points)

icp_loop(base_set, rotated_set)

# plt.text(-10.0, 10.0, "relative_variance: " + str(relative_variance), ha="left")
# plt.text(-10.0, 9.0, "covariance: " + str(covariance), ha="left")
plt.plot(base_set[:, 0], base_set[:, 1])
plt.plot(rotated_set[:, 0], rotated_set[:, 1])
# plt.plot(base_centroid[0], base_centroid[1], 'bo')
# plt.plot(rotated_centroid[0], rotated_centroid[1], 'bo')
plt.grid(True)
plt.show()