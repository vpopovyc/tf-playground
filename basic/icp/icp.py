import numpy as np
import matplotlib.pyplot as plt

base_set = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0], [4.0, 4.0, 0.0], [5.0, 5.0, 0.0], [6.0, 6.0, 0.0], [7.0, 7.0, 0.0], [8.0, 8.0, 0.0], [9.0, 9.0, 0.0], [10.0, 10.0, 0.0]], np.float32)

angle = -45
theta = angle * np.pi / 180.0
rot_matrix = np.array([
                       [np.cos(theta), -np.sin(theta), 0], 
                       [np.sin(theta),  np.cos(theta), 0],
                       [0            ,  0            , 1]
                      ])

for point in base_set:
  point[1] += 4
  # point[1] += np.random.randint(4, high=8)

rotated_set = np.array(base_set) 

for i, point in enumerate(rotated_set):
  rotated_set[i] = np.matmul(rot_matrix, point)

def closest_point(point, scene_points):
  min_distance = np.inf
  closest_point = [0.0, 0.0, 0.0]
  for scene_point in scene_points:
      distance = np.sqrt(np.square(np.sum(point - scene_point)))
      if min_distance > distance:
          min_distance = distance
          closest_point = scene_point

  assert np.isinf(min_distance) == False
  return closest_point

def l2_norm(x):
    x = np.sqrt(np.sum(np.square(x)))

def compute_mse(set):
  set = np.sum(set, axis=0)  
  return np.sqrt(np.dot(np.transpose(set), set))

def find_rts(scene_points, model_points, translation_only):

  # Find centroids of two point sets
  scene_centroid = np.mean(scene_points, axis=0)
  model_centroid = np.mean(model_points, axis=0)
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

  print("EI values\n", ei_values, "\nEI vector\n", ei_vector)

  quat = ei_vector[:, 0]

  # q = np.array(
  # [
  #  [quat[3], -quat[0], -quat[1], -quat[2]],
  #  [quat[0],  quat[3], -quat[2],  quat[1]],
  #  [quat[1],  quat[2],  quat[3], -quat[0]],
  #  [quat[2], -quat[1],  quat[0],  quat[3]]
  # ])

  # q_bar = np.array(
  # [
  #  [quat[3], -quat[0], -quat[1], -quat[2]],
  #  [quat[0],  quat[3],  quat[2], -quat[1]],
  #  [quat[1], -quat[2],  quat[3],  quat[0]],
  #  [quat[2],  quat[1], -quat[0],  quat[3]]
  # ])

  q = np.array(
  [
   [quat[0], -quat[1], -quat[2], -quat[3]],
   [quat[1],  quat[0], -quat[3],  quat[2]],
   [quat[2],  quat[3],  quat[0], -quat[1]],
   [quat[3], -quat[2],  quat[1],  quat[0]]
  ])

  q_bar = np.array(
  [
   [quat[0], -quat[1], -quat[2], -quat[3]],
   [quat[1],  quat[0],  quat[3], -quat[2]],
   [quat[2], -quat[3],  quat[0],  quat[1]],
   [quat[3],  quat[2], -quat[1],  quat[0]]
  ])

  rotation = np.matmul(np.transpose(q_bar), q)[1:4, 1:4]

  for i, point in enumerate(model_prime_points):
    model_prime_points[i] = np.matmul(rotation, point)

  '''
             Σ||y'||^2 
  Scale => ( ––––––––– ) ^ -1
             Σ||p'||^2
  '''
  sum_sp = np.sum(np.dot(np.transpose(scene_prime_points), scene_prime_points))
  # print("Sum sp", sum_sp)
  sum_mp = np.sum(np.dot(np.transpose(model_prime_points), model_prime_points))
  # print("Sum mp", sum_mp)

  scale = np.sqrt(sum_sp / sum_mp)

  if translation_only:
    print("Scene centroid: ", scene_centroid, "Model centroid: ", model_centroid)
    translation = scene_centroid - model_centroid
  else:
    translation = scene_centroid - np.matmul(rotation, model_centroid)

  print("Translation\n", translation, "\nRotation\n", rotation, "\nScale", scale)

  return rotation, scale, translation


def icp_loop(scene_points, model_points, translation_only=False, rotation_only=False):

  closest_points = []

  for model_point in model_points:
    # Find closest point in scene_points
    closest_points.append(closest_point(model_point, scene_points))

  rotation, scale, translation = find_rts(closest_points, model_points, translation_only)

  new_model_points = []

  if rotation_only:
    for point in model_points:
      new_model_points.append(np.matmul(rotation, point))
  elif translation_only:
    for point in model_points:
      new_model_points.append(point + translation)
  else:
    for point in model_points:
      new_model_points.append(np.matmul(rotation, point) + translation)


  new_model_points = np.array(new_model_points)

  loss = compute_mse(scene_points - new_model_points)

  return new_model_points, loss

if __name__ == "__main__":
  new_rotated_set = icp(base_set, rotated_set)

  plt.plot(base_set[:, 0], base_set[:, 1])
  plt.plot(rotated_set[:, 0], rotated_set[:, 1])
  plt.plot(new_rotated_set[:, 0], new_rotated_set[:, 1], linestyle='dashed')
  # plt.plot(base_centroid[0], base_centroid[1], 'bo')
  # plt.plot(rotated_centroid[0], rotated_centroid[1], 'bo')
  plt.grid(True)
  plt.show()