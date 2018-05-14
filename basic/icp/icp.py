import numpy as np
import matplotlib.pyplot as plt

base_set = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0], [9.0, 9.0], [10.0, 10.0]], np.float32)
rotated_set = base_set * -1.0

base_centroid = np.mean(base_set, axis=0)
rotated_centroid = np.mean(rotated_set, axis=0)

variance = np.sum(np.square((base_set-base_centroid)), axis=0)/(base_set.size-1)

relative_variance = variance/base_centroid

covariance = np.mean((base_set-base_centroid)*(rotated_set-rotated_centroid), axis=0)

plt.text(-10.0, 10.0, "relative_variance: " + str(relative_variance), ha="left")
plt.text(-10.0, 9.0, "covariance: " + str(covariance), ha="left")
plt.plot(base_set, linestyle='dashed')
plt.plot(rotated_set[:, 0], rotated_set[:, 1], linestyle='dashed')
plt.plot(base_centroid[0], base_centroid[1], 'bo')
plt.plot(rotated_centroid[0], rotated_centroid[1], 'bo')
plt.grid(True)
plt.show()