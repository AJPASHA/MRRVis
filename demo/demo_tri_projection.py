from mrrvis import ConfigurationGraph, plot_square_config

import mrrvis as mvs
import numpy as np
import matplotlib.pyplot as plt

arr = np.array([v for _,v in mvs.cells.Tri.adjacent_transformations('edge', [0,0,0]).items()])
arr = np.append(arr, np.array([[0,0,0]]),axis=0)
arr[:,1] = -arr[:,1]
print(arr)

arr_transformed = mvs.geometry_utils.isometric(arr)

plt.scatter(arr_transformed[:,0], arr_transformed[:,1])
plt.show()