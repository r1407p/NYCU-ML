import matplotlib.pyplot as plt
import numpy as np
import math
points = np.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])


x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y)
#doraw a line with slope 2 and intercept 0
l = 5
scalar = points.dot(np.array([1, 2]))
# initial a np array with 5 [1,2]
arr = np.array([[1, 2]] * 5)
#sclar multiply with the np array
print(scalar)
print(scalar[:, np.newaxis])
projection = scalar[:, np.newaxis] * arr/5
print(arr)
print(projection)
plt.plot([0, l], [0, l*2])
plt.scatter(projection[:, 0], projection[:, 1])
#projection points to the line 
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()