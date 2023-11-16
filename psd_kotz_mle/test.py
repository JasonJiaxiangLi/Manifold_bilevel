import numpy as np

mean = [0, 0]
cov = [[1, 0], [0, 1000]]
x = np.random.multivariate_normal(mean, cov, 3)
print(x)