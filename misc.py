import numpy as np

def projection_simplex_bisection(v, z=1, tau=1e-6, max_iter=1000, epsilon=0.0):
    """
    projection onto probability simplex

    :param v: the vector
    :param z: sum of entries in the simplex
    :param tau: tolerance
    :param max_iter: maximum iteration
    :return:
    """
    func = lambda x: np.sum(np.maximum(v - x, epsilon)) - z
    lower = np.min(v) - z / len(v)
    upper = np.max(v)

    for it in range(max_iter):
        midpoint = (upper + lower) / 2.0
        value = func(midpoint)

        if abs(value) <= tau:
            break

        if value <= 0:
            upper = midpoint
        else:
            lower = midpoint

    return np.maximum(v - midpoint, epsilon)