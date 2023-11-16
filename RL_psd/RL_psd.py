"""
    Modeling the following PSD policy evaluation problem:
    \min_{\Sigma} F(y^*(\Sigma))=1/2*\|y^*(\Sigma)\|^2
    s.t. y^*(\Sigma)\in\argmin_y 1/2*\|y - G(\Sigma)\|^2
"""
import manifolds
from scipy.stats import multivariate_normal
import numpy as np

# def inv_2d(A):
#     """
#     invert a 2x2 matrix
#     :param A: a given non-singular 2x2 matrix
#     :return:
#     """
#     return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])


class problem(object):
    """
    This is the class for the bilevel objective for reinforcement learning experiment
    """

    def __init__(self, d, s, D, rho=0.1, model_data=None, data=None):
        """

        :param d: d is the dimension, in our experiment d = 2
        :param s: s^d is the number of states (we have an s times s times ... times s grid)
        :param D: number of the Gaussian models
        :param rho: discount factor
        :param model_data: include transition probability P and true reward r
        :param data: omega, mu, Sigma, i.e. param of Gaussian model
        """
        self.d = d
        self.s = s
        self.D = D
        self.rho = rho
        self.manifold = manifolds.psd.PositiveDefinite(d, k=D)
        if model_data:
            self.P, self.r, self.S = model_data
        else:
            self.P = np.zeros((s, s, s, s))
            for i in range(s):
                for j in range(s):
                    temp = np.random.uniform(size=(s, s))
                    self.P[i, j] = temp / np.sum(temp)
            self.r = np.random.uniform(size=(s, s, s, s))
            self.Sx = np.random.normal(size=s)  # randomly generate state space as grids
            self.Sy = np.random.normal(size=s)

        if data:
            self.omega, self.mu, self.Sigma = data
        else:  # create data on our own
            temp = np.random.uniform(size=D)
            self.omega = temp / np.sum(temp)
            self.mu = np.random.normal(size=(D, d))
            self.Sigma = self.manifold.rand()  # Sigma is of size (D, d, d)
            # self.Sigma = self.Sigma.reshape((self.D, self.d, self.d))
            print("model created with D=%d, S=%d" % (D, s))

    def fval(self, Sigma, y):
        return np.linalg.norm(y, 'fro') ** 2 / 2

    def phi_val(self, x, Sigma):
        output = 0
        for i in range(self.D):
            output += self.omega[i] * multivariate_normal(mean=self.mu[i], cov=Sigma[i]).pdf(x)
        return output

    def Gval(self, Sigma):
        output = np.zeros((self.s, self.s))
        phi_vec = np.zeros((self.s, self.s))
        for i in range(self.s):
            for j in range(self.s):
                phi_vec[i, j] = self.phi_val([self.Sx[i], self.Sy[j]], Sigma)

        for i in range(self.s):
            for j in range(self.s):
                output[i, j] += phi_vec[i, j]

                for p in range(self.s):
                    for q in range(self.s):
                        output[i, j] -= self.P[i, j, p, q] * (self.r[i, j, p, q] + self.rho * phi_vec[p, q])
        return output  # s by s

    def gval(self, Sigma, y):
        return np.linalg.norm(y - self.Gval(Sigma), 'fro') ** 2 / 2

    def get_y(self, Sigma):
        """
        The exact solution for lower level problem
        """
        return self.Gval(Sigma)

    def Phi_val(self, Sigma):
        return self.fval(Sigma, self.get_y(Sigma))

    # Riemannian gradient
    # y should be s by s
    def gradx_f(self, Sigma, y):
        return np.zeros((self.D, self.d, self.d))

    def grady_f(self, Sigma, y):  # output is s by s
        return y

    def grady_g(self, Sigma, y):
        return y - self.Gval(Sigma)

    # Hessian vector product
    def grady_grady_g(self, Sigma, y, v):
        return v

    # the following is the second order (Riemannian) derivative for g function
    def gradx_grady_g(self, Sigma, y, v):  # output is D by d by d
        G = np.zeros((self.s, self.s, self.D, self.d, self.d))
        # calculate all the inverse first
        inv_Sigma = np.zeros((self.D, self.d, self.d))
        for i in range(self.D):
            inv_Sigma[i] = np.linalg.inv(Sigma[i])
            # inv_Sigma[i] = inv_2d(Sigma[i])

        # calculate the Euclidean gradient
        grads = np.zeros((self.s, self.s, self.d, self.d))
        for i in range(self.D):

            for p in range(self.s):
                for q in range(self.s):
                    x = np.array([self.Sx[p], self.Sy[q]])
                    temp = (x - self.mu[i]).reshape((self.d, 1))
                    grads[p, q] = self.omega[i] * multivariate_normal(mean=self.mu[i], cov=Sigma[i]).pdf(x) * (
                                          - inv_Sigma[i] + inv_Sigma[i].dot(temp.dot(temp.T)).dot(inv_Sigma[i])) / 2

            for p in range(self.s):
                for q in range(self.s):
                    # calculate the gradient of g_s w.r.t. Sigma_i
                    G[p, q, i] += grads[p, q]

                    for pp in range(self.s):
                        for qq in range(self.s):
                            G[p, q, i] -= self.P[p, q, pp, qq] * self.rho * grads[pp, qq]

        # calculate the Riemannian-Jacobian vector product
        output = np.zeros((self.D, self.d, self.d))
        for p in range(self.s):
            for q in range(self.s):
                G[p, q] = self.manifold.proj(Sigma, G[p, q])
                output += v[p, q] * G[p, q]

        return -output
