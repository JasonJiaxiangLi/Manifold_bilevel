"""
    Modeling the following robust dictionary learning problem:
    min_{A^\top A=I_d}  \sum_{i=1}^{n}p_i h_{\mu}(x^\top Y_i)
    s.t. p\in\argmin_{p\in\Delta_n} \lambda * \|p - \ones / n \|^{2} - \sum_{i=1}^{n} p_{i} h_{\mu}(x^\top Y_i)
"""
import manifolds
import numpy as np
from misc import projection_simplex_bisection

class problem(object):
    """
    This is the class for the bilevel objective
    """
    def __init__(self, d, p, n, theta=0.2, mu=1e-1, data=None, lam = 1):
        """

        :param data: the sparse data matrix, should be d x p x n
                    if no data then will generate the data by Bernoulli-Gaussian
        """
        self.d, self.p, self.n = d, p, n
        self.mu = mu
        self.ones = np.ones(n)
        self.manifold0 = manifolds.rotations.Rotations(d)
        self.manifold = manifolds.sphere.Sphere(d)
        self.A_star = self.manifold0.rand()  # randomly generate the optimum
        if not data:
            data = np.zeros(shape=[d, p, n])
            X_star = np.zeros(shape=[d, p, n])
            for i in range(n):
                X_star[:, :, i] = np.multiply(np.random.binomial(size=[d, p], n=1, p=theta), np.random.randn(d, p))
                data[:, :, i] = self.A_star.dot(X_star[:, :, i])
            self.X_star = X_star
        self.data = data  # data are the Y_i's
        self.lam = lam

    def h_mu(self, X):
        # approximate np.sum(abs(X))
        mu = self.mu
        return mu * np.sum(np.log(np.cosh(X / mu)))

    def fval(self, x, y):
        return np.sum([y[i] * self.h_mu(x.T.dot(self.data[:, :, i])) for i in range(self.n)])

    def gval(self, x, y):
        return - np.sum([y[i] * self.h_mu(x.T.dot(self.data[:, :, i])) for i in range(self.n)]) + self.lam * np.linalg.norm(y - self.ones / self.n) ** 2

    def get_y(self, x):
        return projection_simplex_bisection(np.array([self.h_mu(x.T.dot(self.data[:, :, i])) for i in range(self.n)])/(2 * self.lam) + self.ones / self.n)

    def Phi_val(self, x):
        return self.fval(x, self.get_y(x))

    # Riemannian gradient
    def gradx_f(self, x, y):
        egradx = np.zeros(self.d)
        for i in range(self.n):
            egradx += y[i] * np.tanh(x.T.dot(self.data[:, :, i]) / self.mu).dot(self.data[:, :, i].T)
        return self.manifold.proj(x, egradx)  # output is d by 1

    def grady_f(self, x, y):  # output is n by 1
        return np.array([self.h_mu(x.T.dot(self.data[:, :, i])) for i in range(self.n)])

    def gradx_g(self, x, y):
        return -self.gradx_f(x, y)

    def grady_g(self, x, y):
        return -np.array([self.h_mu(x.T.dot(self.data[:, :, i])) for i in range(self.n)]) +self.lam* 2*(y - self.ones / self.n)

    # Hessian vector product
    def grady_grady_g(self, x, y, v):
        return 2 * v

    # the following is the second order (Riemannian) derivative for g function
    def gradx_grady_g(self, x, y, v):  # output is d x n
        output = np.zeros((self.d))
        for i in range(self.n):
            a = self.data[:, :, i]
            output += self.manifold.proj(x, -np.tanh(x.T.dot(a) / self.mu).dot(a.T)) * v[i]
        return output