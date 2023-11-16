"""
    Solves the following robust dictionary learning problem:
    min_{A^\top A=I_d}  \sum_{i=1}^{n}p_i h_{\mu}(A^\top Y_i)
    s.t. p\in\argmin_{p\in\Delta_n} \lambda * \|p - \ones / n \|^{2} - \sum_{i=1}^{n} p_{i} h_{\mu}(A^\top Y_i)
"""
import manifolds
import numpy as np
from misc import projection_simplex_bisection

class problem(object):
    """
    This is the class for the bilevel objective
    """
    def __init__(self, d, p, n, theta=0.25, mu=1e-1, data=None, lam = 1):
        """

        :param data: the sparse data matrix, should be d x p x n
                    if no data then will generate the data by Bernoulli-Gaussian
        """
        self.d, self.p, self.n = d, p, n
        self.mu = mu
        self.ones = np.ones(n)
        self.manifold = manifolds.rotations.Rotations(d)
        self.A_star = self.manifold.rand()  # randomly generate the optimum
        if not data:
            data = np.zeros(shape=[d, p, n])
            X_star = np.zeros(shape=[d, p, n])
            for i in range(n):
                X_star[:, :, i] = np.multiply(np.random.binomial(size=[d, p], n=1, p=theta), np.random.randn(d, p))
                data[:, :, i] = self.A_star.dot(X_star[:, :, i])
            self.X_star = X_star
        self.data = data  # data are the Y_i's

    def h_mu(self, X):
        # approximate np.sum(abs(X))
        mu = self.mu
        return mu * np.sum(np.log(np.cosh(X / mu)))

    def fval(self, X, y):
        return np.sum([y[i] * self.h_mu(X.T.dot(self.data[:, :, i])) for i in range(self.n)])

    def gval(self, X, y):
        return - np.sum([y[i] * self.h_mu(X.T.dot(self.data[:, :, i])) for i in range(self.n)]) + np.linalg.norm(y - self.ones / self.n) ** 2

    def get_y(self, X):
        return projection_simplex_bisection(np.array([self.h_mu(X.T.dot(self.data[:, :, i])) for i in range(self.n)])/2 + self.ones / self.n)

    def Phi_val(self, X):
        return self.fval(X, self.get_y(X))

    # Riemannian gradient
    def gradx_f(self, X, y):
        egradx = np.zeros(shape=[self.d, self.d])
        for i in range(self.n):
            egradx += y[i] * np.tanh(X.T.dot(self.data[:, :, i]) / self.mu).dot(self.data[:, :, i].T)
        return self.manifold.proj(X, egradx)  # output is d by p

    def grady_f(self, X, y):  # output is n by 1
        return np.array([self.h_mu(X.T.dot(self.data[:, :, i])) for i in range(self.n)])

    def gradx_g(self, X, y):
        return -self.gradx_f(X, y)

    def grady_g(self, X, y):
        return -np.array([self.h_mu(X.T.dot(self.data[:, :, i])) for i in range(self.n)]) + 2*(y - self.ones / self.n)

    # second order derivative
    def grady_grady_g(self, X, y):
        return 2 * np.eye(self.n)

    # the following is the second order (Riemannian) derivative for f function
    def gradx_grady_f(self, X, y):  # output is d x d x n
        output = np.zeros((self.d, self.d, self.n))
        for i in range(self.n):
            output[:, :, i] = self.manifold.proj(X, np.tanh(X.T.dot(self.data[:, :, i]) / self.mu).dot(self.data[:, :, i].T))
        return output