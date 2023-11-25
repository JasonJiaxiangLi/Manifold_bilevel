"""
    Modeling the following robust kPCA problem:
    
    \min -\sum_{i} y_{i} \trace(X^\top H_i X) + \lambda * \|y-\ones/n\|^2
    s.t. \min \sum_{i} y_{i} \trace(X^\top H_i X)
    The matrices H_i are constructed such that H_i=z_i*z_i^\top
    where z_i is standard Gaussian random vector
"""
import manifolds
import random
import numpy as np

class problem(object):
    """
    This is the class for the bilevel objective
    """
    def __init__(self, d, p, n, data=None, lam=1):
        """

        :param d: dimension of X\in\RR^{d x p}
        :param p: dimension of X\in\RR^{d x p}
        :param n: number of samples, also the dimension of weight vector y
        :param data: the data matrix, if no data then will generate the data by Gaussian
        """
        self.d, self.p, self.n = d, p, n
        self.ones = np.ones(n)
        # self.manifold = manifolds.grassmann.Grassmann(d, p)
        self.manifold = manifolds.stiefel.Stiefel(d, p)
        if not data:
            data = np.random.randn(n, d)  # each row vector is a data
        self.data = data
        self.lam = lam
        self.name = "kpca"

        # compute the group truth
        A = data.T.dot(data)
        eigenValues, eigenVectors = np.linalg.eig(A)

        idx = eigenValues.argsort()[::-1]
        self.X_star = eigenVectors[:,idx[:p]]

    def fval(self, X, y):
        return sum([y[i] * self.data[i].dot(X.dot(X.T)).dot(self.data[i]) for i in range(self.n)]) + \
               self.lam * np.linalg.norm(y - self.ones / self.n) ** 2

    def gval(self, X, y):
        return sum([-y[i] * self.data[i].dot(X.dot(X.T)).dot(self.data[i]) for i in range(self.n)])

    def get_X(self, y, inner_iter=100, beta=0.01):
        X = self.manifold.rand()
        for _ in range(inner_iter):  # inner loop
            if np.linalg.norm(problem.gradx_g(X, y))<=1e-8:
                break
            X = self.manifold.retr(X, -beta * problem.gradx_g(X, y))
        return X

    def Phi_val(self, y):
        return self.fval(self.get_X(y), y)

    # Riemannian gradient
    def gradx_g(self, X, y):
        # output is d by p
        egradx = np.zeros_like(X)
        for i in range(self.n):
            egradx += -2 * y[i] * self.data[i].reshape(self.d, 1).dot((self.data[i].dot(X)).reshape(1, self.p))
        # egradx = sum([-2 * y[i] * self.data[i].reshape(self.d, 1).dot((self.data[i].dot(X)).reshape(1, self.p)) for i in range(self.n)])
        return self.manifold.proj(X, egradx)

    def grady_g(self, X, y):  
        # output is n by 1
        return np.array([- self.data[i].dot(X.dot(X.T)).dot(self.data[i]) for i in range(self.n)])

    def gradx_f(self, X, y):
        return -self.gradx_g(X, y)

    def grady_f(self, X, y):
        return np.array([self.data[i].dot(X.dot(X.T)).dot(self.data[i]) for i in range(self.n)]) + self.lam * 2 * (y - self.ones / self.n)

    # Hessian vector product
    def gradx_gradx_g(self, X, y, v):
        # output is d by p
        v = self.manifold.proj(X, v)
        egradx = sum([-2 * y[i] * self.data[i].reshape(self.d, 1).dot((self.data[i].dot(X)).reshape(1, self.p)) for i in range(self.n)])
        ehessx = sum([-2 * y[i] * self.data[i].reshape(self.d, 1).dot((self.data[i].dot(v)).reshape(1, self.p)) for i in range(self.n)])
        return self.manifold.ehess2rhess(X, egradx, ehessx, v)

    # the following is the Riemannian cross-derivative for g function
    def grady_gradx_g(self, X, y, v):
        # output is n by 1
        v = self.manifold.proj(X, v)
        return np.array([-2 * self.data[i].dot(X.dot(v.T)).dot(self.data[i]) for i in range(self.n)])
    
    def get_stoc_v(self, X, y, Q=10, eta=0.01):
        Qp = random.randint(1, Q)
        res = self.gradx_f(X, y)
        for _ in range(Qp):
            res = res - eta * self.gradx_gradx_g(X, y, res)
        return eta * Q * res