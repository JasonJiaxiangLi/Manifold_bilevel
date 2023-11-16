"""
Modeling the MLE problem for kotz type distributions
"""
import manifolds
import numpy as np
import random

def symm(A):
    return (A + A.T) / 2

class problem(object):
    """
    This is the class for the bilevel objective
    All functions are stochastic
    """
    def __init__(self, d, S=None):
        """
        S is ground truth parameter
        alpha and beta are taken so that the true distribution is just 
        Gaussian with mean 0 and covariance S
        """
        # These are ground truth
        if S is None:
            temp = np.random.randn(d, d)
            S = temp.dot(temp.T)
        self.d, self.alpha, self.beta, self.S = d, d / 2, 1, S
        self.manifold = manifolds.psd.PositiveDefinite(d)

    def mle(self, alpha, beta, S, data=None, sample=1):
        # the loss function, maximum likelihood estimator
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, sample)
        n, _ = data.shape
        invS = np.linalg.inv(S)
        temp = [x.dot(invS).dot(x) for x in data]
        return n / 2 * np.log(np.linalg.det(S)) + \
                (self.d / 2 - alpha) * np.sum([np.log(t) for t in temp]) + \
                np.sum([t ** beta / 2 for t in temp])

    # stochastic function values
    def fval(self, alpha, beta, S, data=None):
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, 1)
        return self.mle(alpha, beta, S, data)

    def gval(self, alpha, beta, S, data=None):
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, 1)
        return self.mle(alpha, beta, S, data) + self.manifold.dist(S, np.eye(self.d)) ** 2 / 2

    def get_y(self, X):
        raise NotImplementedError

    def Phi_val(self, X):
        # not able to get it
        # return self.fval(X, self.get_y(X))
        raise NotImplementedError

    # stochastic Riemannian gradient
    def grad_alpha_f(self, alpha, beta, S, data=None):
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, 1)
        invS = np.linalg.inv(S)
        return -sum([np.log(x.dot(invS).dot(x)) for x in data])
    
    def grad_beta_f(self, alpha, beta, S, data=None):
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, 1)
        invS = np.linalg.inv(S)
        return sum([(x.dot(invS).dot(x) / 2)**beta * np.log(x.dot(invS).dot(x)) for x in data])

    def nabla_S_f(self, alpha, beta, S, data=None):
        # output is d by d
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, 1)
        n, _ = data.shape
        invS = np.linalg.inv(S)
        return n / 2 * invS + np.sum([(- (self.d / 2 - alpha) / (x.dot(invS).dot(x))  + 1/2 * (x.dot(invS).dot(x) / 2)**(beta - 1)) * \
                invS.dot(x.reshape((self.d, 1))).dot(x.reshape((1, self.d))).dot(invS) for x in data])

    def grad_S_f(self, alpha, beta, S, data=None, sample=1):
        # output is d by d
        # return self.manifold.egrad2rgrad(S, self.nabla_S_f(alpha, beta, S, data))
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, sample)
        n, _ = data.shape
        invS = np.linalg.inv(S)
        return n / 2 * S + np.sum([(- (self.d / 2 - alpha) / (x.dot(invS).dot(x)) - 1/2 * (x.dot(invS).dot(x) / 2)**(beta - 1)) * \
                x.reshape((self.d, 1)).dot(x.reshape((1, self.d))) for x in data])

    def grad_alpha_g(self, alpha, beta, S, data=None):
        return self.grad_alpha_f(alpha, beta, S, data=data)
    
    def grad_beta_g(self, alpha, beta, S, data=None):
        return self.grad_beta_f(alpha, beta, S, data=data)

    def grad_S_g(self, alpha, beta, S, data=None, sample=1):  # output is d by d
        return self.grad_S_f(alpha, beta, S, data=data, sample=sample)

    # Hessian vector product
    def H_s(self, alpha, beta, S, V, data=None):
        # output is d by d
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, 1)
        n, _ = data.shape
        invS = np.linalg.inv(S)
        temp1 = - n / 2 * V + np.sum([(self.d/2 - alpha)/(x.dot(invS).dot(x)) * \
                                   symm(V.dot(invS).dot(x.reshape((self.d, 1)).dot(x.reshape((1, self.d))))) - \
                                   (self.d/2 - alpha) * (x.dot(invS.dot(V).dot(invS)).dot(x)) /(x.dot(invS).dot(x)) *\
                                   x.reshape((self.d, 1)).dot(x.reshape((1, self.d))) + \
                                   1/2 * (x.dot(invS).dot(x) / 2) ** (beta - 2) * (x.dot(invS.dot(V).dot(invS)).dot(x)) *\
                                   x.reshape((self.d, 1)).dot(x.reshape((1, self.d))) +\
                                   1/2 * (x.dot(invS).dot(x) / 2) ** (beta - 1) *\
                                   symm(V.dot(invS).dot(x.reshape((self.d, 1)).dot(x.reshape((1, self.d)))))
                                for x in data])
        temp2 = n / 2 * np.eye(self.d) + [(- (self.d / 2 - alpha) / (x.dot(invS).dot(x))  + 1/2 * (x.dot(invS).dot(x) / 2)**(beta - 1)) * \
                (x.reshape((self.d, 1))).dot(x.reshape((1, self.d))).dot(invS).dot(V) for x in data]
        temp2 = symm(temp2)
        return temp1 + temp2

    # the following are the cross (Riemannian) derivative for g function
    def grad_S_grad_alpha_g(self, alpha, beta, S, V, data=None):
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, 1)
        n, _ = data.shape
        invS = np.linalg.inv(S)
        return np.sum([(x.dot(invS.dot(V).dot(invS)).dot(x)) /(x.dot(invS).dot(x)) for x in data])

    def grad_S_grad_beta_g(self, alpha, beta, S, V, data=None):
        if data is None:
            data = np.random.multivariate_normal(np.zeros(self.d), self.S, 1)
        n, _ = data.shape
        invS = np.linalg.inv(S)
        return np.sum([-(1/2*np.log((x.dot(invS).dot(x))/2) + 1) * (x.dot(invS).dot(x) / 2)**(beta-1) *\
                    (x.dot(invS.dot(V).dot(invS)).dot(x)) for x in data])

    def get_stoc_v(self, alpha, beta, S, Q=10, eta=0.01):
        Qp = random.randint(1, Q)
        res = self.grad_S_f(alpha, beta, S)
        for _ in range(Qp):
            res = res - eta * self.H_s(alpha, beta, S, res)
        return eta * Q * res
