"""
    Modeling the following robust Karcher mean problem:
    
    \min -\sum_{i} y_{i} d(S, A_i)^2 + \lambda * \|y-\ones/n\|^2
    s.t. \min \sum_{i} y_{i} d(S, A_i)^2
    d(S, A_i)^2 is the geodesic distance to matrix A_i
"""
import time
import manifolds
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
from misc import projection_simplex_bisection

def symm(A):
    return (A + A.T) / 2

class problem(object):
    """
    This is the class for the bilevel objective
    """
    def __init__(self, d, n, data=None, lam=1):
        """

        :param d: dimension of S\in\RR^{d x d}
        :param n: number of samples, also the dimension of weight vector y
        :param data: the data matrix, if no data then will generate the data by Gaussian
        """
        self.d, self.n = d, n
        self.ones = np.ones(n)
        self.manifold = manifolds.psd.PositiveDefinite(d)
        if not data:
            data = [self.manifold.rand() for _ in range(n)]
        self.data = data
        print(f"Create Karcher mean problem with (d, n)={d, n}")
        
        self.lam = lam
        self.name = "karcher_mean"

    def sqdist(self, A, B, inv_sqrt_A=None):
        if inv_sqrt_A is None:
            inv_sqrt_A = np.linalg.inv(scipy.linalg.sqrtm(A))
        return np.linalg.norm(inv_sqrt_A.dot(B).dot(inv_sqrt_A))

    def fval(self, S, y):
        inv_sqrt_S = np.linalg.inv(scipy.linalg.sqrtm(S))
        return sum([- y[i] * self.sqdist(S, self.data[i], inv_sqrt_S) for i in range(self.n)]) + \
               self.lam * np.linalg.norm(y - self.ones / self.n) ** 2

    # TODO: continue here
    def gval(self, S, y):
        return sum([y[i] * self.mle(S, self.data[i]) for i in range(self.n)])

    def get_s(self, y):
        return sum([y[i] * self.data[i].reshape(self.d, 1).dot(self.data[i].reshape(1, self.d)) for i in range(self.n)])
    
    def get_y(self, S):
        temp = [self.ones[i] / self.n - self.mle(S, self.data[i])/(2*self.lam) for i in range(self.n)]
        return projection_simplex_bisection(temp)

    def Phi_val(self, y):
        return self.fval(self.get_s(y), y)

    # Riemannian gradient
    def gradx_g(self, S, y):
        # output is d by d
        res = np.zeros_like(S)
        for i in range(self.n):
            x = self.data[i]
            res += y[i] / 2 * (S - x.reshape((self.d, 1)).dot(x.reshape((1, self.d))))
        return res

    def grady_g(self, S, y):  
        # output is n by 1
        return np.array([self.mle(S, self.data[i]) for i in range(self.n)])

    def gradx_f(self, S, y):
        return -self.gradx_g(S, y)

    def grady_f(self, S, y):
        return np.array([- self.mle(S, self.data[i]) for i in range(self.n)]) + self.lam * 2 * (y - self.ones / self.n)

    # Hessian vector product
    def gradx_gradx_g(self, S, y, v):
        # output is d by d
        invS = np.linalg.inv(S)
        return sum([y[i] * (1/2 * symm(v.dot(invS).dot(self.data[i].reshape(self.d, 1)).dot(self.data[i].reshape(1, self.d))))\
                     for i in range(self.n)])

    # the following is the Riemannian cross-derivative for g function
    def grady_gradx_g(self, S, y, v):  
        # output is n by 1
        v = self.manifold.proj(S, v)
        return np.array([1/2 * (np.trace(S.dot(v)) - self.data[i].dot(v).dot(self.data[i])) for i in range(self.n)])
    
    def get_stoc_v(self, X, y, Q=10, eta=0.01):
        Qp = random.randint(1, Q)
        res = self.gradx_f(X, y)
        for _ in range(Qp):
            res = res - eta * self.gradx_gradx_g(X, y, res)
        return eta * Q * res
    
if __name__=="__main__":
    K, T = 1000, 50
    alpha, beta = 1e-5, 1e-5
    prob = problem(d=15, n=30, lam=1e2)
    mani = prob.manifold
    x = mani.rand()
    y = np.ones(prob.n) / prob.n

    fval_record = [0.0] * K
    norm_record = [0.0] * K
    time_record = [0.0] * K
    dist_record = [0.0] * K

    time0 = time.time()
    for k in range(K):
        # for _ in range(T):  # inner loop
        #     x = mani.retr(x, -beta * prob.gradx_g(x, y))
        x = prob.get_s(y)

        # AID calculation: solving the linear equation
        # v is the solution of linear equation grady_grady_g*v = grady_f
        # need to first translate the tensor equation into matrix equation
        v = prob.get_stoc_v(x, y, Q=30, eta=alpha)
        # v = np.linalg.solve(prob.grady_grady_g(x,y), prob.grady_f(x,y))
        grad_hat = prob.grady_f(x, y) - prob.grady_gradx_g(x, y, v)

        grad_map = 1 / alpha * (y - projection_simplex_bisection(y - alpha * grad_hat))

        # # update
        y = y - alpha * grad_map

        # recording
        time_record[k] = time.time() - time0
        
        if prob.name == "kpca":
            val = prob.fval(x, y)
        elif prob.name == "robust_mle":
            val = prob.Phi_val(y)
            
        if k % 10 == 0:
            print("iter: %d, fval: %f, dist: %f" % (k, val, np.linalg.norm(x - prob.S)))
            # print(np.sum(y))
            # print(y)
        fval_record[k] = val
        # norm_record[k] = np.linalg.norm(grad_map)
        dist_record[k] = np.linalg.norm(x - prob.S)
    
    plt.plot(time_record, fval_record)
    plt.xlabel("CPU time")
    plt.ylabel("Function value $\Phi(x_k)$")
    plt.show()
    # plt.savefig('DL_time_function_val_' + str(d) + '_' + str(p) + '_' + str(n) + '.pdf')

    plt.plot(time_record, norm_record)
    plt.yscale("log")
    plt.xlabel("CPU time")
    plt.ylabel("Norm of the gradient map")
    plt.show()
    # plt.savefig('DL_time_norm_grad_' + str(d) + '_' + str(p) + '_' + str(n) + '.pdf')

    plt.plot(time_record, dist_record)
    # plt.yscale("log")
    plt.xlabel("CPU time")
    plt.ylabel("Dist to optimal")
    plt.show()