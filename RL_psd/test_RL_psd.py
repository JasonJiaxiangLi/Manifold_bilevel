"""
    This file defines functions that optimize the given bilevel Riemannian optimization problem
    over a given manifold. Specifically we would need the following properties for the manifold:
    proj: projection onto the tangent space at a given point on the manifold
    retr: the retraction

    This file only deal with the deterministic situation
"""
import time
import numpy as np

import RL_psd
import matplotlib.pyplot as plt

def nearest_psd(X, thre=1e-2):
    n, d, _ = X.shape
    for i in range(n):
        Xi = X[i].reshape((d, d))
        S, V = np.linalg.eig(Xi)
        S = np.maximum(S, thre)
        X[i] = V.dot(np.diag(S)).dot(V.T)
    return X


def BiO_AID(problem, x0, y0, K, D, N, alpha, beta):
    """
    The code for manifold bilevel optimization with AID

    :param problem:
    :param x0:
    :param y0:
    :param K:
    :param D:
    :param N:
    :param alpha:
    :param beta:
    :return:
    """
    manifold = problem.manifold
    x, y = x0, y0
    fval_record = [0] * K
    norm_record = [0] * K
    time_record = [0.0] * K
    time0 = time.time()
    for k in range(K):
        # for t in range(D):  # inner loop
        #     y = projection_simplex_bisection(y - alpha * problem.grady_g(x, y))
        # # closed form solution for y
        y = problem.get_y(x)

        # # AID calculation: solving the linear equation
        # # v is the solution of linear equation grady_grady_g*v = grady_f
        # # need to first translate the tensor equation into matrix equation
        v = problem.grady_f(x, y)  # since we can solve this by closed form
        # v = np.linalg.solve(problem.grady_grady_g(x,y), problem.grady_f(x,y))
        grad_hat = problem.gradx_f(x, y) - problem.gradx_grady_g(x, y, v)

        # # update
        x = manifold.retr(x, - beta * grad_hat)

        # # A safeguard function to make sure the matrix not ill conditioned
        x = nearest_psd(x)

        # # recording
        val = problem.Phi_val(x)
        # val = problem.fval(x, y)
        # print("iter: %d, fval: %f, ||\grad\Phi||=%f" % (
        #                 k, val, np.linalg.norm(grad_hat)))
        # approx_data = np.zeros(shape=[problem.d, problem.p, problem.n])
        # for i in range(problem.n):
        #     approx_data[:, :, i] = x.T.dot(problem.data[:, :, i])
        if k % 100 == 0:
            print("iter: %d, fval: %f, ||\grad\Phi||=%f" % (k, val, np.linalg.norm(grad_hat)))
            # print("          eigvals: " + str(np.linalg.eigvals(x[0])) + str(np.linalg.eigvals(x[1])) )
        fval_record[k] = val
        norm_record[k] = np.linalg.norm(grad_hat)
        time_record[k] = time.time() - time0

    # print(np.sum(y))
    return x, fval_record, norm_record, time_record


if __name__ == "__main__":
    rep = 10
    d = 2
    s = 3
    D = 2
    K = 10000

    fval_record = np.array([0.0] * K)
    norm_record = np.array([0.0] * K)
    time_record = np.array([0.0] * K)

    for re in range(rep):
        print("repitition: %d" % (re))
        problem = RL_psd.problem(d=d, s=s, D=D)
        x0 = problem.manifold.rand()  # random point on the manifold
        # x0 = x0.reshape((D, d, d))
        # x0 = problem.Sigma
        y0 = np.ones((problem.s, problem.s)) / (problem.s * problem.s)
        x, fvals, norms, times = BiO_AID(problem, x0, y0, K=K, D=10, N=problem.s * problem.s, alpha=1e-2, beta=1e-2)
        fval_record += fvals
        norm_record += norms
        time_record += times

    fval_record = fval_record / rep
    norm_record = norm_record / rep
    time_record = time_record / rep

    # # plot
    # plt.plot(fval)
    # plt.xlabel("Iterations")
    # plt.ylabel("Function value $\Phi(x_k)$")
    # plt.show()
    # # plt.savefig('RL_function_val_' + str(s) + '_' + str(D) + '.pdf')
    # plt.clf()

    plt.plot(time_record, fval_record)
    plt.xlabel("CPU time")
    plt.ylabel("Function value $\Phi(x_k)$")
    # plt.show()
    plt.savefig('RL_time_function_val_' + str(s) + '_' + str(D) + '.pdf')
    plt.clf()

    plt.plot(time_record, norm_record)
    plt.yscale("log")
    plt.xlabel("CPU time")
    plt.ylabel("Norm of $\operatorname{grad}\Phi(x_k)$")
    # plt.show()
    plt.savefig('RL_time_norm_grad_' + str(s) + '_' + str(D) + '.pdf')