"""
    This file defines functions that optimize the given bilevel stochastic Riemannian optimization 
    problem over a given manifold. Specifically we would need the following properties for the manifold:
    proj: projection onto the tangent space at a given point on the manifold
    retr: the retraction
"""
import time
import numpy as np
from psd_kotz_mle import psd_kotz
import matplotlib.pyplot as plt

def stoc_ttsa(problem, a0, b0, S0, K, T, alpha, beta, flag=0):
    manifold = problem.manifold
    a, b = a0, b0
    S = S0
    fval_record = [0] * K
    norm_record = [0] * K
    time_record = [0.0] * K
    time0 = time.time()
    for k in range(K):
        for _ in range(T):  # inner loop
            S = manifold.retr(S, -alpha * problem.grad_S_g(a, b, S, sample=10))

        # v is the approximate of linear equation grady_grady_g*v = grady_f
        # need to first translate the tensor equation into matrix equation
        # v = problem.get_stoc_v(a, b, S)
        
        # grad_a = problem.grad_alpha_f(alpha, beta, S) - problem.grad_S_grad_alpha_g(a, b, S, v)
        # grad_b = problem.grad_beta_f(alpha, beta, S) - problem.grad_S_grad_beta_g(a, b, S, v)

        # update
        # a = max(min(a - beta * grad_a, problem.d), 1)
        # b = max(min(b - beta * grad_b, problem.d), 1)
        a, b = problem.d / 2, 1

        # recording
        val = problem.mle(a, b, S, sample=10)
        
        # if flag and k % 10 == 0:
        print("iter: %d, fval: %f, ||S - S*||=%f" % (
                    k, val, np.linalg.norm(S - problem.S)))
        print(a, b)
        # approx_data = np.zeros(shape=[problem.d, problem.p, problem.n])
        # for i in range(problem.n):
        #     approx_data[:, :, i] = x.T.dot(problem.data[:, :, i])
        # print("iter: %d, fval: %f, ||\grad\Phi||=%f, ||X - X^*||=%f" % (k, val,
        #                 np.linalg.norm(grad_hat), np.linalg.norm(approx_data - problem.X_star)))
        fval_record[k] = val
        norm_record[k] = np.linalg.norm(S - problem.S)
        time_record[k] = time.time() - time0

    # print(np.sum(y))
    return a, b, S, fval_record, norm_record, time_record


if __name__ == "__main__":
    d = 5
    problem = psd_kotz.problem(d=d)
    S0 = problem.manifold.rand()  # random point on the manifold
    a0 = d / 2
    b0 = 1
    a, b, S, fval, norms, times = stoc_ttsa(problem, a0, b0, S0, K=1000, T=10, alpha=1e-5, beta=1e-5)

    # # plot
    # plt.plot(fval)
    # plt.xlabel("Iterations")
    # plt.ylabel("Function value $\Phi(x_k)$")
    # plt.show()
    # # plt.savefig('DL_function_val_' + str(d) + '_' + str(p) + '_' + str(n) + '.pdf')
    # plt.clf()

    plt.plot(times, fval)
    plt.xlabel("CPU time")
    plt.ylabel("Function value $\Phi(x_k)$")
    plt.show()
    # plt.savefig('DL_time_function_val_' + str(d) + '_' + str(p) + '_' + str(n) + '.pdf')

    plt.plot(times, norms)
    plt.yscale("log")
    plt.xlabel("CPU time")
    plt.ylabel("$||S_k - S*||$")
    plt.show()
    # plt.savefig('DL_time_norm_grad_' + str(d) + '_' + str(p) + '_' + str(n) + '.pdf')