"""
    This file defines functions that optimize the given bilevel Riemannian optimization problem
    over a given manifold. Specifically we would need the following properties for the manifold:
    proj: projection onto the tangent space at a given point on the manifold
    retr: the retraction

    This file only deal with the deterministic situation
"""
import time
import numpy as np
import kPCA
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles
from misc import projection_simplex_bisection

def bio(problem, x0, y0, K, T, alpha, beta, flag=0):
    manifold = problem.manifold
    x, y = x0, y0
    fval_record = [0] * K
    norm_record = [0] * K
    time_record = [0.0] * K
    if problem.name == "kpca":
        angle_record = [0.0] * K
    elif problem.name == "robust_mle":
        dist_record = [0.0] * K

    time0 = time.time()
    for k in range(K):
        for _ in range(T):  # inner loop
            x = manifold.retr(x, -beta * problem.gradx_g(x, y))

        # AID calculation: solving the linear equation
        # v is the solution of linear equation grady_grady_g*v = grady_f
        # need to first translate the tensor equation into matrix equation
        v = problem.get_stoc_v(x, y, Q=5, eta=alpha)
        # v = np.linalg.solve(problem.grady_grady_g(x,y), problem.grady_f(x,y))
        grad_hat = problem.grady_f(x, y) - problem.grady_gradx_g(x, y, v)

        grad_map = 1 / alpha * (y - projection_simplex_bisection(y - alpha * grad_hat))

        # update
        y = y - alpha * grad_map
        # y = (1 - alpha) * y + alpha * problem.get_y(x)

        # recording
        time_record[k] = time.time() - time0
        
        if problem.name == "robust_mle":
            val = problem.Phi_val(y)
        else:
            val = problem.fval(x, y)

        if flag and k % 10 == 0:
            print("iter: %d, fval: %f, ||grad map||=%f" % (
                        k, val, np.linalg.norm(grad_map)))
            # print(y)
        fval_record[k] = val
        norm_record[k] = np.linalg.norm(grad_map)
        if problem.name == "kpca":
            # angle_record[k] = manifold.dist(x, problem.X_star)
            angle_record[k] = sum(subspace_angles(x, problem.X_star))
        elif problem.name == "robust_mle":
            dist_record[k] = np.linalg.norm(x - problem.S)

    if problem.name == "kpca":
        return x, y, fval_record, norm_record, time_record, angle_record
    elif problem.name == "robust_mle":
        return x, y, fval_record, norm_record, time_record, dist_record
    else:
        return x, y, fval_record, norm_record, time_record


if __name__ == "__main__":
    problem = kPCA.problem(d=50, p=5, n=10, lam=1)
    x0 = problem.manifold.rand()  # random point on the manifold
    # x0.reshape(problem.d, problem.p)
    y0 = np.ones(problem.n) / problem.n
    x, y, fval, norms, times, angle = bio(problem, x0, y0, K=2000, T=50, alpha=5e-5, beta=1e-5, flag=1)

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
    plt.ylabel("Norm of the gradient map")
    plt.show()
    # plt.savefig('DL_time_norm_grad_' + str(d) + '_' + str(p) + '_' + str(n) + '.pdf')

    plt.plot(times, angle)
    # plt.yscale("log")
    plt.xlabel("CPU time")
    plt.ylabel("Principal angles")
    plt.show()