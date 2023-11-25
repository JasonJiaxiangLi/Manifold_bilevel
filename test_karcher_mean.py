import time
import numpy as np

import Karcher_mean
from DRO_Riemannian_bilevel import bio
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles
from misc import projection_simplex_bisection

d_list = [10, 20, 30]
n_list = [5, 5, 5]
alpha_list = [1e-2, 5e-3, 1e-3]
beta_list = [1e-1, 5e-2, 1e-2]
rep = 1
K = 200
inner_iter = 200

fig1 = plt.figure()
ax1 = fig1.gca()
fig2 = plt.figure()
ax2 = fig2.gca()
fig3 = plt.figure()
ax3 = fig3.gca()

for t in range(3):
    d, n = d_list[t], n_list[t]
    fval_record = np.array([0.0] * K)
    norm_record = np.array([0.0] * K)
    time_record = np.array([0.0] * K)
    alpha = alpha_list[t]
    beta = beta_list[t]
    print("Test on (d, n)=(%d, %d)" % (d, n))

    problem = Karcher_mean.problem(d=d, n=n, lam=1)
    for re in range(rep):
        print("Repitition %d" % re)
        x0 = problem.manifold.rand()  # random point on the manifold
        y0 = np.ones(problem.n) / problem.n
        v0 = np.ones(problem.n) / problem.n
        x, y, fval, norms, times = bio(problem, x0, y0, K=K, T=inner_iter, alpha=alpha, beta=beta, flag=1)
        fval_record += fval
        norm_record += norms
        time_record += times

    fval_record = fval_record / rep
    norm_record = norm_record / rep
    time_record = time_record / rep

    ax1.plot(time_record, fval_record, label='('+str(d)+', '+str(n)+')')
    ax2.plot(time_record, norm_record, label='('+str(d)+', '+str(n)+')')

ax1.set_xlabel("CPU time")
ax1.set_ylabel("Stochastic function value")
ax1.legend()
fig1.show()
fig1.savefig('karcher_plots/karcher_time_function_val.pdf')

ax2.set_yscale("log")
ax2.set_xlabel("CPU time")
ax2.set_ylabel("Norm of grad mapping")
ax2.legend()
fig2.show()
fig2.savefig('karcher_plots/karcher_time_norm_grad.pdf')