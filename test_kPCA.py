import time
import numpy as np

import kPCA
from DRO_Riemannian_bilevel import bio
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles
from misc import projection_simplex_bisection

d_list = [10, 50, 100]
p_list = [3, 3, 3]
n_list = [10, 10, 10]
alpha_list = [1e-4, 5e-5, 1e-5]
beta_list = [1e-4, 5e-5, 1e-5]
rep = 1
K = 1000
inner_iter = 30

fig1 = plt.figure()
ax1 = fig1.gca()
fig2 = plt.figure()
ax2 = fig2.gca()
fig3 = plt.figure()
ax3 = fig3.gca()

for t in range(3):
    d, p, n = d_list[t], p_list[t], n_list[t]
    fval_record = np.array([0.0] * K)
    norm_record = np.array([0.0] * K)
    time_record = np.array([0.0] * K)
    angle_record = np.array([0.0] * K)
    alpha = alpha_list[t]
    beta = beta_list[t]
    # alpha, beta = 5e-5, 5e-5
    print("Test on (d, p, n)=(%d, %d, %d)" % (d, p, n))

    problem = kPCA.problem(d=d, p=p, n=n, lam=0.0)
    for re in range(rep):
        print("Repitition %d" % re)
        x0 = problem.manifold.rand()  # random point on the manifold
        y0 = np.ones(problem.n) / problem.n
        v0 = np.ones(problem.n) / problem.n
        x, y, fval, norms, times, angles = bio(problem, x0, y0, K=K, T=inner_iter, alpha=alpha, beta=beta, flag=1)
        fval_record += fval
        norm_record += norms
        time_record += times
        angle_record += angles

    fval_record = fval_record / rep
    norm_record = norm_record / rep
    time_record = time_record / rep
    angle_record = angle_record / rep

    ax1.plot(time_record, fval_record, label='('+str(d)+', '+str(p)+', '+str(n)+')')
    ax2.plot(time_record, norm_record, label='('+str(d)+', '+str(p)+', '+str(n)+')')
    ax3.plot(time_record, angle_record, label='('+str(d)+', '+str(p)+', '+str(n)+')')

ax1.set_xlabel("CPU time")
ax1.set_ylabel("Stochastic function value")
ax1.legend()
fig1.show()
fig1.savefig('kpca_plots/kPCA_time_function_val.pdf')

ax2.set_yscale("log")
ax2.set_xlabel("CPU time")
ax2.set_ylabel("Norm of grad mapping")
ax2.legend()
fig2.show()
fig2.savefig('kpca_plots/kPCA_time_norm_grad.pdf')

ax3.set_xlabel("CPU time")
ax3.set_ylabel("Principal angles")
ax3.legend()
fig3.show()
fig3.savefig('kpca_plots/kPCA_time_angle.pdf')