import numpy as np
import DL_sphere as dl
import matplotlib.pyplot as plt
from DRO_Riemannian_bilevel import BiO_AID

d_list = [10, 20, 50, 100]
p_list = [5, 10, 15, 20]
n_list = [10, 20, 30, 40]
beta_list = [6e-3, 2e-3, 2e-4, 1e-4]
rep = 10
K = 5000
inner_iter = 10

fig1 = plt.figure()
ax1 = fig1.gca()
fig2 = plt.figure()
ax2 = fig2.gca()

for t in range(4):
    d, p, n = d_list[t], p_list[t], n_list[t]
    fval_record = np.array([0.0] * K)
    norm_record = np.array([0.0] * K)
    time_record = np.array([0.0] * K)
    beta = beta_list[t]
    print("Test on (d, p, n)=(%d, %d, %d)" % (d, p, n))

    problem = dl.problem(d=d, p=p, n=n, lam=1)
    for re in range(rep):
        print("Repitition %d" % re)
        x0 = problem.manifold.rand()  # random point on the manifold
        y0 = np.ones(problem.n) / problem.n
        v0 = np.ones(problem.n) / problem.n
        x, fval, norms, times = BiO_AID(problem, x0, y0, K=K, D=inner_iter, N=problem.n, alpha=1e-3, beta=beta)
        fval_record += fval
        norm_record += norms
        time_record += times
    fval_record = fval_record / rep
    norm_record = norm_record / rep
    time_record = time_record / rep

    ax1.plot(time_record, fval_record, label='('+str(d)+', '+str(p)+', '+str(n)+')')
    ax2.plot(time_record, norm_record, label='('+str(d)+', '+str(p)+', '+str(n)+')')

ax1.set_xlabel("CPU time")
ax1.set_ylabel("Function value $\Phi(x_k)$")
ax1.legend()
fig1.show()
# fig1.savefig('DL_time_function_val.pdf')

ax2.set_yscale("log")
ax2.set_xlabel("CPU time")
ax2.set_ylabel("Norm of $\operatorname{grad}\Phi(x_k)$")
ax2.legend()
fig2.show()
# fig2.savefig('DL_time_norm_grad.pdf')