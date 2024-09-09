import torch
import time
from torch.autograd.functional import jvp, hvp

def RieSBOstep(problem,hparams,params,args,data):
    data_lower, data_upper = data
    start_time = time.time()
    
    params = [p.detach().clone().requires_grad_(True) for p in params]
    # solve the inner:
    # for ii in range(args.lower_iter):
    for _ in range(50):
        # grad = autograd(loss_lower(hparams, params), params)
        grads = torch.autograd.grad(problem.loss_lower(hparams, params, data=data_lower), params)
        with torch.no_grad():
            for param, grad in zip(params, grads):
                new_param = param - args.eta_y * grad
                param.copy_(new_param)
    
    del grads
    
    # def hvp_func(ppp):
    #     return problem.loss_lower(hparams, ppp, data_lower)
    # ppp = tuple(p.detach().requires_grad_() for p in params)
    # print(hvp_func(ppp))
    # print(hvp(func=hvp_func, inputs=ppp, v=tuple(grads)))
    # exit()
    
    loss_u = problem.loss_upper(hparams, params, data=data_upper)
    grads = torch.autograd.grad(loss_u, hparams)
    hgradnorm = 0.0
    with torch.no_grad():
        for hparam, egrad in zip(hparams, grads):
            rgrad = hparam.manifold.egrad2rgrad(hparam, egrad)
            new_hparam = hparam.manifold.retr(hparam, -args.eta_x * rgrad)
            # new_hparam = hparam - args.eta_x * egrad
            hgradnorm += torch.linalg.norm(rgrad)
            hparam.copy_(new_hparam)
    
    return hparams, params, loss_u.detach(), hgradnorm, time.time() - start_time
    # return hparams, params, loss_u, hgradnorm, step_time