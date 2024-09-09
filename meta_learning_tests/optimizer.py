import torch
import time
import random
from torch.autograd import grad as torch_grad
from torch import Tensor
from torch.autograd.functional import jvp, hvp
from typing import List, Callable

def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g

def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))

def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams

def cat_list_to_tensor(list_tx):
    # return torch.cat([xx.view([-1]) for xx in list_tx])
    return torch.cat([xx.reshape([-1]) for xx in list_tx])

def neumann(params: List[Tensor],
            hparams: List[Tensor],
            K: int ,
            fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
            outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
            tol=1e-10) -> List[Tensor]:
    """ Saves one iteration from the fixed point method"""
    params = [w.detach().clone().requires_grad_(True) for w in params]
    # hparams = [w.detach().clone().requires_grad_(True) for w in hparams]
    o_loss = outer_loss(hparams, params)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    w_mapped = fp_map(params, hparams)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    # w_mapped_cp = [p.detach() for p in w_mapped]
    # K = random.randint(1, K)
    for _ in range(K):
        gs_prev_vec = gs_vec
        vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        # gs = [v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(torch.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = torch_grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    # grads = [K * g + v for g, v in zip(grads, grad_outer_hparams)]
    return grads


def RieSBOstep(problem,hparams,params,args,data):
    data_lower, data_upper = data
    start_time = time.time()
    
    params = [p.detach().clone().requires_grad_(True) for p in params]
    # solve the inner:
    for ii in range(args.lower_iter):
        # grad = autograd(loss_lower(hparams, params), params)
        loss_l = problem.loss_lower(hparams, params, data=data_lower)
        # print(f"loss_l: {loss_l}")
        grads = torch.autograd.grad(loss_l, params)
        with torch.no_grad():
            for param, grad in zip(params, grads):
                new_param = param - args.eta_y * grad
                param.copy_(new_param)
                
    # print()
    
    def fp_map(params, hparams):
        # params = [p.detach().clone().requires_grad_(True) for p in params]
        # hparams = [w.detach().clone().requires_grad_(True) for w in hparams]
        loss = problem.loss_lower(hparams, params, data=data_lower)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        return [param - args.ns_gamma * grad for param, grad in zip(params, grads)]
    
    def outer_loss(hparams, params):
        return problem.loss_upper(hparams, params, data=data_upper)
        
    loss_u = problem.loss_upper(hparams, params, data=data_upper)
    
    # grads = torch.autograd.grad(loss_u, hparams)
    grads = neumann(params, hparams, K=args.ns_iter, fp_map=fp_map, outer_loss=outer_loss)
    
    update_tensor_grads(hparams, grads)
    
    return hparams, params, loss_u.detach(), time.time() - start_time
    # return hparams, params, loss_u, hgradnorm, step_time