import torch
from torch import nn
import torch.nn.functional as F
from geoopt import Stiefel, ManifoldParameter, Euclidean
from grassman import Grassman # geoopt doesn't have grassman so I do myself
# from manifolds import EuclideanMod
# from pymanopt.manifolds import Euclidean
import math
import numpy as np
import argparse
import json

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from geoopt.optim import RiemannianSGD
import torchvision.transforms as Tr
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
import os
import time
import higher

from optimizer import RieSBOstep
# from utils import autograd
from problem_class import Task, split_into_adapt_eval, meta_learning_problem, process_data

# code based on https://github.com/sowmaster/esjacobians/blob/master/meta_learning.py

class CNN(nn.Module):
    def __init__(self, hidden_size, ic=3, ks=3, padding=1):
        super().__init__()

        self.ic = ic
        self.hidden_size = hidden_size
        self.ks = ks
        self.pad = padding
        self.stride = 1

        # self.stiefel = Stiefel(canonical=False)
        self.manifold = Grassman()

        self.conv0_kernel = ManifoldParameter(self.manifold.random(ic*ks*ks, hidden_size//2), manifold=self.manifold)

        self.conv1_kernel = ManifoldParameter(self.manifold.random(hidden_size//2*ks*ks, hidden_size),
                                              manifold=self.manifold)
        self.conv2_kernel = ManifoldParameter(self.manifold.random(hidden_size*ks*ks, hidden_size),
                                              manifold=self.manifold)
        self.conv3_kernel = ManifoldParameter(self.manifold.random(hidden_size*ks*ks, hidden_size),
                                              manifold=self.manifold)
        # self.FC_w = ManifoldParameter(torch.Tensor(14112, 256).uniform_(-0.001, 0.001), manifold=Euclidean(ndim=2))
        # self.FC_b = ManifoldParameter(torch.Tensor(256).uniform_(-0.001, 0.001), manifold=Euclidean(ndim=1))
        self.bn0 = nn.BatchNorm2d(hidden_size//2, momentum=1., affine=False,
                           track_running_stats=False)
        self.bn1 = nn.BatchNorm2d(hidden_size, momentum=1., affine=False,
                                  track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(hidden_size, momentum=1., affine=False,
                                  track_running_stats=False)
        self.bn3 = nn.BatchNorm2d(hidden_size, momentum=1., affine=False,
                                  track_running_stats=False)

    def conv_layer(self, x, conv_param, bn):
        x = F.relu(F.conv2d(x, conv_param, padding=self.pad))
        x = F.max_pool2d(x, 2)
        x = bn(x)
        
        # x = F.conv2d(x, conv_param, padding=self.pad)
        # x = bn(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        return x

    def forward(self, x, hparams):

        conv0_kernel = hparams[0]
        conv1_kernel = hparams[1]
        conv2_kernel = hparams[2]
        conv3_kernel = hparams[3]
        # FC_w = hparams[2]
        # FC_b = hparams[3]

        x = self.conv_layer(x, conv0_kernel.transpose(-1,-2).view(self.hidden_size//2,self.ic,self.ks, self.ks), self.bn0)
        x = self.conv_layer(x, conv1_kernel.transpose(-1,-2).view(self.hidden_size,self.hidden_size//2,self.ks, self.ks), self.bn1)
        x = self.conv_layer(x, conv2_kernel.transpose(-1,-2).view(self.hidden_size, self.hidden_size, self.ks,
                                                                   self.ks), self.bn2)
        x = self.conv_layer(x, conv3_kernel.transpose(-1,-2).view(self.hidden_size, self.hidden_size, self.ks,
                                                                   self.ks), self.bn3)
        x = x.view(x.size(0), -1)
        # x = F.relu(x @ FC_w + FC_b)
        return x

class FC(nn.Module):
    def __init__(self, input_size, num_class):
        super().__init__()
        # self.weight = ManifoldParameter(torch.Tensor(input_size, num_class).uniform_(-0.0001, 0.0001), manifold=EuclideanMod(ndim=2))
        # self.bias = ManifoldParameter(torch.Tensor(num_class).uniform_(-0.0001, 0.0001), manifold=EuclideanMod(ndim=1))
        self.weight = ManifoldParameter(torch.Tensor(input_size, num_class).uniform_(-0.0001, 0.0001))
        self.bias = ManifoldParameter(torch.Tensor(num_class).uniform_(-0.0001, 0.0001))

    def forward(self, x, params):
        weight = params[0]
        bias = params[1]
        return x @ weight + bias


if __name__ == '__main__':
    """ Run this test with:
    python meta_learning_tests/test_meta_learning.py \
        --algorithm="Riemannian"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='miniimagenet', metavar='N',
                        help='omniglot or miniimagenet or fc100')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume from checkpoint')
    parser.add_argument('--save_record', type=bool, default=True, help='whether to save training records')
    parser.add_argument('--ckpt_dir', type=str, default='metalogs', help='path of checkpoint file')
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4, help='meta batch size')
    parser.add_argument('--ways', type=int, default=5, help='num classes in few shot learning')
    parser.add_argument('--shots', type=int, default=5, help='num training shots in few shot learning')
    parser.add_argument('--steps', type=int, default=5000, help='total number of outer steps')
    parser.add_argument('--reg_param', type=float, default=0.5, help='reg param for inner problem')

    parser.add_argument('--eta_x', type=float, default=0.002)
    parser.add_argument('--eta_y', type=float, default=0.02)
    parser.add_argument('--ns_gamma', type=float, default=0.01) # the eta in Neumann series
    parser.add_argument('--ns_iter', type=int, default=50) # the K or Q in Neumann series
    parser.add_argument('--lower_iter', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--algorithm', type=str, default='Riemannian', choices=['Riemannian', 'Euclidean'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--T', type=int, default=30)
    
    parser.add_argument('--n_tasks_train', type=int, default=20000)
    parser.add_argument('--n_tasks_test', type=int, default=200)
    parser.add_argument('--n_tasks_val', type=int, default=200)
    
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=20)

    args = parser.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    run = 1
    eval_interval = args.eval_interval
    log_interval = args.log_interval
    K = args.steps
    stop_k = None  # stop iteration for early stopping. leave to None if not using it

    reg_param = args.reg_param  # reg_param = 0.5
    T_test = args.T

    test_inner_lr = args.eta_y

    loc = locals()
    args.out_file = open(os.path.join(args.ckpt_dir, 'log_ESJ_' + args.dataset + str(run) + '.txt'), 'w')
    string = "+++++++++++++++++++ Arguments ++++++++++++++++++++\n"
    for item, value in args.__dict__.items():
        string += "{}:{}\n".format(item, value)

    args.out_file.write(string + '\n')
    args.out_file.flush()
    print(string + '\n')

    string = ""
    for item, value in loc.items():
        string += "{}:{}\n".format(item, value)

    args.out_file.write(string + '\n')
    args.out_file.flush()
    print(string, '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.seed)
    print(device)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # process data
    # train_tasks, val_tasks, test_tasks = process_data(args)
    train_tasks, val_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
                                                                            train_samples=2*args.shots,
                                                                            train_ways=args.ways,
                                                                            test_samples=2*args.shots,
                                                                            test_ways=args.ways,
                                                                            root=os.path.dirname(os.path.abspath(__file__)) + '/data/MiniImageNet')

    # models and model parameters
    meta_model = CNN(32).to(device)
    # task_model = FC(3200, args.ways).to(device)
    task_model = FC(800, args.ways).to(device)
    hparams = list(meta_model.parameters())
    params = list(task_model.parameters())
    
    # define the problem class
    problem = meta_learning_problem(meta_model, task_model, args)

    # training starts
    start_iter = 0
    total_time = 0
    inner_log_interval = None
    inner_log_interval_test = None
    meta_bsz = args.batch_size
    
    res = {"total_time": [], 
           "train_loss": [],
           "norm_grad": [],
           "val_loss": [],
           "test_loss": [],
           "val_acc": [],
           "test_acc": []}
    
    total_time = 0.0
    running_loss = 0.0
    hgradnorm = 0.0
    for k in range(start_iter, K):
        start_time = time.time()

        val_loss, val_acc = 0, 0
        forward_time, backward_time = 0, 0

        for t_idx in range(meta_bsz):
            
            start_time_task = time.time()
            # sample a training task
            task_data = train_tasks.sample()

            train_input, train_target, test_input, test_target = split_into_adapt_eval(task_data,
                                                shots=args.shots,
                                                ways=args.ways,
                                                device=device)

            data_lower = [train_input, train_target]
            data_upper = [test_input, test_target]
            # single task set up
            # task = Task(reg_param, meta_model, task_model, task_data, batch_size=meta_bsz)
            
            hparams, params, loss_u, _ = RieSBOstep(problem, hparams, params, args,
                                                                        data=[data_lower, data_upper])
            running_loss += loss_u
        
        with torch.no_grad():
            for hparam in hparams:
                egrad = hparam.grad / meta_bsz
                
                if args.algorithm == "Euclidean":
                    # new_hparam = hparam.manifold.projx(hparam - args.eta_x * egrad)
                    new_hparam = hparam - args.eta_x * egrad
                    hgradnorm += torch.linalg.norm(egrad)
                else: # the Riemannian algorithm
                    rgrad = hparam.manifold.egrad2rgrad(hparam, egrad)
                    new_hparam = hparam.manifold.retr(hparam, - args.eta_x * rgrad)
                    hgradnorm += torch.linalg.norm(rgrad)
                
                hparam.copy_(new_hparam)
        
        step_time = time.time() - start_time
        total_time += step_time
        
        # evaluate on test data
        if (k + 1) % eval_interval == 0:
            print(f"iter {k}, step time: {step_time}")
            print(f"          norm_grad: {hgradnorm / eval_interval / meta_bsz}")
            print(f"          Train loss: {running_loss / eval_interval / meta_bsz}")
            
            # params0 = params.detach().clone()
            params0 = [param.detach().clone() for param in params]
            val_losses, val_accs = problem.evaluate(val_tasks, hparams, params0, reg_param,
                                            T_test, args)

            # evals.append((val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std()))
            string = "          Val loss {:.2e} (+/- {:.2e}): Val acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(
                val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std(), len(val_losses))
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)

            test_losses, test_accs = problem.evaluate(test_tasks, hparams, params0, reg_param,
                                              T_test, args)

            # evals.append((test_losses.mean(), test_losses.std(), 100. * test_accs.mean(), 100. * test_accs.std()))

            string = "          Test loss {:.2e} (+/- {:.2e}): Test acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(
                test_losses.mean(), test_losses.std(), 100. * test_accs.mean(), 100. * test_accs.std(),
                len(test_losses))
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)
            
            # recording
            res["total_time"].append(total_time)
            res["train_loss"].append(running_loss / eval_interval / meta_bsz)
            res["norm_grad"].append(hgradnorm.item() / eval_interval / meta_bsz)
            res["val_loss"].append(float(val_losses.mean()))
            res["test_loss"].append(float(test_losses.mean()))
            res["val_acc"].append(float(100. * val_accs.mean()))
            res["test_acc"].append(float(100. * test_accs.mean()))
            
            # for k, v in res.items():
            #     print(f"k: {k}, v type: {type(v[0])}")
            # exit()
            
            running_loss = 0.0
            hgradnorm = 0.0
    
    if args.save_record:
        json_file = "res_" + args.algorithm + ".json"
        with open(json_file, "w") as outfile: 
            json.dump(res, outfile)