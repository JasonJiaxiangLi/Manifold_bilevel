import torch
from torch import nn
import torch.nn.functional as F
from geoopt import Stiefel, ManifoldParameter, Euclidean
# from manifolds import EuclideanMod
# from pymanopt.manifolds import Euclidean
import math
import numpy as np
import argparse

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from geoopt.optim import RiemannianSGD
import torchvision.transforms as Tr
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
import os

import time

from optimizer import RieSBOstep
# from utils import autograd
from problem_class import Task, split_into_adapt_eval, meta_learning_problem

import higher


# code based on https://github.com/sowmaster/esjacobians/blob/master/meta_learning.py


def process_data(args):
    MEAN = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    STD = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
    normalize = Tr.Normalize(mean=MEAN, std=STD)

    # use the same data-augmentation as in lee et al.
    transform_train = Tr.Compose([
        # Tr.ToPILImage(),
        # Tr.RandomCrop(84, padding=8),
        # Tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # Tr.RandomHorizontalFlip(),
        # Tr.ToTensor(),
        normalize
    ])

    transform_test = Tr.Compose([
        normalize
    ])

    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=os.path.dirname(os.path.abspath(__file__)) + '/data/MiniImageNet',
        mode='train',
        transform=transform_train,
        download=True)
    # print('got train dataset...')
    val_dataset = l2l.vision.datasets.MiniImagenet(
        root=os.path.dirname(os.path.abspath(__file__)) + '/data/MiniImageNet',
        mode='validation',
        transform=transform_test,
        download=True)
    # print('got val dataset...')
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=os.path.dirname(os.path.abspath(__file__)) + '/data/MiniImageNet',
        mode='test',
        transform=transform_test,
        download=True)

    train_dataset = l2l.data.MetaDataset(train_dataset)
    val_dataset = l2l.data.MetaDataset(val_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [FusedNWaysKShots(train_dataset, n=args.ways, k=2 * args.shots),
                        LoadData(train_dataset),
                        RemapLabels(train_dataset),
                        ConsecutiveLabels(train_dataset)]

    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=args.n_tasks_train)

    val_transforms = [FusedNWaysKShots(val_dataset, n=args.ways, k=2 * args.shots),
                      LoadData(val_dataset),
                      ConsecutiveLabels(val_dataset),
                      RemapLabels(val_dataset)]

    val_tasks = l2l.data.TaskDataset(val_dataset, task_transforms=val_transforms, num_tasks=args.n_tasks_val)

    test_transforms = [FusedNWaysKShots(test_dataset, n=args.ways, k=2 * args.shots),
                       LoadData(test_dataset),
                       RemapLabels(test_dataset),
                       ConsecutiveLabels(test_dataset)]

    test_tasks = l2l.data.TaskDataset(test_dataset, task_transforms=test_transforms, num_tasks=args.n_tasks_test)

    return train_tasks, val_tasks, test_tasks


class CNN(nn.Module):
    def __init__(self, hidden_size, ic=3, ks=3, padding=1):
        super().__init__()

        self.ic = ic
        self.hidden_size = hidden_size
        self.ks = ks
        self.pad = padding
        self.stride = 1

        self.stiefel = Stiefel(canonical=False)

        self.conv0_kernel = ManifoldParameter(self.stiefel.random(ic*ks*ks, hidden_size//2), manifold=self.stiefel)

        self.conv1_kernel = ManifoldParameter(self.stiefel.random(hidden_size//2*ks*ks, hidden_size),
                                              manifold=self.stiefel)
        self.conv2_kernel = ManifoldParameter(self.stiefel.random(hidden_size*ks*ks, hidden_size),
                                              manifold=self.stiefel)
        self.conv3_kernel = ManifoldParameter(self.stiefel.random(hidden_size*ks*ks, hidden_size),
                                              manifold=self.stiefel)
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
        # x = F.relu(F.conv2d(x, conv_param,padding=self.pad))
        # x = F.max_pool2d(x, 2)
        # x = bn(x)
        
        x = F.conv2d(x, conv_param, padding=self.pad)
        x = bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='miniimagenet', metavar='N',
                        help='omniglot or miniimagenet or fc100')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume from checkpoint')
    parser.add_argument('--ckpt_dir', type=str, default='metalogs', help='path of checkpoint file')
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4, help='meta batch size')
    parser.add_argument('--ways', type=int, default=5, help='num classes in few shot learning')
    parser.add_argument('--shots', type=int, default=5, help='num training shots in few shot learning')
    parser.add_argument('--steps', type=int, default=10000, help='total number of outer steps')
    parser.add_argument('--reg_param', type=float, default=0.5, help='reg param for inner problem')

    parser.add_argument('--eta_x', type=float, default=0.0005)
    parser.add_argument('--eta_y', type=float, default=0.01)
    parser.add_argument('--ns_gamma', type=float, default=0.01) # the eta in Neumann series
    parser.add_argument('--ns_iter', type=int, default=30) # the K or Q in Neumann series
    parser.add_argument('--lower_iter', type=int, default=15)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--hygrad_opt', type=str, default='ns', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--T', type=int, default=30)
    
    parser.add_argument('--n_tasks_train', type=int, default=20000)
    parser.add_argument('--n_tasks_test', type=int, default=200)
    parser.add_argument('--n_tasks_val', type=int, default=200)
    
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=50)

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

    meta_model = CNN(32).to(device)
    # meta_model = NewCNN(32).to(device)
    # task_model = FC(3200, args.ways).to(device)
    task_model = FC(800, args.ways).to(device)
    
    # define the problem class
    problem = meta_learning_problem(meta_model, task_model, args)

    # training starts
    start_iter = 0
    total_time = 0

    run_time, accs, vals, evals = [], [], [], []

    hparams = list(meta_model.parameters())
    params = list(task_model.parameters())

    inner_log_interval = None
    inner_log_interval_test = None

    meta_bsz = args.batch_size
    
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
            
            hparams, params, loss_u, step_time = RieSBOstep(problem, hparams, params, args,
                                                                        data=[data_lower, data_upper])
            running_loss += loss_u
        
        with torch.no_grad():
            for hparam in hparams:
                egrad = hparam.grad / meta_bsz
                new_hparam = hparam - args.eta_x * egrad
                hgradnorm += torch.linalg.norm(egrad)
                
                # rgrad = hparam.manifold.egrad2rgrad(hparam, egrad)
                # new_hparam = hparam.manifold.retr(hparam, - args.eta_x * rgrad)
                # hgradnorm += torch.linalg.norm(rgrad)
                
                hparam.copy_(new_hparam)
        
        if (k + 1) % eval_interval == 0:
            print(f"iter {k}, step time: {step_time}")
            print(f"          norm_grad: {hgradnorm / eval_interval / meta_bsz}")
            print(f"          Train loss: {running_loss / eval_interval / meta_bsz}")
            running_loss = 0.0
            hgradnorm = 0.0

        run_time.append(total_time)
        vals.append(val_loss)  # this is actually train loss in few-shot learning
        accs.append(val_acc)  # this is actually train accuracy in few-shot learning

        # evaluate on test data
        if (k + 1) % eval_interval == 0:
            # params0 = params.detach().clone()
            params0 = [param.detach().clone() for param in params]
            # val_losses, val_accs = evaluate(val_tasks, meta_model, task_model, hparams, params0, reg_param,
            #                                 test_inner_lr, args)
            val_losses, val_accs = problem.evaluate(val_tasks, hparams, params0, reg_param,
                                            T_test, args)

            evals.append((val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std()))
            string = "          Val loss {:.2e} (+/- {:.2e}): Val acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(
                val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std(), len(val_losses))
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)

            # test_losses, test_accs = evaluate(test_tasks, meta_model, task_model, hparams, params0, reg_param,
            #                                   test_inner_lr, args)
            test_losses, test_accs = problem.evaluate(test_tasks, hparams, params0, reg_param,
                                              T_test, args)

            evals.append((test_losses.mean(), test_losses.std(), 100. * test_accs.mean(), 100. * test_accs.std()))

            string = "          Test loss {:.2e} (+/- {:.2e}): Test acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(
                test_losses.mean(), test_losses.std(), 100. * test_accs.mean(), 100. * test_accs.std(),
                len(test_losses))
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)