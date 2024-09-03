import torch
from torch import nn
import torch.nn.functional as F
from geoopt import Stiefel, ManifoldParameter, Euclidean
from manifolds import EuclideanMod
import math
import numpy as np
import argparse

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# from geoopt.optim import RiemannianSGD
import torchvision.transforms as Tr
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
import os

import time

from optimizer import RHGDstep
from utils import autograd

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
        root='data/MiniImageNet',
        mode='train',
        transform=transform_train,
        download=True)
    # print('got train dataset...')
    val_dataset = l2l.vision.datasets.MiniImagenet(
        root='data/MiniImageNet',
        mode='validation',
        transform=transform_test,
        download=True)
    # print('got val dataset...')
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root='data/MiniImageNet',
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

    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=n_tasks_train)

    val_transforms = [FusedNWaysKShots(val_dataset, n=args.ways, k=2 * args.shots),
                      LoadData(val_dataset),
                      ConsecutiveLabels(val_dataset),
                      RemapLabels(val_dataset)]

    val_tasks = l2l.data.TaskDataset(val_dataset, task_transforms=val_transforms, num_tasks=n_tasks_val)

    test_transforms = [FusedNWaysKShots(test_dataset, n=args.ways, k=2 * args.shots),
                       LoadData(test_dataset),
                       RemapLabels(test_dataset),
                       ConsecutiveLabels(test_dataset)]

    test_tasks = l2l.data.TaskDataset(test_dataset, task_transforms=test_transforms, num_tasks=n_tasks_test)

    return train_tasks, val_tasks, test_tasks


def MiniimageNetFeats(hidden_size):
    def conv_layer(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=False
                           )
        )

    net = nn.Sequential(
        conv_layer(3, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten())

    #initialize(net)
    return net



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

        self.conv1_kernel = ManifoldParameter(self.stiefel.random(hidden_size//2*ks*ks,hidden_size),
                                              manifold=self.stiefel)
        self.conv2_kernel = ManifoldParameter(self.stiefel.random(hidden_size*ks*ks,hidden_size),
                                              manifold=self.stiefel)
        # self.FC_w = ManifoldParameter(torch.Tensor(14112, 256).uniform_(-0.001, 0.001), manifold=Euclidean(ndim=2))
        # self.FC_b = ManifoldParameter(torch.Tensor(256).uniform_(-0.001, 0.001), manifold=Euclidean(ndim=1))
        self.bn0 = nn.BatchNorm2d(hidden_size//2, momentum=1., affine=False,
                           track_running_stats=False)
        self.bn1 = nn.BatchNorm2d(hidden_size, momentum=1., affine=False,
                                  track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(hidden_size, momentum=1., affine=False,
                                  track_running_stats=False)

    def conv_layer(self, x, conv_param, bn):
        x = F.relu(F.conv2d(x, conv_param,padding=self.pad), inplace=True)
        # x = F.batch_norm(x, running_mean=torch.zeros(self.hidden_size).to(device),
        #                  running_var=torch.ones(self.hidden_size).to(device),
        #                  weight=bn_w, bias=bn_b, training=self.training,
        #                  momentum=0.1, eps=1e-5)
        x = F.max_pool2d(x, 2)
        x = bn(x)
        return x

    def forward(self, x, hparams):

        conv0_kernel = hparams[0]
        conv1_kernel = hparams[1]
        conv2_kernel = hparams[2]
        # FC_w = hparams[2]
        # FC_b = hparams[3]

        x = self.conv_layer(x, conv0_kernel.transpose(-1,-2).view(self.hidden_size//2,self.ic,self.ks, self.ks), self.bn0)
        x = self.conv_layer(x, conv1_kernel.transpose(-1,-2).view(self.hidden_size,self.hidden_size//2,self.ks, self.ks), self.bn1)
        x = self.conv_layer(x, conv2_kernel.transpose(-1,-2).view(self.hidden_size, self.hidden_size, self.ks,
                                                                   self.ks), self.bn2)
        x = x.view(x.size(0), -1)
        # x = F.relu(x @ FC_w + FC_b)
        return x


class FC(nn.Module):
    def __init__(self, input_size, num_class):
        super().__init__()
        self.weight = ManifoldParameter(torch.Tensor(input_size, num_class).uniform_(-0.0001, 0.0001), manifold=EuclideanMod(ndim=2))
        self.bias = ManifoldParameter(torch.Tensor(num_class).uniform_(-0.0001, 0.0001), manifold=EuclideanMod(ndim=1))

    def forward(self, x, params):
        weight = params[0]
        bias = params[1]
        return x @ weight + bias


def split_into_adapt_eval(batch,
               shots,
               ways,
               device=None):

    # Splits task data into adaptation/evaluation sets

    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    adapt_idx = np.zeros(data.size(0), dtype=bool)
    adapt_idx[np.arange(shots * ways) * 2] = True

    eval_idx = torch.from_numpy(~adapt_idx)
    adapt_idx = torch.from_numpy(adapt_idx)
    adapt_data, adapt_labels = data[adapt_idx], labels[adapt_idx]
    eval_data, eval_labels = data[eval_idx], labels[eval_idx]

    return adapt_data, adapt_labels, eval_data, eval_labels



class Task:
    """
    Handles the train and validation loss for a single task
    """
    def __init__(self, reg_param, meta_model, task_model, data, batch_size=None): # here batchsize = number of tasks used at each step. we will do full GD for each task
        device = next(meta_model.parameters()).device

        # stateless version of meta_model
        self.fmeta = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)
        self.ftask = higher.monkeypatch(task_model, device=device, copy_initial_weights=True)
        # self.fmeta = meta_model.to(device)
        # self.ftask = task_model.to(device)

        #self.n_params = len(list(meta_model.parameters()))
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None

    def compute_feats(self, hparams):
        # compute train feats
        self.train_feats = self.fmeta(self.train_input, params= hparams)

    def reg_f(self, params):
        # l2 regularization
        return sum([(p ** 2).sum() for p in params])

    def train_loss_f(self, params):
        # regularized cross-entropy loss
        out = self.ftask(self.train_feats, params=params)
        return F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.reg_f(params)

    def val_loss_f(self, params, hparams):
        # cross-entropy loss (uses only the task-specific weights in params
        feats = self.fmeta(self.test_input, params=hparams)
        out = self.ftask(feats, params=params)
        val_loss = F.cross_entropy(out, self.test_target)/self.batch_size
        self.val_loss = val_loss.item()  # avoid memory leaks

        with torch.no_grad():
            pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss









# def inner_solver(task, hparams, params, steps, optim, params0=None, log_interval=None):
#
#     if params0 is not None:
#         for param, param0 in zip(params, params0):
#             param.data = param0.data
#
#     task.compute_feats(hparams) # compute feats only once to make inner iterations lighter (only linear transformations!)
#
#     for t in range(steps):
#         loss = task.train_loss_f(params)
#         grads = torch.autograd.grad(loss, params)
#
#
#         if log_interval and (t % log_interval==0 or t==steps-1):
#             print('Inner step t={}, Loss: {:.6f}'.format(t, loss.item()))
#
#     return [param.detach().clone() for param in params]


# def update_tensor_grads(params, grads):
#     for l, g in zip(params, grads):
#         if l.grad is None:
#             l.grad = torch.zeros_like(l)
#         if g is not None:
#             l.grad += g



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='miniimagenet', metavar='N',
                        help='omniglot or miniimagenet or fc100')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume from checkpoint')
    parser.add_argument('--ckpt_dir', type=str, default='metalogs', help='path of checkpoint file')
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16, help='meta batch size')
    parser.add_argument('--ways', type=int, default=5, help='num classes in few shot learning')
    parser.add_argument('--shots', type=int, default=5, help='num training shots in few shot learning')
    parser.add_argument('--steps', type=int, default=10000, help='total number of outer steps')
    parser.add_argument('--reg_param', type=float, default=0.5, help='reg param for inner problem')

    parser.add_argument('--eta_x', type=float, default=0.001)
    parser.add_argument('--eta_y', type=float, default=0.005)
    parser.add_argument('--lower_iter', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--hygrad_opt', type=str, default='cg', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--ns_gamma', type=float, default=0.01)
    parser.add_argument('--ns_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)


    def loss_lower(hparams, params, data):
        train_input, train_target = data
        # task.compute_feats(hparams)
        # loss = task.train_loss_f(params)
        feats = meta_model(train_input, hparams)
        out = task_model(feats, params)
        loss = F.cross_entropy(out, train_target) + 0.5 * reg_param * sum([(p ** 2).sum() for p in params])/len(params)
        return loss

    def loss_upper(hparams, params, data):
        test_input, test_target = data
        # loss = task.val_loss_f(params, hparams)
        feats = meta_model(test_input, hparams)
        out = task_model(feats, params)
        loss = F.cross_entropy(out, test_target)
        # loss = val_loss.item()  # avoid memory leaks
        return loss


    def evaluate(metadataset, meta_model, task_model, hparams, params0, reg_param, inner_steps, args):
        # meta_model.train()
        device = next(meta_model.parameters()).device

        iters = metadataset.num_tasks
        eval_losses, eval_accs = [], []

        for k in range(iters):

            data = metadataset.sample()
            data = split_into_adapt_eval(data,
                                         shots=args.shots,
                                         ways=args.ways,
                                         device=device)

            task = Task(reg_param, meta_model, task_model, data)  # metabatchsize will be 1 here

            # single task inner loop
            params = [p.detach().clone().requires_grad_(True) for p in params0]
            for ii in range(inner_steps):
                grad = autograd(loss_lower(hparams, params), params)
                with torch.no_grad():
                    for param, egrad in zip(params, grad):
                        rgrad = param.manifold.egrad2rgrad(param, egrad)
                        new_param = param.manifold.retr(param, -args.eta_y * rgrad)
                        param.copy_(new_param)

            task.val_loss_f(params, hparams)

            eval_losses.append(task.val_loss)
            eval_accs.append(task.val_acc)

            if k >= 999:  # use at most 1000 tasks for evaluation
                return np.array(eval_losses), np.array(eval_accs)

        return np.array(eval_losses), np.array(eval_accs)


    run = 1
    mu = 0.1
    inner_lr = .01
    outer_lr = .01
    inner_mu = 0.9
    K = args.steps
    stop_k = None  # stop iteration for early stopping. leave to None if not using it
    n_tasks_train = 20000
    n_tasks_test = 200  # usually around 1000 tasks are used for testing
    n_tasks_val = 200

    reg_param = args.reg_param  # reg_param = 0.5
    T = 30  # T = 30

    T_test = T
    log_interval = 25
    eval_interval = 50

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
    train_tasks, val_tasks, test_tasks = process_data(args)

    meta_model = CNN(32).to(device)
    task_model = FC(3200, args.ways).to(device)


    # training starts
    start_iter = 0
    total_time = 0

    run_time, accs, vals, evals = [], [], [], []

    w0 = [torch.zeros_like(p).to(device) for p in task_model.parameters()]

    hparams = list(meta_model.parameters())
    params = list(task_model.parameters())

    inner_log_interval = None
    inner_log_interval_test = None

    meta_bsz = args.batch_size

    for k in range(start_iter, K):
        start_time = time.time()

        val_loss, val_acc = 0, 0
        forward_time, backward_time = 0, 0
        # w_accum = [torch.zeros_like(w).to(device) for w in w0]

        th = 0.0

        for t_idx in range(meta_bsz):
            start_time_task = time.time()

            # sample a training task
            task_data = train_tasks.sample()

            task_data = split_into_adapt_eval(task_data,
                                              shots=args.shots,
                                              ways=args.ways,
                                              device=device)

            train_input, train_target, test_input, test_target = task_data
            data_lower = [train_input, train_target]
            data_upper = [test_input, test_target]
            # single task set up
            # task = Task(reg_param, meta_model, task_model, task_data, batch_size=meta_bsz)

            hparams, params, loss_u, hgradnorm, step_time = RHGDstep(loss_lower, loss_upper, hparams, params, args,
                                                                     data=[data_lower, data_upper])
            print(loss_u)

        run_time.append(total_time)
        vals.append(val_loss)  # this is actually train loss in few-shot learning
        accs.append(val_acc)  # this is actually train accuracy in few-shot learning

        # evaluate on test data
        if (k + 1) % eval_interval == 0:
            params0 = params.detach().clone()
            val_losses, val_accs = evaluate(val_tasks, meta_model, task_model, hparams, params0, reg_param,
                                            test_inner_lr, args)

            evals.append((val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std()))
            string = "Val loss {:.2e} (+/- {:.2e}): Val acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(
                val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std(), len(val_losses))
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)

            test_losses, test_accs = evaluate(test_tasks, meta_model, task_model, hparams, params0, reg_param,
                                              test_inner_lr, args)

            evals.append((test_losses.mean(), test_losses.std(), 100. * test_accs.mean(), 100. * test_accs.std()))

            string = "Test loss {:.2e} (+/- {:.2e}): Test acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(
                test_losses.mean(), test_losses.std(), 100. * test_accs.mean(), 100. * test_accs.std(),
                len(test_losses))
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)







