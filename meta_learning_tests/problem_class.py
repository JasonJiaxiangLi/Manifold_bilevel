import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from geoopt import Stiefel

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
        # self.fmeta = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)
        # self.ftask = higher.monkeypatch(task_model, device=device, copy_initial_weights=True)
        self.fmeta = meta_model.to(device)
        self.ftask = task_model.to(device)

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
        feats = self.fmeta(self.test_input, hparams=hparams)
        out = self.ftask(feats, params=params)
        val_loss = F.cross_entropy(out, self.test_target)/self.batch_size
        self.val_loss = val_loss.item()  # avoid memory leaks

        with torch.no_grad():
            pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss

class meta_learning_problem:
    def __init__(self, meta_model, task_model, args):
        self.meta_model = meta_model
        self.task_model = task_model
        self.args = args
        self.upper_manifold = Stiefel(canonical=False)
    
    def loss_lower(self, hparams, params, data):
        train_input, train_target = data
        # task.compute_feats(hparams)
        # loss = task.train_loss_f(params)
        feats = self.meta_model(train_input, hparams)
        out = self.task_model(feats, params)
        loss = F.cross_entropy(out, train_target) + 0.5 * self.args.reg_param * sum([(p ** 2).sum() for p in params])/len(params)
        return loss

    def loss_upper(self, hparams, params, data):
        test_input, test_target = data
        # loss = task.val_loss_f(params, hparams)
        feats = self.meta_model(test_input, hparams)
        out = self.task_model(feats, params)
        loss = F.cross_entropy(out, test_target)
        # loss = val_loss.item()  # avoid memory leaks
        return loss


    def evaluate(self, metadataset, hparams, params0, reg_param, inner_steps, args):
        # meta_model.train()
        device = next(self.meta_model.parameters()).device

        # iters = metadataset.num_tasks
        iters = args.n_tasks_val
        eval_losses, eval_accs = [], []

        for k in range(iters):

            data = metadataset.sample()
            data = split_into_adapt_eval(data,
                                         shots=args.shots,
                                         ways=args.ways,
                                         device=device)

            task = Task(reg_param, self.meta_model, self.task_model, data)  # metabatchsize will be 1 here
            
            train_input, train_target, test_input, test_target = data

            # single task inner loop
            params = [p.detach().clone().requires_grad_(True) for p in params0]
            for ii in range(inner_steps):
                # grad = autograd(loss_lower(hparams, params), params)
                grad = torch.autograd.grad(self.loss_lower(hparams, params, data=[train_input, train_target]), params)
                with torch.no_grad():
                    for param, egrad in zip(params, grad):
                        # rgrad = param.manifold.egrad2rgrad(param, egrad)
                        # new_param = param.manifold.retr(param, -args.eta_y * rgrad)
                        new_param = param - args.eta_y * egrad
                        param.copy_(new_param)

            task.val_loss_f(params, hparams)

            eval_losses.append(task.val_loss)
            eval_accs.append(task.val_acc)

            if k >= 999:  # use at most 1000 tasks for evaluation
                return np.array(eval_losses), np.array(eval_accs)

        return np.array(eval_losses), np.array(eval_accs)