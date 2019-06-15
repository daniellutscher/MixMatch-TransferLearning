from __future__ import print_function

import errno
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), 'models')
from models.resnet import WideResNet
from models.efficientnet import EfficientNet
from .logger import Logger

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SemiLoss(object):
    def __init__(self, lambda_u):
        super(SemiLoss, self).__init__()
        self.lambda_u = lambda_u

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) \
                                * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, lr = 0.002, alpha=0.999, \
                model_version = 'efficient', efficient_version = 'b0',
                num_classes=7, device = 'cuda'):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.lr = lr
        self.device = device

        if model_version == 'resnet':
            self.tmp_model = WideResNet(num_classes = num_classes)
        else:
            self.tmp_model = EfficientNet(
                                    version = efficient_version,
                                    num_classes = num_classes)
        self.wd = 0.02 * self.lr

        for param, ema_param in zip(self.model.parameters(), \
                                    self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        self.model.to('cpu')
        self.ema_model.to('cpu')
        self.tmp_model.to('cpu')
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), \
                                            self.tmp_model.parameters()):

                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), \
                                            self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), \
                                        self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)
        self.model.to(self.device)
        self.ema_model.to(self.device)
        self.tmp_model.to(self.device)


def get_models(args):

    print("==> Creating model.")
    model = create_model(args,
                         model = args.model,
                         efficient_version = args.efficient_version)
    ema_model = create_model(args,
                             model = args.model,
                             efficient_version = args.efficient_version,
                             ema=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger = Logger(os.path.join(args.out, 'log.txt'))
    logger.set_names(['Train Loss', 'Train Loss X', \
                      'Train Loss U',  'Valid Loss', \
                      'Valid Acc.', 'Test Loss', 'Test Acc.'])

    start_epoch, best_acc = 0, 0

    print('Total params: %.2fM' % \
    (sum(p.numel() for p in model.parameters())/1000000.0))

    # initialize exponential mean average optimizer
    ema_optimizer= WeightEMA(model = model,
                             ema_model = ema_model,
                             lr = args.lr,
                             alpha = args.ema_decay,
                             model_version = args.model,
                             num_classes = args.num_classes,
                             device = args.device)

    if args.resume:

        print('==> Resuming from checkpoint.')
        model, ema_model, optimizer, \
        logger, start_epoch, best_acc = load_checkpoint(args, model,
                                                        ema_model, optimizer)
        if args.transfer_learning and start_epoch > args.unfreeze:
            model = unfreeze_all_layers(model, ema_model)

    return model, ema_model, optimizer, ema_optimizer, \
           logger, start_epoch, best_acc


def load_checkpoint(args, model, ema_model, optimizer):

    checkpoint_file = os.path.join(args.resume, 'checkpoint.pth.tar')

    assert os.path.isfile(checkpoint_file), \
        'no checkpoint found in output folder.'

    checkpoint = torch.load(checkpoint_file)

    model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    logger = Logger(os.path.join(args.resume, 'log.txt'), resume=True)

    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']

    return model, ema_model, optimizer, logger, start_epoch, best_acc


def create_model(args, model, efficient_version = 'b0', ema = False):

    if model == 'resnet':
        model = WideResNet(num_classes = args.num_classes)
    elif model == 'efficient':
        model = EfficientNet(version = efficient_version,
                             num_classes = args.num_classes)
    model = model.to(args.device)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def make_dir(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def set_random_seeds(seed):
    # type: (int) -> None
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def unfreeze_all_layers(model, ema_model):
    print('Unfreezing layers of model and ema_model.')

    for child in model.model_ft.children():

      for param in child.parameters():
        param.requires_grad = True

    for child in ema_model.model_ft.children():

          for param in child.parameters():
            param.requires_grad = True

    return model

