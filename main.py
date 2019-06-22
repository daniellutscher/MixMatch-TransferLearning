from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from utils import *
from train import train, validate


def main(args):

    # create labeled, unlabeled, validation, and test data loader
    labeled_trainloader, unlabeled_trainloader, \
        val_loader, test_loader, args = get_data_loaders(args)

    # create or load models
    model, ema_model, optimizer,\
    ema_optimizer, logger, start_epoch, best_acc  = get_models(args)

    # initialize loss functions
    train_criterion = SemiLoss(args.lambda_u)
    criterion = nn.CrossEntropyLoss()

    # initialize useful stats / logger variables
    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []

    # train and val
    for epoch in range(start_epoch, args.epochs):

        # transfer learning approach for the efficientNet model
        # First run only the last layers while keeping pre-trained frozen
        # after args.unfreeze epochs, fine-tune the whole network
        if args.transfer_learning and epoch == args.unfreeze:
            model = unfreeze_layer(model)
            ema_model = unfreeze_layer(ema_model)

        print(f'\nEpoch: [{epoch+1} | {args.epochs}] LR: {args.lr}')

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader = labeled_trainloader,
                                                       unlabeled_trainloader = unlabeled_trainloader,
                                                       model = model,
                                                       optimizer = optimizer,
                                                       ema_optimizer = ema_optimizer,
                                                       criterion = train_criterion,
                                                       epoch = epoch,
                                                       args = args)

        # get training accuracy
        _, train_acc = validate(labeled_trainloader,
                                model = ema_model,
                                criterion = criterion,
                                epoch = epoch,
                                mode='Train Stats',
                                device = args.device)

        # get validation loss and accuracy
        val_loss, val_acc = validate(val_loader,
                                     model = ema_model,
                                     criterion = criterion,
                                     epoch = epoch,
                                     mode='Valid Stats',
                                     device = args.device)

        # get test loss and accuracy
        test_loss, test_acc = validate(test_loader,
                                       model = ema_model,
                                       criterion = criterion,
                                       epoch = epoch,
                                       mode='Test Stats ',
                                       device = args.device)
        step = args.batch_size * args.val_iteration * (epoch + 1)

        # loggin stats
        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, \
                       val_loss, val_acc, test_loss, test_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint = args.out)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')

    # Optimization options
    parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                        metavar='LR', help='initial learning rate')
    # Checkpoints
    parser.add_argument('--resume', default=True, type=bool, metavar='PATH',
                        help='Continue from latest checkpoint. Default is True.')

    # Miscs
    parser.add_argument('--seed', type=int, default=0, help='specified random seed.')
    parser.add_argument('--out', type=str, help='output dir in dataset folder.')

    #Method options
    parser.add_argument('--n-labeled', type=int, default=250,
                            help='Number of labeled data')
    parser.add_argument('--val-iteration', type=int, default=1024,
                            help='Number of labeled data')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--model', default='resnet', type=str, help='model that will be used. \
                                Default is resnet (alternative is pretrained EfficentNet)')
    parser.add_argument('--efficient_version', default='b0', help='efficient-net version. Default is b0.')
    parser.add_argument('--dataset', default='cifar', type=str, help='choose dataset, cifar10 or x-ray. Default is cifar.')
    parser.add_argument('--unfreeze', default=2, type=int, help='number of epochs before unfreezing network. Default is 2.')
    parser.add_argument('--trans', default=True, type=bool, dest='transfer_learning',\
                        help='turns on transfer learning, default is False. If True, needs --unfreeze epoch')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.data_dir = os.path.join(os.path.abspath(os.getcwd()), 'dataset')
    args.out = os.path.join(os.path.abspath(os.getcwd()), f'results/{args.out}')

    # enables the inbuilt cudnn auto-tuner to find best algorithm
    # to use for your hardware.
    cudnn.benchmark = True

    # set all random seeds to same value
    set_random_seeds(args.seed)

    # create output folder
    make_dir(args.out)

    main(args)
