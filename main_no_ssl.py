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
from train import train, validate, train_no_ssl



def main(args):

    # create labeled, validation, and test data loader
    # unlabeled data loader not needed for baseline training
    train_loader, val_loader, args = get_data_loaders_no_ssl(args)

    # create models
    model = create_model(args, model = 'efficient',
                         efficient_version = args.efficient_version)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # stats and logger
    logger = Logger(os.path.join(args.out, 'log.txt'))
    logger.set_names(['Train Loss', 'Valid Loss', \
                      'Valid Acc.', 'Train Acc.'])
    start_epoch, best_acc = 0, 0


    # load from checkpoints
    if args.resume:
        print('==> Resuming from checkpoint.')
        load_checkpoint(args, model, optimizer, ema_model=None)

        if args.transfer_learning and start_epoch > args.unfreeze:
            print('Unfreezing layers of model.')
            model = unfreeze_layer(model)


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

        print(f'\nEpoch: [{epoch+1} | {args.epochs}] LR: {args.lr}')

        train_loss, train_acc = train_no_ssl(model = model,
                                  optimizer = optimizer,
                                  criterion = criterion,
                                  train_loader = train_loader,
                                  args = args)

        # get validation loss and accuracy
        val_loss, val_acc = validate(val_loader,
                                     model = model,
                                     criterion = criterion,
                                     epoch = epoch,
                                     mode='Validating',
                                     device = args.device)

        step = args.batch_size * len(train_loader) * (epoch + 1)

        # loggin stats
        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)

        # append logger file
        logger.append([train_loss, val_loss, val_acc, train_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint = args.out)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')

    # Optimization options
    parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Miscs
    parser.add_argument('--seed', type=int, default=0, help='specified random seed.')
    parser.add_argument('--n-labeled', default=None,
                            help='used for compatibility with SSL functions. Default is None.')
    parser.add_argument('--out', type=str, help='output dir in dataset folder.')

    parser.add_argument('--model', default='efficient', type=str, help='model that will be used. \
                                Default is resnet (alternative is pretrained EfficentNet)')
    parser.add_argument('--efficient_version', default='b0', help='efficient-net version. Default is b0.')
    parser.add_argument('--dataset', default='x_ray', type=str, help='choose dataset, cifar10 or x-ray. Default is cifar.')
    parser.add_argument('--unfreeze', default=2, type=int, help='number of epochs before unfreezing network. Default is 2.')
    parser.add_argument('--trans', default=True, type=bool, dest='transfer_learning',\
                        help='turns on transfer learning, default is False. If True, needs --unfreeze epoch')
    parser.add_argument('--balanced', default=True, type=bool, help='load balanced or original (unbalanced) dataset. Default is balanced.')

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