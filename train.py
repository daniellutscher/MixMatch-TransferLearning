from __future__ import print_function

import time
import numpy as np
import torch
from utils import *
from progress.bar import Bar as Bar


def train(labeled_trainloader, unlabeled_trainloader, model,
          optimizer, ema_optimizer, criterion, epoch, args):

    # initialize all stats
    bar = Bar('Training', max=args.val_iteration)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    # initialize data loader
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    model.to(args.device)

    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        # Transform label to one-hot
        targets_x = torch.zeros(args.batch_size, args.num_classes)\
                            .scatter_(1, targets_x.view(-1,1), 1)

        inputs_x, targets_x = inputs_x.to(args.device), \
                              targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.to(args.device)
        inputs_u2 = inputs_u2.to(args.device)


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) \
                 + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples
        # between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, args.batch_size))
        mixed_input = interleave(mixed_input, args.batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, args.batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:args.batch_size], \
                              logits_u, mixed_target[args.batch_size:], \
                              epoch+batch_idx/args.val_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        progress_batch = '({batch}/{size}) Data: {data:.3f}s '.format(
                                                    batch=batch_idx + 1,
                                                    size=args.val_iteration,
                                                    data=data_time.avg)
        progress_total = '| Batch: {bt:.3f}s | Total: {total:} '.format(
                                                    bt=batch_time.avg,
                                                    total=bar.elapsed_td)
        progress_train_loss = '| ETA: {eta:} | Loss: {loss:.4f} '.format(
                                                    eta=bar.eta_td,
                                                    loss=losses.avg)
        progress_loss_x = '| Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} '\
                                            .format(loss_x=losses_x.avg,
                                                    loss_u=losses_u.avg)
        progress_w = '| W: {w:.4f}'.format(w=ws.avg)
        bar.suffix  = '{batch}{total}{train_loss}{loss_x}{w}'.format(
                                            batch = progress_batch,
                                            total = progress_total,
                                            train_loss = progress_train_loss,
                                            loss_x = progress_loss_x,
                                            w = progress_w)
        bar.next()
    bar.finish()

    ema_optimizer.step(bn=True)

    return (losses.avg, losses_x.avg, losses_u.avg,)



def validate(valloader, model, criterion, epoch, mode, device = 'cuda'):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.to(device)

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(args.device), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            progress_batch = '({batch}/{size}) Data: {data:.3f}s '.format(
                                            batch=batch_idx + 1,
                                            size=len(valloader),
                                            data=data_time.avg)
            progress_total = '| Batch: {bt:.3f}s | Total: {total:} '.format(
                                            bt=batch_time.avg,
                                            total=bar.elapsed_td)
            progress_train_loss = '| ETA: {eta:} | Loss: {loss:.4f} '.format(
                                            eta=bar.eta_td,
                                            loss=losses.avg)
            progress_topk = '| top1: {top1: .4f} | top5: {top5: .4f}'.format(
                                            top1 = top1.avg,
                                            top5 = top5.avg)
            # plot progress
            bar.suffix  = '{batch}{total}{train_loss}{topk}'.format(
                                            batch = progress_batch,
                                            total = progress_total,
                                            train_loss = progress_train_loss,
                                            topk = progress_topk)

            bar.next()
        bar.finish()

    return (losses.avg, top1.avg)



def train_no_ssl(model, optimizer, criterion, train_loader, args):

    # initialize all stats
    args.val_iteration = len(train_loader) * args.batch_size
    bar = Bar('Training', max = args.val_iteration)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.train()
    model.to(args.device)

    for idx, batch in enumerate(train_loader):
        
        # send data to GPU
        inputs, targets = batch[0].to(args.device), batch[1].to(args.device)

        # forward
        outputs = model(inputs)

        # backward
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        progress_batch = '({batch}/{size}) '.format(batch=idx + 1,
                                                    size=args.val_iteration)
        progress_total = '| Batch: {bt:.3f}s | Total: {total:} '.format(
                                                    bt=batch_time.avg,
                                                    total=bar.elapsed_td)
        progress_train_loss = '| ETA: {eta:} | Loss: {loss:.4f} '.format(
                                                    eta=bar.eta_td,
                                                    loss=losses.avg)
        bar.suffix  = '{batch}{total}{train_loss}'.format(
                                            batch = progress_batch,
                                            total = progress_total,
                                            train_loss = progress_train_loss)
        bar.next()
    bar.finish()

    return losses.avg