from datetime import datetime
from typing import Dict
import config
import logging
import numpy as np
import os
import time

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from datasets import KAISTPed
from inference import val_epoch, save_results
from model import SSD300, MultiBoxLoss
from utils import utils
from utils.evaluation_script import evaluate

torch.backends.cudnn.benchmark = False

# random seed fix 
utils.set_seed(seed=9)


def main():
    """Train and validate a model"""

    # TODO(sohwang): why do we need these global variables?
    # global epochs_since_improvement, start_epoch, label_map, best_loss, epoch

    args = config.args
    train_conf = config.train
    checkpoint = train_conf.checkpoint
    start_epoch = train_conf.start_epoch
    epochs = train_conf.epochs
    phase = "Multispectral"

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SSD300(n_classes=args.n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                            {'params': not_biases}],
                                    lr=train_conf.lr,
                                    momentum=train_conf.momentum,
                                    weight_decay=train_conf.weight_decay,
                                    nesterov=False)

        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                               milestones=[int(epochs * 0.5), int(epochs * 0.9)],
                                                               gamma=0.1)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        train_loss = checkpoint['loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, train_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5)], gamma=0.1)

    # Move to default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = nn.DataParallel(model)

    criterion = MultiBoxLoss(priors_cxcy=model.module.priors_cxcy).to(device)

    train_dataset = KAISTPed(args, condition='train')
    train_loader = DataLoader(train_dataset, batch_size=train_conf.batch_size, shuffle=True,
                              num_workers=config.dataset.workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here

    test_dataset = KAISTPed(args, condition='test')
    test_batch_size = args["test"].eval_batch_size * torch.cuda.device_count()
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             num_workers=config.dataset.workers,
                             collate_fn=test_dataset.collate_fn,
                             pin_memory=True)
    # Set job directory
    if args.exp_time is None:
        args.exp_time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    
    # TODO(sohwang): should config.exp_name be updated from command line argument?
    exp_name = ('_' + args.exp_name) if args.exp_name else '_'
    jobs_dir = os.path.join('jobs', args.exp_time + exp_name)
    os.makedirs(jobs_dir, exist_ok=True)
    args.jobs_dir = jobs_dir

    # Make logger
    logger = utils.make_logger(args)

    # Epochs
    kwargs = {'grad_clip': args['train'].grad_clip, 'print_freq': args['train'].print_freq}
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        logger.info('#' * 20 + f' << Epoch {epoch:3d} >> ' + '#' * 20)
        train_loss = train_epoch(model=model,
                                 dataloader=train_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 logger=logger,
                                 **kwargs)

        optim_scheduler.step()

        # Save checkpoint
        utils.save_checkpoint(epoch, model.module, optimizer, train_loss, jobs_dir)
        
        if epoch >= 3:
            result_filename = os.path.join(jobs_dir, f'Epoch{epoch:03d}_test_det.txt')

            # High min_score setting is important to guarantee reasonable number of detections
            # Otherwise, you might see OOM in validation phase at early training epoch
            results = val_epoch(model, test_loader, config.test.input_size, min_score=0.1)

            save_results(results, result_filename)
            
            evaluate(config.PATH.JSON_GT_FILE, result_filename, phase) 


def train_epoch(model: SSD300,
                dataloader: torch.utils.data.DataLoader,
                criterion: MultiBoxLoss,
                optimizer: torch.optim.Optimizer,
                logger: logging.Logger,
                **kwargs: Dict) -> float:
    """Train the model during an epoch

    Parameters
    ----------
    model: SSD300
        SSD300 model for multispectral pedestrian detection defined by src/model.py
    dataloader: torch.utils.data.dataloader
        Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    criterion: MultiBoxLoss
        Compute multibox loss for single-shot detection
    optimizer: torch.optim.Optimizer
        Pytorch optimizer(e.g. SGD, Adam, etc)
    logger: logging.Logger
        Logger instance
    kwargs: Dict
        Other parameters to control grid_clip, print_freq

    Returns
    -------
    float
        A single scalar value for averaged loss
    """

    device = next(model.parameters()).device
    model.train()  # training mode enables dropout

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses_sum = utils.AverageMeter()  # loss_sum
    # losses_loc = utils.AverageMeter()  # loss_loc
    # losses_cls = utils.AverageMeter()  # loss_cls

    start = time.time()
    
    # Batches
    for batch_idx, (image_vis, image_lwir, boxes, labels, _) in enumerate(dataloader):
        data_time.update(time.time() - start)

        # Move to default device
        image_vis = image_vis.to(device)
        image_lwir = image_lwir.to(device)

        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss, cls_loss, loc_loss, n_positives = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # TODO(sohwang): Do we need this?
        if np.isnan(loss.item()):
            loss, cls_loss, loc_loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Clip gradients, if necessary
        if kwargs.get('grad_clip', None):
            utils.clip_gradient(optimizer, kwargs['grad_clip'])

        # Update model
        optimizer.step()

        losses_sum.update(loss.item())
        # losses_loc.update(loc_loss.sum().item())
        # losses_cls.update(cls_loss.sum().item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if batch_idx % kwargs.get('print_freq', 10) == 0:
            logger.info('Iteration: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'num of Positive {Positive}\t'.format(batch_idx, len(dataloader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              loss=losses_sum,
                                                              Positive=n_positives))

    return losses_sum.avg


if __name__ == '__main__':
    main()
