import sys, os, pdb, time, json, argparse
from tqdm import tqdm
from pprint import PrettyPrinter

from collections import namedtuple

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

### Logging
import logging
# import logging.handlers
from datetime import datetime
from tensorboardX import SummaryWriter



import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
# import torch.optim
# import torch.utils.data

import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import SSD300, MultiBoxLoss

from utils import *
from datasets import *
from torchcv.datasets.transforms import *
from torchcv.utils import run_tensorboard
from torchcv.utils import Timer, kaist_results_file as write_result, write_coco_format as write_result_coco

### Evaluation
from torchcv.evaluations.coco import COCO
from torchcv.evaluations.eval_MR_multisetup import COCOeval

annType = 'bbox'

DB_ROOT = './datasets/kaist-rgbt'
JSON_GT_FILE = os.path.join( DB_ROOT, 'kaist_annotations_test20.json' )
cocoGt = COCO(JSON_GT_FILE)

# Parameters
data_folder = './datasets/kaist-rgbt/'
input_size = [512., 640.]
############################################################################################


# Data parameters
#DB_Root = './datasets/kaist-rgbt/'  # folder with data files

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 3  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
# checkpoint = './jobs/2021-01-28_07h54m_SSD_KAIST_LF_Multi_Label/checkpoint_ssd300.pth.tar003'
checkpoint = None
batch_size = 8 # batch size
start_epoch = 0  # start at this epoch
epochs = 40  # number of epochs to run without early-stopping
epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
best_loss = 100.  # assume a high loss at first
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 10  # print training or validation status every __ batches
lr = 1e-4   # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
port = 8807
cudnn.benchmark = True
ori_size = (512, 640)  

# random seed fix 
torch.manual_seed(9)
torch.cuda.manual_seed(9)
np.random.seed(9)
random.seed(9)
torch.backends.cudnn.deterministic=True

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--exp_time',   default=None, type=str,  help='set if you want to use exp time')
parser.add_argument('--exp_name',   default=None, type=str,  help='set if you want to use exp name')
parser.add_argument('--annotation',   default='AR-CNN', type=str,  help='set if you want to use annotation, defalut is Original(KAIST)')
# AR-CNN, Sanitize, Original 

args = parser.parse_args()

def main():
    """
    Training and validation.
    """

    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ int(epochs*0.5), int(epochs*0.9)], gamma=0.1)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        train_loss = checkpoint['loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch,train_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ int(epochs*0.5)], gamma=0.1)

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # preprocess2 = Compose([  ColorJitter(0.3, 0.3, 0.3), ColorJitterLWIR(contrast=0.3),  FaultTolerant([0.5, 0.5, [0.25,0.5,0.75]])])
    # preprocess2 = Compose([  ColorJitter(0.3, 0.3, 0.3), ColorJitterLWIR(contrast=0.3)])
    preprocess2 = Compose([  ColorJitter(0.3, 0.3, 0.3), ColorJitterLWIR(contrast=0.3)])
    transforms2 = Compose([ TT_RandomHorizontalFlip(p=0.5, flip=0.5), \
                        TT_RandomResizedCrop([512,640], scale=(0.25, 4.0), ratio=(0.8, 1.2)), \
                        ToTensor(), \
                        Normalize( [0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
                        Normalize([0.1598], [0.0813], 'T')\
                        ])
    # transforms2 = Compose([ TT_RandomHorizontalFlip(), \
    #                         ToTensor(), \
    #                         Normalize( [0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
    #                         Normalize([0.1598], [0.0813], 'T')\
    #                         ])

    preprocess1 = Compose([ ])    
    transforms1 = Compose([ Resize(input_size), \
                            ToTensor(), \
                            Normalize([0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
                            Normalize([0.1598], [0.0813], 'T')                        
                            ])


    train_dataset = KAISTPed('train-all-02.txt',img_transform=preprocess2, co_transform=transforms2, condition='train',annotation=args.annotation)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers,
                                               collate_fn=train_dataset.collate_fn, 
                                               pin_memory=True)  # note that we're passing the collate function here

    test_dataset = KAISTPed('test-all-20.txt',img_transform=preprocess1, co_transform=transforms1, condition='test',annotation=args.annotation)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=workers,
                                                collate_fn=test_dataset.collate_fn, 
                                                pin_memory=True)    
    #############################################################################################################################
    
    ### Set job directory

    if args.exp_time is None:
        args.exp_time        = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    
    exp_name        = ('_' + args.exp_name) if args.exp_name else '_' 
    jobs_dir        = os.path.join( 'jobs', args.exp_time + exp_name )
    args.jobs_dir   = jobs_dir

    snapshot_dir    = os.path.join( jobs_dir, 'snapshots' )
    tensorboard_dir    = os.path.join( jobs_dir, 'tensorboardX' )
    if not os.path.exists(snapshot_dir):        os.makedirs(snapshot_dir)
    if not os.path.exists(tensorboard_dir):     os.makedirs(tensorboard_dir)
    run_tensorboard( tensorboard_dir, port )

    ### Backup current source codes
    
    import tarfile
    tar = tarfile.open( os.path.join(jobs_dir, 'sources.tar'), 'w' )
    tar.add( 'torchcv' )    
    tar.add( __file__ )

    import glob
    for file in sorted( glob.glob('*.py') ):
        tar.add( file )

    tar.close()

    ### Set logger
    
    writer = SummaryWriter(os.path.join(jobs_dir, 'tensorboardX'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(levelname)s] [%(asctime)-11s] %(message)s')
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    logger.addHandler(h)

    h = logging.FileHandler(os.path.join(jobs_dir, 'log_{:s}.txt'.format(args.exp_time)))
    h.setFormatter(fmt)
    logger.addHandler(h)

    settings = vars(args)
    for key, value in settings.items():
        settings[key] = value   

    logger.info('Exp time: {}'.format(settings['exp_time']))
    for key, value in settings.items():
        if key == 'exp_time':
            continue
        logger.info('\t{}: {}'.format(key, value))

    logger.info('Preprocess for training')
    logger.info( preprocess2 )
    logger.info('Transforms for training')
    logger.info( transforms2 )

    #############################################################################################################################

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Paper describes decaying the learning rate at the 80000th, 100000th, 120000th 'iteration', i.e. model update or batch
        # The paper uses a batch size of 32, which means there were about 517 iterations in an epoch
        # Therefore, to find the epochs to decay at, you could do,
        # if epoch in {80000 // 517, 100000 // 517, 120000 // 517}:
        #     adjust_learning_rate(optimizer, 0.1)

        # In practice, I just decayed the learning rate when loss stopped improving for long periods,
        # and I would resume from the last best checkpoint with the new learning rate,
        # since there's no point in resuming at the most recent and significantly worse checkpoint.
        # So, when you're ready to decay the learning rate, just set checkpoint = 'BEST_checkpoint_ssd300.pth.tar' above
        # and have adjust_learning_rate(optimizer, 0.1) BEFORE this 'for' loop

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger,
                           writer=writer)

        optim_scheduler.step()
        
        # Save checkpoint
        writer.add_scalars('train/epoch', {'epoch_train_loss': train_loss},global_step=epoch )
        save_checkpoint(epoch, model, optimizer, train_loss, jobs_dir)
        
        if epoch >= 3 :
            evaluate_coco(test_loader, model, epoch,jobs_dir, writer)
            



def train(train_loader, model, criterion, optimizer, epoch, logger, writer):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss_sum
    losses_loc = AverageMeter()  # loss_loc
    losses_cls = AverageMeter()  # loss_cls

    start = time.time()
    
    # Batches
    for batch_idx, (image_vis, image_lwir, boxes, labels, _) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # Move to default device
        image_vis = image_vis.to(device) 
        image_lwir = image_lwir.to(device) 

        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss,cls_loss,loc_loss,n_positives = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        if np.isnan(loss.item()):
            import pdb; pdb.set_trace()
            loss,cls_loss,loc_loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # losses.update(loss.item(), images.size(0))
        losses.update(loss.item())
        losses_loc.update(loc_loss)
        losses_cls.update(cls_loss)
        batch_time.update(time.time() - start)

        start = time.time()

        if batch_idx and batch_idx % print_freq == 0:         
            writer.add_scalars('train/loss', {'loss': losses.avg}, global_step=epoch*len(train_loader)+batch_idx )
            writer.add_scalars('train/loc', {'loss': losses_loc.avg}, global_step=epoch*len(train_loader)+batch_idx )                
            writer.add_scalars('train/cls', {'loss': losses_cls.avg}, global_step=epoch*len(train_loader)+batch_idx )

        # Print status
        if batch_idx % print_freq == 0:

            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'num of Positive {Positive}\t'.format(epoch, batch_idx, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, Positive=n_positives))

    del predicted_locs, predicted_scores, image_vis, image_lwir, boxes, labels  # free some memory since their histories may be stored
    return  losses.avg

def evaluate_coco(test_loader, model,epoch,jobs_dir, writer):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    fig_test,  ax_test  = plt.subplots(figsize=(18,15))

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    #For CoCo
    results = []

    with torch.no_grad():
        # Batches
        for i, (image_vis, image_lwir, boxes, labels, index) in enumerate(tqdm(test_loader, desc='Evaluating')):

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs, predicted_scores = model(image_vis, image_lwir)
            
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch, _ = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.425,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos      
            
            for box_t, label_t, score_t, ids in zip(det_boxes_batch ,det_labels_batch, det_scores_batch, index):
                for box, label, score in zip(box_t, label_t, score_t) :

                    bb = box.cpu().numpy().tolist()
                    
                    results.append( {\
                                    'image_id': ids.item(), \
                                    'category_id': label.item(), \
                                    'bbox': [bb[0]*input_size[1], bb[1]*input_size[0], (bb[2]-bb[0])*input_size[1], (bb[3]-bb[1])*input_size[0]], \
                                    'score': score.mean().item()} )

    rstFile = os.path.join(jobs_dir, './COCO_TEST_det_{:d}.json'.format(epoch))            
    write_result_coco(results, rstFile)

    try:

        cocoDt = cocoGt.loadRes(rstFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds  = [1]    
        cocoEval.evaluate(0)
        cocoEval.accumulate()
        curPerf = cocoEval.summarize(0)    

        cocoEval.draw_figure(ax_test, rstFile.replace('json', 'jpg'))        
        writer.add_scalars('LAMR/fppi', {'test': curPerf}, epoch)
        
        print('Recall: {:}'.format( 1-cocoEval.eval['yy'][0][-1] ) )

    except:
        import torchcv.utils.trace_error
        print('[Error] cannot evaluate by cocoEval. ')

if __name__ == '__main__':

    main()
