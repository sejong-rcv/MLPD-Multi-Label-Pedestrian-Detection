import sys, os
from os.path import join

import argparse, json, pdb
from tqdm import tqdm
from pprint import PrettyPrinter

from collections import namedtuple

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image, ImageDraw, ImageFont

from utils import *
from datasets import *

from torchcv.datasets.transforms import *
from torchcv.utils import Timer, kaist_results_file as write_result, write_coco_format as write_result_coco

### Evaluation
from torchcv.evaluations.coco import COCO
from torchcv.evaluations.eval_MR_multisetup import COCOeval


# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './datasets/kaist-rgbt/'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './BEST_checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
preprocess1 = Compose([ Resize([300, 300])])    
transforms1 = Compose([ ToTensor(), \
                        Normalize([0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
                        Normalize([0.1598], [0.0813], 'T')                        
                        ])
test_dataset = KAISTPed('test-all-20.txt',img_transform=preprocess1, co_transform=transforms1)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=workers,
                                            collate_fn=test_dataset.collate_fn, 
                                            pin_memory=True)     



def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels,_) in enumerate(test_loader):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos


            ###### torchcv.ver map
            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            

            print('\n t : %d' % len(test_loader) )
            print(' n : %d'% i )

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
    

    #Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


def evaluate_coco(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    fig_test,  ax_test  = plt.subplots(figsize=(18,15))

    data_time = AverageMeter()

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
            
            start = time.time()
            # Forward prop.
            predicted_locs, predicted_scores = model(image_vis, image_lwir)
            data_time.update(time.time() - start)
            #print(data_time.val)
            
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch, det_bg_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.425,
                                                                                       top_k=50)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos
            
            for box_t, label_t, score_t, bg_score_t, ids in zip(det_boxes_batch ,det_labels_batch, det_scores_batch, det_bg_scores_batch, index):
                for box, label, score, bg_score in zip(box_t, label_t, score_t, bg_score_t) :
                    
                    score_max = score.max().item()
                    bg_score = bg_score.item()
                    
                    #if bg_score  > score_max : 
                    #    continue

                    bb = box.cpu().numpy().tolist()

                    try : 
                        results.append( {\
                                        'image_id': ids.item(), \
                                        'category_id': label.item(), \
                                        'bbox': [bb[0]*input_size[1], bb[1]*input_size[0], (bb[2]-bb[0])*input_size[1], (bb[3]-bb[1])*input_size[0]], \
                                        'score': score.mean().item()} )
                    except : 
                        import pdb;pdb.set_trace()
                    
    rstFile = os.path.join(checkpoint_root, './COCO_TEST_det_{:s}.json'.format(checkpoint_name))            
    write_result_coco(results, rstFile)
    
    # rstFile = os.path.join('./result/COCO_TEST_det_MBNET_SF_Base.json')

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
        
        print('Recall: {:}'.format( 1-cocoEval.eval['yy'][0][-1] ) )
        print('Time : ',data_time.avg)

    except:
        import torchcv.utils.trace_error
        print('[Error] cannot evaluate by cocoEval. ')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_matlab(test_loader, model):
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
        
        file_name = join(anno_save, f'det-test-{test_mode}')
        f = open(file_name, 'w')

        for i, (image_vis, image_lwir, boxes, labels, index) in enumerate(tqdm(test_loader, desc='Evaluating')):

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs, predicted_scores = model(image_vis, image_lwir)
            
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch, det_bg_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.425,
                                                                                       top_k=50)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            for box_t, label_t, score_t, ids in zip(det_boxes_batch ,det_labels_batch, det_scores_batch, index):
        
                frame_id = img_set_list[ids.item()]


                for box, label, score in zip(box_t, label_t, score_t) :
                    
                    bb = box.cpu().numpy().tolist()
                    
                    bbox = [bb[0]*input_size[1], bb[1]*input_size[0], (bb[2])*input_size[1], (bb[3])*input_size[0]]
                    image_id = ids.item()
                    score = score.mean().item()
                    out_txt = str(image_id+1) + ',' + str(format(bbox[0],".4f")) + ',' + str(format(bbox[1],".4f")) + ',' + str(format(bbox[2]-bbox[0],".4f")) + ',' + str(format(bbox[3]-bbox[1],".4f")) + ',' + str(format(score,".8f")) + '\n'
                    f.write(out_txt)
        
        f.close()


if __name__ == '__main__':
    evaluate(test_loader, model)
