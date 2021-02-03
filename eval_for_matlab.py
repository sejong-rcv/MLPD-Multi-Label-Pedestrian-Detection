from utils import *
from datasets import *
from torchcv.datasets.transforms import *
import torch.nn.functional as F
from tqdm import tqdm
from pprint import PrettyPrinter
import argparse

import torch
import torch.utils.data as data
import json
import os
import os.path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from utils import *
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import pdb
from collections import namedtuple


from torchcv.utils import Timer, kaist_results_file as write_result, write_coco_format as write_result_coco

### Evaluation
from torchcv.evaluations.coco import COCO
from torchcv.evaluations.eval_MR_multisetup import COCOeval

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--synth_fail',         default=['None', 'None'], nargs='+', type=str, help='Specify synthetic failure: e.g. crack1.jpg None')
parser.add_argument('--mode',         default='all', nargs='+', type=str, help='mode')

annType = 'bbox'

DB_ROOT = './datasets/kaist-rgbt'
JSON_GT_FILE = os.path.join( DB_ROOT, 'kaist_annotations_test20.json' )
cocoGt = COCO(JSON_GT_FILE)

# Parameters
data_folder = './datasets/kaist-rgbt/'
folder_root = './LF_ALL_RRC/det'
batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## AR-CNN Annotation (FLIP)
checkpoint = './jobs/2021-01-16_02h28m_SSD_KAIST_LF_FLIP_addrandomresizecrop/checkpoint_ssd300.pth.tar027'
anno_save = './LF_ALL_RRC'


input_size = [512., 640.]
ori_size = (512, 640)  

args = parser.parse_args()
test_mode = args.mode[0]


# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
preprocess1 = Compose([ ])
transforms1 = Compose([ Resize(input_size), \
                        ToTensor(), \
                        Normalize([0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
                        Normalize([0.1598], [0.0813], 'T')                        
                        ])

if test_mode == 'all' : 
    test_dataset = KAISTPed('test-all-20.txt',img_transform=preprocess1, co_transform=transforms1, condition='test')
elif test_mode == 'day' :
    test_dataset = KAISTPed('test-day-20.txt',img_transform=preprocess1, co_transform=transforms1, condition='test')
elif test_mode == 'night' :
    test_dataset = KAISTPed('test-night-20.txt',img_transform=preprocess1, co_transform=transforms1, condition='test')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=workers,
                                            collate_fn=test_dataset.collate_fn, 
                                            pin_memory=True)     

img_set_list = list()

if test_mode == 'all' : 
    for line in open(os.path.join(DB_ROOT, 'imageSets', 'test-all-20.txt')):
        img_set_list.append((DB_ROOT, line.strip().split('/')))
elif test_mode == 'day' :
    for line in open(os.path.join(DB_ROOT, 'imageSets', 'test-day-20.txt')):
        img_set_list.append((DB_ROOT, line.strip().split('/')))
elif test_mode == 'night' :
    for line in open(os.path.join(DB_ROOT, 'imageSets', 'test-night-20.txt')):
        img_set_list.append((DB_ROOT, line.strip().split('/')))



def evaluate_coco(test_loader, model):
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
        
        if test_mode == 'all' :
            file_name = anno_save + '/det-test-all.txt'
        elif test_mode == 'day' :
            file_name = anno_save + '/det-test-day.txt'
        elif test_mode == 'night' : 
            file_name = anno_save + '/det-test-night.txt'
        
        f = open(file_name, 'w')

        for i, (image_vis, image_lwir, boxes, labels, index) in enumerate(tqdm(test_loader, desc='Evaluating')):

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs, predicted_scores = model(image_vis, image_lwir)
            
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.425,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            for box_t, label_t, score_t, ids in zip(det_boxes_batch ,det_labels_batch, det_scores_batch, index):
        
                frame_id = img_set_list[ids.item()]
                # file_name = folder_root + '/' + frame_id[1][0] + '_' + frame_id[1][1] + '_' + frame_id[1][2] + '.txt'
                # f = open(file_name, 'w')

                for box, label, score in zip(box_t, label_t, score_t) :
                    bb = box.cpu().numpy().tolist()
                    bbox = [bb[0]*input_size[1], bb[1]*input_size[0], (bb[2])*input_size[1], (bb[3])*input_size[0]]
                    image_id = ids.item()
                    category = 'person'
                    score = score.item()
                    # out_txt = category + ' ' + str(format(bbox[0],".4f")) + ' ' + str(format(bbox[1],".4f")) + ' ' + str(format(bbox[2],".4f")) + ' ' + str(format(bbox[3],".4f")) + ' ' + str(score) + '\n'
                    out_txt = str(image_id+1) + ',' + str(format(bbox[0],".4f")) + ',' + str(format(bbox[1],".4f")) + ',' + str(format(bbox[2]-bbox[0],".4f")) + ',' + str(format(bbox[3]-bbox[1],".4f")) + ',' + str(format(score,".8f")) + '\n'
                    f.write(out_txt)

                # f.close()
        
        f.close()
                    

if __name__ == '__main__':
    evaluate_coco(test_loader, model)
