from utils import *
from datasets import *
from torchcv.datasets.transforms import *
import torch.nn.functional as F
from tqdm import tqdm
from pprint import PrettyPrinter
import argparse

import time
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
parser.add_argument('--epoch',  default=4, type=int)

annType = 'bbox'

DB_ROOT = './datasets/kaist-rgbt'
JSON_GT_FILE = os.path.join( DB_ROOT, 'kaist_annotations_test20.json' )
cocoGt = COCO(JSON_GT_FILE)

# Parameters
data_folder = './datasets/kaist-rgbt/'
batch_size = 1
workers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

# checkpoint = './jobs/2021-01-31_06h51m_SSD_KAIST_Double_Head_ALL_Sanitized/checkpoint_ssd300.pth.tar{:03d}'.format(args.epoch)
# checkpoint = './jobs/2021-01-31_09h07m_SSD_KAIST_Multi_Labe_MBNet_txt/checkpoint_ssd300.pth.tar{:03d}'.format(args.epoch)
checkpoint = './jobs/2021-01-28_12h35m_SSD_KAIST_LF_Multi_Label_continue/checkpoint_ssd300.pth.tar{:03d}'.format(args.epoch)
# checkpoint = './jobs/2021-01-28_13h24m_SSD_KAIST_Multi_Label_Sanitized/checkpoint_ssd300.pth.tar{:03d}'.format(args.epoch)

checkpoint_name = 'LF_Multi_Label_AVG_AVG'
checkpoint_root = './result_check'
input_size = [512., 640.]
ori_size = (512, 640)  


if not args.synth_fail == ['None', 'None']:
    if(str(args.synth_fail[0])=='None') :
        str1 = "None"
    elif(str(args.synth_fail[0])=="blackout") :
        str1 = "blackout"
    else :
        str1 = str(args.synth_fail[0][:-4])

    if(str(args.synth_fail[1])=='None') :
        str2 = "None"
    elif(str(args.synth_fail[1])=="blackout") :
        str2 = "blackout"
    else :
        str2 = str(args.synth_fail[1][:-4])

    checkpoint_name = checkpoint_name + str1 +'_'+ str2
else :
    checkpoint_name = checkpoint_name + "_Base"

checkpoint_name = checkpoint_name + "Epoch_{:03d}".format(args.epoch)

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
preprocess1 = Compose([ ])

if not args.synth_fail == ['None', 'None']:
    fail_mask = [ os.path.join( 'synthetic_failure_masks', mask ) for mask in args.synth_fail ]
    preprocess1.add( [ SynthFail(fail_mask, (ori_size)) ] )

transforms1 = Compose([ Resize(input_size), \
                        ToTensor(), \
                        Normalize([0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
                        Normalize([0.1598], [0.0813], 'T')                        
                        ])  

test_dataset = KAISTPed('test-all-20.txt',img_transform=preprocess1, co_transform=transforms1, condition='test')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=workers,
                                            collate_fn=test_dataset.collate_fn, 
                                            pin_memory=True)      

data_time = AverageMeter()


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

if __name__ == '__main__':

    evaluate_coco(test_loader, model)
