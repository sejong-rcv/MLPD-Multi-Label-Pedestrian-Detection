import sys, os, argparse, json, pdb, time, importlib

from tqdm import tqdm

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from PIL import Image

### Dataset
from datasets import KAISTPed, LoadBox

### Evaluation
from utils.coco import COCO
from utils.eval_MR_multisetup import COCOeval
from utils.utils import *

from utils.transforms import FusionDeadZone

### config
import config
args = importlib.import_module('config').args


def evaluate_coco(test_loader, model, rstFile=None):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    fig_test,  ax_test  = plt.subplots(figsize=(18,15))

    data_time = AverageMeter()

    # MLPD config
    input_size = args.test.input_size
    device = args.device

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
            
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch, det_bg_scores_batch = model.module.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.425,
                                                                                       top_k=50)
    
            
            for box_t, label_t, score_t, bg_score_t, ids in zip(det_boxes_batch ,det_labels_batch, det_scores_batch, det_bg_scores_batch, index):
                for box, label, score, bg_score in zip(box_t, label_t, score_t, bg_score_t) :
    
                    bb = box.cpu().numpy().tolist()
                    
                    results.append( {\
                                    'image_id': ids.item(), \
                                    'category_id': label.item(), \
                                    'bbox': [bb[0]*input_size[1], bb[1]*input_size[0], (bb[2]-bb[0])*input_size[1], (bb[3]-bb[1])*input_size[0]], \
                                    'score': score.mean().item()} )

    if rstFile is None:
        rstFile = os.path.join(args.test.result_path, './original_TEST_det_{:s}.json'.format(args.test.day))         
    else:
        rstFile=rstFile + f'_{args.test.day}.json'
    write_coco_format(results, rstFile)

    try:
        cocoGt = COCO(args.path.JSON_GT_FILE)
        cocoDt = cocoGt.loadRes(rstFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.catIds  = [1]    
        
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate(0)
        cocoEval.accumulate()
        curPerf = cocoEval.summarize(0)     

        cocoEval.draw_figure(ax_test, rstFile.replace('json', 'jpg'))        
        
        print('Recall: {:}'.format( 1-cocoEval.eval['yy'][0][-1] ) )
        print('Time : ',data_time.avg)

    except:
        import utils.trace_error
        print('[Error] cannot evaluate by cocoEval. ')

def evaluate_matlab(test_loader, model, file_name):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    fig_test,  ax_test  = plt.subplots(figsize=(18,15))

    # MLPD config
    input_size = args.test.input_size
    device = args.device

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
        
        if file_name is None:
            file_name = os.path.join(anno_save, f'det-test-{args.test.day}')
        f = open(file_name, 'w')

        for i, (image_vis, image_lwir, boxes, labels, index) in enumerate(tqdm(test_loader, desc='Evaluating')):

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs, predicted_scores = model(image_vis, image_lwir)
            
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch, det_bg_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.45,
                                                                                       top_k=200)

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

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--FDZ', default='original',  type=str,
                        help='Setting for the "Fusion Dead Zone" experiment. e.g. original, blackout_R, blackout_T, SidesBlackout_a, SidesBlackout_b, SurroundingBlackout')

    FDZ_CASE = parser.parse_args().FDZ.lower()

    FDZ_list = ['original', 'blackout_r', 'blackout_t', 'sidesblackout_a', 'sidesblackout_b', 'surroundingblackout']

    test = args.test
    
    if FDZ_CASE not in FDZ_list:
        raise NameError(f'{FDZ_CASE} is not support')
    elif FDZ_CASE is not 'original':
        print(f'Experiment {FDZ_CASE}')
        rstFile = f'{test.result_path}/{FDZ_CASE}_TEST_det'
    else:
        rstFile = None # original

    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(test.checkpoint)
    model = checkpoint['model']
    model = model.to(args.device)

    model = nn.DataParallel(model)

    model.eval()

    # Fusion Dead Zone experiment
    FDZ = [FusionDeadZone(args.FDZ_case[FDZ_CASE], tuple(test.input_size)) ]
    test.img_transform.add(FDZ)

    test_dataset = KAISTPed(args, condition="test")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test.batch_size,
                                                num_workers=args.dataset.workers,
                                                collate_fn=test_dataset.collate_fn, 
                                                pin_memory=True)     


    evaluate_coco(test_loader, model, rstFile)
