'''Encode object boxes and labels.'''
import math
import torch
import itertools
import pdb
import numpy as np
from torchcv.utils import meshgrid
from torchcv.utils.box import box_iou, box_iou_ignore, box_nms, change_box_order

class SSDBoxCoder:
    def __init__(self, ssd_model):

        ### Load default_boxes from model
        self.default_boxes = ssd_model._get_anchor_boxes()
        # self.default_boxes = ssd_model._get_manual_anchor_wh()

        ### Define mean/stds (optional)
        self.means   = torch.tensor( [0., 0., 0., 0.] )
        # self.means   = torch.tensor( [0.0007, 0.0008, -0.0107, 0.1331] )
        self.stds    = torch.tensor( [1., 1., 1., 1.] )  
        # self.stds    = torch.tensor( [0.1290, 0.1506, 0.2164, 0.2227] ) * 10
    
    def __call__(self, image, mask, boxes):
        loc_target, cls_target = self.encode(boxes[:,:4], boxes[:,4])
        return image, mask, loc_target, cls_target

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''



        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1].item()
            return (i[j], j )

        try:  

        
            if boxes.size(0) == 1:
                loc_targets = torch.zeros_like(self.default_boxes)
                cls_targets = torch.LongTensor(len(self.default_boxes)).zero_()
                return loc_targets, cls_targets

            default_boxes = self.default_boxes  # xywh            
            default_boxes = change_box_order(default_boxes, 'xywh2xyxy')
            
            ### Ignore dummy boxes from LoadBox func. in kaist_rgbt_ped.py
            boxes[0,:] = torch.tensor( [0.0, 0.0, 0.01, 0.01] )
            labels[ labels == -1 ] = -2
            labels[0] = -1
            
            ious = box_iou_ignore(default_boxes, boxes[1:,:], labels[1:])  # [#anchors, #obj]
            
            best_prior_overlap, index = ious.max(dim=1)
            index += 1
            # index[ best_prior_overlap < 0.5 ] = 0
            # index[ best_prior_overlap >= 0.5 ] += 1     # +1: we will use dummy-included boxes/labels.

            boxes = boxes[index.clamp(min=0)]  # negative index not supported
            boxes[ index == 0 ] = default_boxes[ index == 0 ]
            

            boxes = change_box_order(boxes, 'xyxy2xywh')
            default_boxes = change_box_order(default_boxes, 'xyxy2xywh')

            loc_xy = ( (boxes[:,:2]-default_boxes[:,:2]) / default_boxes[:,2:] - self.means[:2] ) / self.stds[:2]
            loc_wh = ( torch.log(boxes[:,2:]/default_boxes[:,2:]) - self.means[2:] ) / self.stds[2:]

            loc_targets = torch.cat([loc_xy,loc_wh], 1)
            cls_targets = 1 + labels[index.clamp(min=0)]
            cls_targets[ best_prior_overlap < 0.5 ] = 0

            return loc_targets, cls_targets

        except Exception as ex:
            import torchcv.utils.trace_error
            pdb.set_trace()

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45, applyNms=True):
        '''Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        try:
            xy = ( loc_preds[:,:2] * self.stds[:2] + self.means[:2] ) * self.default_boxes[:,2:] + self.default_boxes[:,:2]
            wh = torch.exp(loc_preds[:,2:] * self.stds[2:] + self.means[2:]) * self.default_boxes[:,2:]
            box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

            boxes = []
            labels = []
            scores = []
            num_classes = cls_preds.size(1)
            for i in range(num_classes-1):
                score = cls_preds[:,i+1]  # class i corresponds to (i+1) column
                mask = score > score_thresh
                if not mask.any():
                    continue
                                
                box = box_preds[mask.nonzero().squeeze()]
                box = box if box.dim() == 2 else box.unsqueeze(0)

                score = score[mask]
                
                if applyNms:
                    keep = box_nms(box, score, nms_thresh)                                            
                else:
                    keep = torch.ones_like(score).byte()

                boxes.append(box[keep])
                labels.append(torch.LongTensor(len(box[keep])).fill_(i+1))
                scores.append(score[keep])

            if len(boxes) == 0:
                boxes = torch.zeros( (0,4) )
                labels = torch.zeros( (0) )
                scores = torch.zeros( (0) )
            else:
                boxes = torch.cat(boxes, 0)
                labels = torch.cat(labels, 0)
                scores = torch.cat(scores, 0)
                
            return boxes, labels, scores
        
        except Exception as ex:
            import torchcv.utils.trace_error
            pdb.set_trace()
