from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# import pdb
from torchcv.utils import one_hot_embedding

class SSDLoss(nn.Module):
    def __init__(self, num_classes, num_instance_per_batch=None, use_focal_loss=False):
        super(SSDLoss, self).__init__()        

        self.num_classes = num_classes
        self.num_instance_per_batch = num_instance_per_batch
        self.use_focal_loss = use_focal_loss

        self.register_buffer('num_neg', torch.zeros(1))
        self.register_buffer('num_pos', torch.zeros(1))


        
    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices (= {self.num_instance_per_batch} - {num_pos})

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''

        cls_loss = cls_loss * (pos.float() - 1)
        
        _, idx = cls_loss.sort()  # sort by negative losses
        _, rank = idx.sort()      # [N,#anchors]
                    
        num_neg = torch.clamp( self.num_instance_per_batch - pos.sum(), min=0 )
        neg = rank < num_neg        

        self.num_neg = neg.sum()
        self.num_pos = pos.sum()


        return neg      
            

    def _focal_loss(self, x, y):
        '''Focal loss.

        This is described in the original paper.
        With BCELoss, the background should not be counted in num_classes.

        Args:
          x: (tensor) predictions, sized [N,D].
          y: (tensor) targets, sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y-1, self.num_classes)
        p = x.sigmoid()
        pt = torch.where(t>0, p, 1-p)    # pt = p if t > 0 else 1-p
        w = (1-pt).pow(gamma)
        w = torch.where(t>0, alpha*w, (1-alpha)*w)
        # loss = F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
        loss = F.binary_cross_entropy_with_logits(x, t, w, size_average=True)
        return loss


    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0  # [N,#anchors]
        # batch_size = pos.size(0)
        num_pos = pos.sum().item()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================    
        if num_pos == 0:    # If there is no positive instance, ignore loc_loss.
            loc_loss = torch.tensor(0., device=pos.device)            
        else:
            mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]                    
            loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=True)

        #===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        #===============================================================
        if self.use_focal_loss:            
            
            pos_neg = cls_targets > -1  # exclude ignored anchors
            mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
            masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)

            cls_loss = self._focal_loss(masked_cls_preds, cls_targets[pos_neg])

        else:

            cls_loss = F.cross_entropy(cls_preds.view(-1,self.num_classes), \
                                       cls_targets.view(-1), reduce=False, ignore_index=-1)  # [N*#anchors,]
                
            pos = pos.view(-1)
            neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]                        

            # cls_loss = cls_loss[pos|neg].mean()    
            cls_loss_pos = cls_loss[pos]
            cls_loss_neg = cls_loss[neg]

        ### This normalization is crucial to avoid divergence
        # loc_loss: averaged over # of pos
        # cls_loss: averaged over # of pos+neg
        # return loc_loss, cls_loss
        return loc_loss, cls_loss_pos, cls_loss_neg
