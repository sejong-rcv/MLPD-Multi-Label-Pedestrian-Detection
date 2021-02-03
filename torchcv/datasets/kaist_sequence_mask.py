"""
    KAIST Multispectral Pedestrian Dataset Classes        
        Written by: Soonmin Hwang
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import pdb
from collections import namedtuple

# DB_ROOT = '/media/rcvlab/HDD2TB/datasets/kaist-rgbt/'       # DL124
# DB_ROOT = '/media/kdjoo/HDD4TB/dataset/kaist-cvpr15/'       # DL133
# DB_ROOT = '/media/user/HDD4TB/datasets/kaist-cvpr15/'       # DL178
DB_ROOT = '/HDD2/soonminh/datasets/kaist-cvpr15/'           # DL76
# DB_ROOT = '/media/rcvlab/New4TB/datasets/kaist-rgbt/'       # DL34

DAY_NIGHT_CLS = {
    'set00': 1, 'set01': 1, 'set02': 1,
    'set03': 0, 'set04': 0, 'set05': 0,
    'set06': 1, 'set07': 1, 'set08': 1,
    'set09': 0, 'set10': 0, 'set11': 0,
}



OBJ_CLASSES = [ '__ignore__',   # Object with __backgroun__ label will be ignored.
                'person', 'cyclist', 'people', 'person?']
OBJ_IGNORE_CLASSES = [ 'cyclist', 'people', 'person?' ]
OBJ_CLS_TO_IDX = { cls:(num-1) for num, cls in enumerate(OBJ_CLASSES)}

OBJ_LOAD_CONDITIONS = {    
    'train': {'hRng': (25, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)},
    # 'train': {'hRng': (25, np.inf), 'xRng':(0, 1), 'yRng':(0, 1)},    
}

# # for making bounding boxes pretty
# COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
#           (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

#### General
IMAGE_MEAN = (0.3465,  0.3219,  0.2842)
IMAGE_STD = (0.2358, 0.2265, 0.2274)

LWIR_MEAN = (0.1598)
LWIR_STD = (0.0813)

classInfo = namedtuple('TASK', 'detection')
NUM_CLASSES = classInfo( len(set(OBJ_CLASSES)-set(OBJ_IGNORE_CLASSES)) ) # Including background


class KAISTPedSeqMask(data.Dataset):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """

    def __init__(self, image_set, box_encoder, time_window, sampling=1, img_transform=None, co_transform=None, condition='train'):

        assert condition in OBJ_LOAD_CONDITIONS
        
        self.use_index_jitter = True if 'train' in image_set else False

        self.image_set = image_set
        self.img_transform = img_transform
        self.co_transform = co_transform        
        self.box_encoder = box_encoder
        self.cond = OBJ_LOAD_CONDITIONS[condition]

        self.time_window = time_window
        self.sampling = sampling

        self._parser = LoadBox()  
        self._gen_mask = MaskFromBox()      

        # {SET_ID}/{VID_ID}/{IMG_ID}.jpg        
        self._annopath = os.path.join('%s', 'annotations-xml-new-sanitized', '%s', '%s', '%s.xml')
        # {SET_ID}/{VID_ID}/{MODALITY}/{IMG_ID}.jpg
        self._imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')  

        self.ids = list()
        for line in open(os.path.join(DB_ROOT, 'imageSets', image_set)):
            self.ids.append((DB_ROOT, line.strip().split('/')))

    def __str__(self):
        return self.__class__.__name__ + '_' + self.image_set

    def __getitem__(self, index):        
        ### Check index
        offset = np.random.randint(self.sampling) if self.use_index_jitter else 0
        index *= self.sampling
        index += offset

        while True:
            # indices = [ min(idx, len(self.ids)-1) for idx in list( range(index, index+self.time_window) ) ]
            indices = [ max(0, min( idx, len(self.ids)-1) ) for idx in list( range(index-self.time_window+1, index+1) ) ]
            vid_names = set([ '_'.join(self.ids[idx][-1][:2]) for idx in indices ])
            if len(vid_names) == 1:
                break
            else:
                index = index + 1

        vis, lwir, seg, loc_target, cls_target = self.pull_item(indices)
        return vis, lwir, seg, loc_target, cls_target, torch.tensor(indices)

        # vis, lwir, loc, cls = [], [], [], []
        # for idx in index:
        #     blob = self.pull_item(idx)
        #     vis.append(blob[0])
        #     lwir.append(blob[1])
        #     loc.append(blob[2])
        #     cls.append(blob[3])

        # # return vis, lwir, loc_target, cls_target, index
        # return vis, lwir, loc_target, cls_target, index

    def __len__(self):
        # return len(self.ids)
        return int( len(self.ids)/self.sampling )

    def pull_item(self, index):            
        
        vis_list, lwir_list, seg_list, boxes_list = [], [], [], []
        for idx in index:
            frame_id = self.ids[idx]
            target = ET.parse(self._annopath % ( *frame_id[:-1], *frame_id[-1] ) ).getroot()
            
            set_id, vid_id, img_id = frame_id[-1]

            # isDay = DAY_NIGHT_CLS[set_id]

            vis = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ) )
            lwir = Image.open( self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ) ).convert('L')
                    
            width, height = vis.size
            
            boxes = self._parser(target, width, height)
            seg = self._gen_mask( (height, width), boxes.copy())

            vis_list.append(vis)
            lwir_list.append(lwir)
            seg_list.append(seg)
            boxes_list.append(boxes)            
            # mask_list.append(mask)


        ## Apply transforms
        if self.img_transform is not None:
            vis_list, lwir_list, seg_list, _ = self.img_transform(vis_list, lwir_list, seg_list, boxes_list)
        
        if self.co_transform is not None:                    
            # image, lane, boxes = self.co_transform(image, lane, boxes)
            vis_list, lwir_list, seg_list, boxes_list = self.co_transform(vis_list, lwir_list, seg_list, boxes_list)
                    

        for bb in range(len(index)):
            ## Set ignore flags
            ignore = torch.zeros( boxes_list[bb].size(0), dtype=torch.uint8)
            for ii, box in enumerate(boxes_list[bb]):
                x = box[0] * width
                y = box[1] * height
                w = ( box[2] - box[0] ) * width
                h = ( box[3] - box[1] ) * height

                if  x < self.cond['xRng'][0] or \
                    y < self.cond['xRng'][0] or \
                    x+w > self.cond['xRng'][1] or \
                    y+h > self.cond['xRng'][1] or \
                    w < self.cond['wRng'][0] or \
                    w > self.cond['wRng'][1] or \
                    h < self.cond['hRng'][0] or \
                    h > self.cond['hRng'][1]:

                    ignore[ii] = 1

            boxes_list[bb][ignore, 4] = -1

        # ## Encoding
        # _vis, _lwir, _loc, _cls = [], [], [], []
        # for vis, lwir, boxes in zip(vis_list, lwir_list, boxes_list):
        #     blob = self.box_encoder(vis, lwir, boxes)
        #     _vis.append(blob[0])
        #     _lwir.append(blob[1])
        #     _loc.append(blob[2])
        #     _cls.append(blob[3])
        ## Encoding
        # pdb.set_trace()

        _, _, loc_target, cls_target = self.box_encoder(None, None, boxes_list[-1])

        # _vis, _lwir, _loc, _cls = [], [], [], []
        # for vis, lwir, boxes in zip(vis_list, lwir_list, boxes_list):
            
        #     _vis.append(blob[0])
        #     _lwir.append(blob[1])
        #     _loc.append(blob[2])
        #     _cls.append(blob[3])

        # vis, lwir, loc_target, cls_target
        # return torch.stack(vis_list, 0), torch.stack(lwir_list, 0), torch.cat(seg_list, 0).squeeze(1).long(), loc_target, cls_target.long()
        return torch.stack(vis_list, 0), torch.stack(lwir_list, 0), seg_list[-1].squeeze(0).long(), loc_target, cls_target.long()
        # return torch.stack(_vis, 0), torch.stack(_lwir, 0), torch.stack(_loc, 0), torch.stack(_cls, 0).long()

        # ## Apply transforms
        # if self.img_transform is not None:
        #     vis, lwir, _ = self.img_transform(vis, lwir)
        
        # if self.co_transform is not None:                    
        #     # image, lane, boxes = self.co_transform(image, lane, boxes)
        #     vis, lwir, boxes = self.co_transform(vis, lwir, boxes)
                    
        # ## Set ignore flags
        # ignore = torch.zeros( boxes.size(0), dtype=torch.uint8)
        # for ii, box in enumerate(boxes):
        #     x = box[0] * width
        #     y = box[1] * height
        #     w = ( box[2] - box[0] ) * width
        #     h = ( box[3] - box[1] ) * height

        #     if  x < self.cond['xRng'][0] or \
        #         y < self.cond['xRng'][0] or \
        #         x+w > self.cond['xRng'][1] or \
        #         y+h > self.cond['xRng'][1] or \
        #         w < self.cond['wRng'][0] or \
        #         w > self.cond['wRng'][1] or \
        #         h < self.cond['hRng'][0] or \
        #         h > self.cond['hRng'][1]:

        #         ignore[ii] = 1

        # boxes[ignore, 4] = -1

        # ## Encoding
        # vis, lwir, loc_target, cls_target = self.box_encoder(vis, lwir, boxes)

        # return vis, lwir, loc_target, cls_target.long()              
        

class LoadBox(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, bbs_format='xyxy'):
        # self.class_to_ind = class_to_ind or dict(
        #     zip(KAIST_CLASSES, range(len(KAIST_CLASSES))))

        assert bbs_format in ['xyxy', 'xywh']                
        self.bbs_format = bbs_format
        self.pts = ['x', 'y', 'w', 'h']

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """                
        res = [ [0, 0, 0, 0, -1] ]
        for obj in target.iter('object'):            
            name = obj.find('name').text.lower().strip()            
            bbox = obj.find('bndbox')
                        
            label_idx = OBJ_CLS_TO_IDX[name] if name not in OBJ_IGNORE_CLASSES else -1
            bndbox = [ int(bbox.find(pt).text) for pt in self.pts ]

            ### squarify (hold height, modify bounding box to ar==.41 = w/h)
            # if label_idx == 0:  # Squarify 'person' only
            #     bw = float(bndbox[2])
            #     bh = float(bndbox[3])

            #     sq_bw = bh * 0.41
            #     sq_dd = sq_bw - bw

            #     bndbox[0] -= sq_dd / 2.0
            #     bndbox[2] += sq_dd
            
            if self.bbs_format in ['xyxy']:
                bndbox[2] = min( bndbox[2] + bndbox[0], width )
                bndbox[3] = min( bndbox[3] + bndbox[1], height )

            
            bndbox = [ cur_pt / width if i % 2 == 0 else cur_pt / height for i, cur_pt in enumerate(bndbox) ]
            
            bndbox.append(label_idx)
            # bndbox.append( int(obj.find('occlusion').text) )
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind, occ]
            
        return np.array(res, dtype=np.float)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]



class MaskFromBox(object):
    
    def __call__(self, im_size, boxes):
        mask = np.zeros( im_size, dtype=np.uint8 )

        ## Assume, box coordinate is [ x1, y1, x2, y2 ]    
        boxes[:,(0,2)] *= im_size[1]
        boxes[:,(1,3)] *= im_size[0]

        for b in boxes[1:]:                        
            x1, y1, x2, y2 = b[:4].astype(np.uint16)
            mask[ y1:y2, x1:x2 ] = 1
    

        img = Image.fromarray(mask)
        # img = Image.fromarray(mask, mode='P')
        # img.putpalette([
        #         0, 0, 0, # black background
        #         255, 0, 0, # index 1 is red
        #         255, 255, 0, # index 2 is yellow
        #         255, 153, 0, # index 3 is orange
        #     ])

        return img


def SeqCollate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """

    img1 = []
    img2 = []
    loc_target = []
    cls_target = []    
    index = []

    for sample in batch:
        img1.append(sample[0])
        img2.append(sample[1])
        loc_target.append(sample[2])
        cls_target.append(sample[3])        
        index.append(sample[4])
            
    return torch.cat(img1, 0), torch.cat(img2, 0), torch.stack(loc_target, 0), torch.stack(cls_target, 0), index



if __name__ == '__main__':

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # from torchcv.datasets import UnNormalize, Compose, ToTensor, ToPILImage, Normalize, Resize, RandomHorizontalFlip, RandomResizedCrop, ColorJitter
        # from torchcv.datasets import SynthFail, FaultTolerant, ColorJitterLWIR
        from torchcv.datasets.transforms3 import *
        from torchcv.models.ssd import SSD300, SSD512, MSSDPed
        from torchcv.models import SSDBoxCoder, MVSSDPedDeform
        from torchcv.visualizations import draw_boxes

        # img_size = 512
        # net = MSSDPed(2, False)
        fusion_params = [ {'type': 'sum'}]
        net = MVSSDPedDeform(NUM_CLASSES.detection, 2, fusion_params)

        img_size = net.input_size
        preprocess = Compose([  ColorJitter(0.5, 0.5, 0.3), ColorJitterLWIR(contrast=0.5) ])
        # preprocess.add( [ SynthFail('bug1.png', (512, 640), 'T-') ] )
        # preprocess.add( [ FaultTolerant([0.5, 0.5]) ] )

        ori_size = (512, 640)

        transforms = Compose([  \
                                RandomHorizontalFlip(), \
                                RandomResizedCrop( img_size, scale=(0.25, 2.0), ratio=(0.8, 1.2)), \
                                ResizeMask( ori_size ), \
                                ToTensor(), \
                                Normalize([0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
                                Normalize([0.1598], [0.0813], 'T')
                                ])

        # trainset = KAISTPed('train-all-04.txt', SSDBoxCoder(net), img_transform=preprocess, co_transform=transforms)
        trainset = KAISTPedSeqMask('train-all-02.txt', SSDBoxCoder(net), 3, img_transform=preprocess, co_transform=transforms)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=32)


        
        tensor2image = Compose( [UnNormalize((0.3465,0.3219,0.2842), (0.2358,0.2265,0.2274)), ToPILImage('RGB'), Resize(ori_size)])
        tensor2lwir = Compose( [UnNormalize([0.1598], [0.0813]), ToPILImage('L'), Resize(ori_size)])
        coder = SSDBoxCoder(net)

        fig, ax = plt.subplots(figsize=(6,5))


        gen_mask = MaskFromBox()
        # seg_palette = [
        #         0, 0, 0, # black background
        #         255, 0, 0, # index 1 is red
        #         255, 255, 0, # index 2 is yellow
        #         255, 153, 0, # index 3 is orange
        #     ]
        seg_palette = np.array( [
                [0, 0, 0], # black background
                [255, 0, 0], # index 1 is red
                [255, 255, 0], # index 2 is yellow
                [255, 153, 0], # index 3 is orange
            ])

        # ################################################################################################
        # ### Compute mean/std for CLS/REG
        # std_loc = torch.zeros( (0, 4) )        
        # for ii, blob in enumerate(trainloader):
            
        #     vis, lwir, loc_gt, cls_gt, idx = blob
        #     std_loc = torch.cat( (std_loc, loc_gt[ cls_gt > 0 ].view(-1,4)) )
        #     if ii and ii % 10 == 0:
        #         print('ii: {} / {}'.format(ii, len(trainloader)))

        #     if ii and ii % 1000 == 0:
        #         pdb.set_trace()

        # std_loc.mean(dim=0)        
        # ################################################################################################

        # pdb.set_trace()

        # ################################################################################################
        # ### Compute anchors (clustering)        
        # from sklearn.cluster import KMeans        
        # kmeans = KMeans(init='k-means++', n_clusters=4, random_state=170)

        # # gt_boxes = []
        # # for ii, (vis, lwir, loc_target, cls_target, index) in enumerate(trainset):

        # #     vis = np.array( tensor2image( vis.clone() )[0] )
        # #     lwir = np.array( tensor2lwir( lwir.clone() )[0] )

        # #     cls_target_gt = torch.zeros( (cls_target.size(0), 2), dtype=torch.uint8 )
        # #     cls_target_gt.scatter_(1, cls_target.unsqueeze(1), 1)
        # #     boxes, labels, scores = coder.decode(loc_target, cls_target_gt)

        # #     if len(boxes):
        # #         boxes[:,(0,2)] *= img_size[1]
        # #         boxes[:,(1,3)] *= img_size[0]

        # #         gt_boxes.append(boxes)

        # #     if ii and ii % 100 == 0:
        # #         print('{}/{}'.format(ii, len(trainset)))
        
        # # bbs = np.concatenate( gt_boxes, axis=0 )
        # # bbs[:,(2,3)] = bbs[:,(2,3)] - bbs[:,(0,1)]
        # # np.save( 'gt_boxes.npy', bbs )

        # bbs = np.load('gt_boxes.npy')

        # plt.clf()
        # plt.plot( np.log(bbs[:,2]), np.log(bbs[:,3]), 'r.' )
        # plt.savefig('log-wh-distribution_{:s}.png'.format(OBJ_CLASSES[1].replace(' ', '_')))
        
        # areas = bbs[:,2] * bbs[:,3]
        
        # nums, ctrs = np.histogram( np.log(areas), 200, normed=True)
        # cdf = nums.cumsum() # cumulative distribution function
        # cdf = cdf / cdf[-1] # normalize

        # log_area_bins = [ np.log(areas.min()) ]
        # num_scales = 7
        # for ss in range(1, num_scales): 
        #     idx = np.where( cdf < 1/num_scales*ss )[0][-1]
        #     log_area_bins.append( ( ctrs[idx] + ctrs[idx+1] ) / 2 )

        # log_area_bins.append( np.log(areas.max()) )
        # log_area_bins = np.array( log_area_bins )


        # plt.clf()        
        # nums, ctrs, _ = plt.hist( np.log(areas), log_area_bins )
        # plt.savefig('area-histogram_{:s}.png'.format(OBJ_CLASSES[1].replace(' ', '_')))
        

        # height_scales = np.sqrt( np.exp( log_area_bins ) * 2 )
        # height_scales[0] = -np.inf
        # height_scales[-1] = np.inf


        # print( 'cluster centers: \n' )
        # plt.clf()        

        # colors = np.zeros( len(bbs), dtype=np.uint8 )
        # for ss in range( len(height_scales)-1 ):
        #     cond = ( bbs[:,3] > height_scales[ss] ) * ( bbs[:,3] < height_scales[ss+1] )
        #     log_width = np.log( bbs[cond, 2])
        #     log_height = np.log( bbs[cond, 3])

        #     y_pred = kmeans.fit_predict( np.stack([log_width, log_height], axis=1) )

        #     colors[cond] = y_pred + ss*4

        #     centers = np.exp( kmeans.cluster_centers_.copy() )
        #     for jj, prior in enumerate(centers):
        #         start = '[' if jj == 0 else ' '
        #         end = ']' if jj == kmeans.cluster_centers_.shape[0] else ','
        #         print('{:s}[ {:>6.2f}, {:>6.2f}, {:>6.2f}, {:>6.2f} ]{:}'.format(start, 0., 0., prior[0], prior[1], end))

        #     plt.scatter( np.log(centers[:,0]), np.log(centers[:,1]), marker='x', s=169, c='r', zorder=20)

        
        # plt.scatter( np.log(bbs[:,2]), np.log(bbs[:,3]), c=colors, cmap=plt.cm.Set1)
        # plt.ylabel('log-height')
        # plt.xlabel('log-width')
        # plt.savefig('kmeans-logscale-dist_{:s}.png'.format(OBJ_CLASSES[1]))                       
        # ################################################################################################

        # pdb.set_trace()


        ################################################################################################
        ### Validity check
        num_pos = 0.
        num_gt = 1.        

        # blob = trainset[0]
        # pdb.set_trace()

        for ii, (vis, lwir, seg, loc_target, cls_target, index) in enumerate(trainloader):
            # for ii, (vis, lwir, seg, loc_target, cls_target, index) in enumerate(trainset):
            #     continue
            B, T, C, H, W = vis.shape

            vis_seq = tensor2image( [ vv for vv in vis.clone().view(B*T, C, H, W) ] )[0]
            lwir_seq = tensor2lwir( [ ll for ll in lwir.clone().view(B*T, 1, H, W) ] )[0]

            # vis_img = np.array( tensor2image( vis.clone()[-1,...] )[0][0] )
            # lwir_img = np.array( tensor2lwir( lwir.clone()[-1,...] )[0][0] )

            cls_target_gt = torch.zeros( (cls_target[-1].size(0), 3), dtype=torch.uint8 )
            cls_target_gt.scatter_(1, cls_target[-1].unsqueeze(1) + 1, 1)
            boxes, labels, scores = coder.decode(loc_target[-1], cls_target_gt[:,1:])            
            anchors = coder.decode(torch.zeros_like(loc_target[-1]), cls_target_gt[:,1:], applyNms=False)

            

            if len(boxes) > 0:
                
                for jj, (vis_img, lwir_img) in enumerate( zip(vis_seq, lwir_seq) ):

                    kk = jj // trainset.time_window

                    if (jj+1) % trainset.time_window == 0:
                        
                        cls_target_gt = torch.zeros( (cls_target[kk].size(0), 3), dtype=torch.uint8 )
                        cls_target_gt.scatter_(1, cls_target[kk].unsqueeze(1) + 1, 1)
                        boxes, labels, scores = coder.decode(loc_target[kk], cls_target_gt[:,1:])
                        anchors = coder.decode(torch.zeros_like(loc_target[kk]), cls_target_gt[:,1:], applyNms=False)                        


                        try:
                            boxes[:,(0,2)] *= ori_size[1]            
                            boxes[:,(1,3)] *= ori_size[0]       

                            aboxes = anchors[0]
                            aboxes[:,(0,2)] *= ori_size[1]            
                            aboxes[:,(1,3)] *= ori_size[0]     

                            alabels = anchors[1]  
                            ascores = anchors[2]

                        except:                        
                            boxes = torch.zeros(0)
                            labels = torch.zeros(0)
                            scores = torch.zeros(0)

                            aboxes = torch.zeros(0)
                            alabels = torch.zeros(0)
                            ascores = torch.zeros(0)

                    else:
                        boxes = torch.zeros(0)
                        labels = torch.zeros(0)
                        scores = torch.zeros(0)

                        aboxes = torch.zeros(0)
                        alabels = torch.zeros(0)
                        ascores = torch.zeros(0)

                    filename = '_'.join( trainset.ids[index[kk][jj % trainset.time_window]][-1] )
                    print('name: {}, image size: {}x{}x{}'.format(filename, *vis.shape))            
                                        
                    draw_boxes(ax, np.array(vis_img), boxes, labels, scores, OBJ_CLASSES)
                    plt.savefig( filename + '_vis.jpg' )


                    draw_boxes(ax, 255*seg[kk, jj% trainset.time_window].squeeze().byte().numpy(), boxes, labels, scores, OBJ_CLASSES)
                    plt.savefig( filename + '_seg.jpg' )

                    # seg_img = Image.fromarray(seg[kk, jj% trainset.time_window].squeeze().byte().numpy(), mode='P')
                    # seg_img.putpalette(seg_palette)
                    # seg_img.save( filename + '_seg.png' )
                    
                    if len(boxes):
                        pdb.set_trace()
                    # if len(boxes):
                    #     gen_mask( ori_size, boxes ).save( filename + '_mask.png' )
                    
                    # draw_boxes(ax, np.array(lwir_img), boxes, labels, scores, OBJ_CLASSES)
                    # plt.savefig( filename + '_lwir.jpg' )                    
                    
                    # draw_boxes(ax, np.array(lwir_img), aboxes, alabels, ascores)
                    # plt.savefig( filename + '_lwir_anchors.jpg' )

                pdb.set_trace()

                num_gt += len(boxes)
                num_pos += len(aboxes)

            if ii and ii % 100 == 0:
                print('Avg. of pos per gt: {}'.format(num_pos/num_gt ))
                # pdb.set_trace()

        # pdb.set_trace()

        # for ii, (vis, lwir, loc_target, cls_target, index) in enumerate(trainset):
        #     # pdb.set_trace()

        #     vis_seq = tensor2image( [ vv for vv in vis.clone() ] )[0]
        #     lwir_seq = tensor2lwir( [ ll for ll in lwir.clone() ] )[0]

        #     # vis_img = np.array( tensor2image( vis.clone()[-1,...] )[0][0] )
        #     # lwir_img = np.array( tensor2lwir( lwir.clone()[-1,...] )[0][0] )

        #     cls_target_gt = torch.zeros( (cls_target.size(0), 3), dtype=torch.uint8 )
        #     cls_target_gt.scatter_(1, cls_target.unsqueeze(1) + 1, 1)
        #     boxes, labels, scores = coder.decode(loc_target, cls_target_gt[:,1:])
            
        #     anchors = coder.decode(torch.zeros_like(loc_target), cls_target_gt[:,1:], applyNms=False)

            

        #     if len(boxes) > 0:

        #         # pdb.set_trace()

        #         for jj, (vis_img, lwir_img) in enumerate( zip(vis_seq, lwir_seq) ):

        #             # kk = jj // trainset.time_window

        #             if (jj+1) % trainset.time_window == 0:
                        
        #                 cls_target_gt = torch.zeros( (cls_target.size(0), 3), dtype=torch.uint8 )
        #                 cls_target_gt.scatter_(1, cls_target.unsqueeze(1) + 1, 1)
        #                 boxes, labels, scores = coder.decode(loc_target, cls_target_gt[:,1:])                        
        #                 anchors = coder.decode(torch.zeros_like(loc_target), cls_target_gt[:,1:], applyNms=False)                        

        #                 boxes[:,(0,2)] *= ori_size[1]            
        #                 boxes[:,(1,3)] *= ori_size[0]       

        #                 aboxes = anchors[0]
        #                 aboxes[:,(0,2)] *= ori_size[1]            
        #                 aboxes[:,(1,3)] *= ori_size[0]     

        #                 alabels = anchors[1]  
        #                 ascores = anchors[2]

        #             else:                        
        #                 boxes = torch.zeros(0)
        #                 labels = torch.zeros(0)
        #                 scores = torch.zeros(0)

        #                 aboxes = torch.zeros(0)
        #                 alabels = torch.zeros(0)
        #                 ascores = torch.zeros(0)

        #             filename = '_'.join( trainset.ids[index[jj % trainset.time_window]][-1] )
        #             print('name: {}, image size: {}'.format(filename, 'x'.join( [str(ss) for ss in vis.shape] ) ))
                                        
        #             draw_boxes(ax, np.array(vis_img), boxes, labels, scores, OBJ_CLASSES)
        #             plt.savefig( filename + '_vis.jpg' )

        #             # draw_boxes(ax, np.array(lwir_img), boxes, labels, scores, OBJ_CLASSES)
        #             # plt.savefig( filename + '_lwir.jpg' )                    
                    
        #             # draw_boxes(ax, np.array(lwir_img), aboxes, alabels, ascores)
        #             # plt.savefig( filename + '_lwir_anchors.jpg' )

        #         pdb.set_trace()

        #         num_gt += len(boxes)
        #         num_pos += len(aboxes)

        #     if ii and ii % 100 == 0:
        #         print('Avg. of pos per gt: {}'.format(num_pos/num_gt ))
        #         pdb.set_trace()



            # vis_img = np.array( tensor2image( vis.clone()[-1,...] )[0][0] )
            # lwir_img = np.array( tensor2lwir( lwir.clone()[-1,...] )[0][0] )

            # cls_target_gt = torch.zeros( (cls_target.size(0), 3), dtype=torch.uint8 )
            # cls_target_gt.scatter_(1, cls_target.unsqueeze(1) + 1, 1)
            # boxes, labels, scores = coder.decode(loc_target, cls_target_gt[:,1:])
            
            # anchors = coder.decode(torch.zeros_like(loc_target), cls_target_gt[:,1:], applyNms=False)

            # pdb.set_trace()

            # if len(boxes) > 0:

            #     filename = '_'.join( trainset.ids[index[-1]][-1] )
            #     print('name: {}, image size: {}x{}x{}'.format(filename, *vis.shape))            
            #     boxes[:,(0,2)] *= ori_size[1]            
            #     boxes[:,(1,3)] *= ori_size[0]       
                
            #     draw_boxes(ax, vis_img, boxes, labels+1, scores, OBJ_CLASSES)
            #     plt.savefig( filename + '_vis.jpg' )

            #     draw_boxes(ax, lwir_img, boxes, labels+1, scores, OBJ_CLASSES)
            #     plt.savefig( filename + '_lwir.jpg' )
                
            #     aboxes = anchors[0]
            #     aboxes[:,(0,2)] *= ori_size[1]            
            #     aboxes[:,(1,3)] *= ori_size[0]       
            #     draw_boxes(ax, lwir_img, aboxes, anchors[1]+1, anchors[2])
            #     plt.savefig( filename + '_lwir_anchors.jpg' )                

            #     num_gt += len(boxes)
            #     num_pos += len(aboxes)

            # if ii and ii % 100 == 0:
            #     print('Avg. of pos per gt: {}'.format(num_pos/num_gt ))
            #     pdb.set_trace()

        ################################################################################################

                
    except:
        import traceback
        ex_type, ex_value, ex_traceback = sys.exc_info()            

        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        # Format stacktrace
        stack_trace = list()

        for trace in trace_back:
            stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

        sys.stderr.write("[Error] Exception type : %s \n" % ex_type.__name__)
        sys.stderr.write("[Error] Exception message : %s \n" %ex_value)
        for trace in stack_trace:
                sys.stderr.write("[Error] (Stack trace) %s\n" % trace)

        pdb.set_trace()
