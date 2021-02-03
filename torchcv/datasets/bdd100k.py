"""
    BDD100k Dataset loader for PyTorch
    	written by Soonmin Hwang    
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
# if sys.version_info[0] == 2:
#     import xml.etree.cElementTree as ET
# else:
#     import xml.etree.ElementTree as ET

import json
from collections import namedtuple
import pdb

# DB_ROOT = os.path.abspath( os.path.join(os.path.dirname(__file__), 'BDD100k') )
DB_ROOT = '/media/rcvlab/New4TB/datasets/BDD100k/bdd100k'


#### Weather classification
WEATHER_CLASSES = [ 'undefined', \
					'rainy', 'snowy', 'clear', 'overcast', 'partly cloudy', \
					'foggy' ]
WEATHER_CLS_TO_IDX = { cls:num for num, cls in enumerate(WEATHER_CLASSES)}

#### Time (Day/Night) classification
TIME_CLASSES = [	'undefined', \
					'daytime', 'night', 'dawn/dusk']
TIME_CLS_TO_IDX = { cls:num for num, cls in enumerate(TIME_CLASSES)}

#### Scene classification
SCENE_CLASSES = [	'undefined', \
					'tunnel', 'residential', 'parking lot', 'city street', 'gas stations', \
					'highway']
SCENE_CLS_TO_IDX = { cls:num for num, cls in enumerate(SCENE_CLASSES)}


#### Object detection
"""
	Example of annotation,
		
		{
            "category": "car",
            "attributes": {
                "occluded": false,
                "truncated": false,
                "trafficLightColor": "none"
            },
            "manualShape": true,
            "manualAttributes": true,
            "box2d": {
                "x1": 566.725826,
                "y1": 386.403971,
                "x2": 596.679623,
                "y2": 413.362386
            },
            "id": 227
        },

"""
OBJ_CLASSES = [	'__ignore__', 	# Object with __backgroun__ label will be ignored.
				'bus', 'traffic light', 'traffic sign', 'person', 'bike', \
				'truck', 'motor', 'car', 'train', 'rider']
OBJ_CLS_TO_IDX = { cls:(num-1) for num, cls in enumerate(OBJ_CLASSES)}

OBJ_LOAD_CONDITIONS = {
    # 'train': {'hRng': (30, np.inf), 'xRng':(5, -5), 'yRng':(5, -5)},
    # 'train': {'hRng': (16, np.inf), 'xRng':(5, -5), 'yRng':(5, -5)},
    'train': {'hRng': (16, np.inf), 'xRng':(0, 0), 'yRng':(0, 0)},
    'Near': {'hRng': (115, np.inf), 'vRng': (0), 'xRng':(5, 635), 'yRng':(5, 475)}
}


#### Lane detection
"""
	Example of annotation,

		{
            "category": "lane",
            "attributes": {
                "laneDirection": "parallel",
                "laneStyle": "solid",
                "laneType": "road curb"
            },
            "manualShape": true,
            "manualAttributes": true,
            "poly2d": [
                {
                    "vertices": [
                        [
                            305.268154,
                            449.929388
                        ],
                        [
                            209.389704,
                            468.664488
                        ],
                        [
                            106.898947,
                            468.664488
                        ],
                        [
                            0,
                            469.766552
                        ]
                    ],
                    "types": "CCCL",
                    "closed": false
                }
            ],
            "id": 236
        },

"""

LANE_CLASSES = [	'__background__', \
					'road curb', 'crosswalk', 'double white', 'double yellow', 'double other color', \
					'single white', 'single yellow', 'single other color']
LANE_CLS_TO_IDX = { cls:num for num, cls in enumerate(LANE_CLASSES)}


#### General
IMAGE_MEAN = (0.3465,  0.3219,  0.2842)
IMAGE_STD = (0.2358, 0.2265, 0.2274)

classInfo = namedtuple('TASK', 'weather scene time detection lane')
NUM_CLASSES = classInfo(len(WEATHER_CLASSES), len(SCENE_CLASSES), len(TIME_CLASSES), len(OBJ_CLASSES), len(LANE_CLASSES)) # Including background




class BDD100kDataset(data.Dataset):
	"""

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

	def __init__(self, phase, img_transform=None, mask_transform=None, co_transform=None):

		assert phase in ['train', 'val']
		
		# self.image_set = image_sets
		self.img_transform = img_transform
		self.mask_transform = mask_transform
		self.co_transform = co_transform				

		# {SET_ID}/{VID_ID}/{IMG_ID}.jpg
		self._image_path = os.path.join(DB_ROOT, 'images', '100k', phase,  '%s')
		self._label_path = os.path.join(DB_ROOT, 'labels', 'bdd100k_labels_images_%s.json' % phase)
		
		self._label = json.load(open(self._label_path, 'r'))

		self._box_parser = ParseBox( OBJ_LOAD_CONDITIONS['train'] )
		self._lane_parser = ParseLane()

		# self.ids = list()
		# for iset in image_sets:
		# 	# self.name += '_' + name + '-' + skip
		# 	for line in open(os.path.join(self.root, 'ImageSets', iset)):
		# 		self.ids.append( tuple(line.strip().split('/')) )

	def __getitem__(self, index):
		# image, lanemask, boxes, weather, scene, time, info = self.pull_item(index)        
		# return image, lanemask, boxes, weather, scene, time, info, index

		image, loc_target, cls_target = self.pull_item(index)        
		return image, loc_target, cls_target


	def __len__(self):
		return len(self._label)

	def pull_item(self, index):
		frame = self._label[index]

		image = Image.open( self._image_path % frame['name'] )

		# ## GT for auxiliary task: weather/scene/time classification
		# image_attr = frame['attributes']
		# # weather, scene, time = image_attr['weather'], image_attr['scene'], image_attr['timeofday']
		# weather = WEATHER_CLS_TO_IDX[ image_attr['weather'] ]
		# scene = SCENE_CLS_TO_IDX[ image_attr['scene'] ]
		# time = TIME_CLS_TO_IDX[ image_attr['timeofday'] ]

		# ## Info. 
		width, height = image.size
		# imageInfo = {'height': height, 'width': width, 'filename': frame['name']}


		## Parse annotations
		boxes, lanes = [], []		
		for label in frame['labels']:
			if label['category'] in OBJ_CLASSES:
				boxes.append( self._box_parser(label, width, height) )
			elif label['category'] in LANE_CLASSES:
				pass
				# lanes.append( self._lane_parser(label) )

		lane = Image.new('I', image.size)
		boxes = np.concatenate(boxes, axis=0)	           
		
		## Apply transforms
		if self.img_transform is not None:
			image, _, _ = self.img_transform(image)

		if self.mask_transform is not None:
			_, lane, _ = self.mask_transform(lane)

		if self.co_transform is not None:                    
			# image, lane, boxes = self.co_transform(image, lane, boxes)
			image, _, loc_target, cls_target = self.co_transform(image, lane, boxes)

		# return image, lane.long(), boxes, weather, scene, time, imageInfo		
		return image, loc_target, cls_target.long()


def multitask_collate_fn(batch):
	"""Custom collate fn for dealing with batches of images that have a different targets/info, etc.
    
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) batch of target masks stacked on their 0 dim
            3) (dictionary) batch of information for input images, e.g. height/width, scene class
            4) (list of integers) batch of image indices
	"""
	image, mask, boxes, weather, scene, time, info, index = [], [], [], [], [], [], [], []
	for sample in batch:		
		image.append(sample[0])
		mask.append(sample[1])    
		boxes.append(sample[2])    

		weather.append(torch.ByteTensor( [sample[3]] ))
		scene.append(torch.ByteTensor( [sample[4]] ))
		time.append(torch.ByteTensor( [sample[5]] ))

		info.append(sample[6])        
		index.append(sample[7])

	return torch.stack(image, 0), torch.stack(mask, 0), boxes, \
		torch.stack(weather, 0), torch.stack(scene, 0), torch.stack(time, 0), info, index


class ParseLane(object):
	pass

class ParseBox(object):
	"""
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
	"""
	def __init__(self, condition=None, bbs_format='xyxy'):

		assert bbs_format in ['xyxy', 'xywh']

		self.bbs_format = bbs_format  
		self.condition = condition      
		# self.info_names = namedtuple('Info', 'video_name frame_idx ext image_size')

	def __call__(self, obj, img_width, img_height):
		"""
		Arguments:
		    target (annotation) : the target annotation to be made usable
		        will be an ET.Element
		Returns:
		    a list containing lists of bounding boxes  [bbox coords, class name]
		"""

		if not 'box2d' in obj or obj['box2d'] is None:			
			raise RuntimeError('Object must have box2d attribute. but, {:s}'.format(str(obj)))

		cond = self.condition

		## Load
		box = obj['box2d']
		if box is not None:
			box = np.array([ box['x1'], box['y1'], box['x2'], box['y2'], OBJ_CLS_TO_IDX[ obj['category'] ] ], dtype=np.float)
		else:
			box = np.zeros(5, dtype=np.float)		# Dummy box because 'torch.stack' does not work for empty array
			box[-1] = -1

		## Check condition
		# if obj['attributes']['occluded'] or obj['attributes']['truncated']:
		# if obj['attributes']['truncated']:
		# 	box[-1] = OBJ_CLS_TO_IDX['__ignore__']

		# Too small/big, or on boundary
		x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]

		if ( obj['category'] not in ['traffic sign', 'traffic light'] ) and \
			(h < cond['hRng'][0] or h > cond['hRng'][1] \
			or x < cond['xRng'][0] or x+w > img_width+cond['xRng'][1] \
			or y < cond['yRng'][0] or y+h > img_height+cond['yRng'][1]):
			box[-1] = OBJ_CLS_TO_IDX['__ignore__']

		box[0] /= img_width
		box[2] /= img_width
		box[1] /= img_height
		box[3] /= img_height

		return np.reshape(box, (1,5))


# def draw_boxes(ax, im, boxes, thres=0.9, filename=None):

# 	ax.imshow(im.astype(np.uint8))
# 	cls_color = ['gray', 'yellow','cyan','red','green','blue','white', 'yellow','cyan','red','green','blue','white']

# 	inds = np.where(boxes[:,4]>thres)[0]
# 	if len(inds) == 0:
# 		return

# 	for box in boxes:		
# 		label = box[5] if len(box) > 5 else box[4]		# len(box) > 5, then box is prediction
# 		score = box[4] if len(box) > 5 else None

# 		label = int(label)
		
# 		ax.add_patch(
# 		    plt.Rectangle((box[0], box[1]),
# 		                  box[2] - box[0],
# 		                  box[3] - box[1], fill=False,
# 		                  edgecolor=cls_color[label], linewidth=2.5)
# 		    )
# 		if label:
# 			ax.text(box[0], box[1] - 2,
# 		    	    '{:s} {:.3f}'.format( OBJ_CLASSES[label], score) if score else '{:s}'.format(OBJ_CLASSES[label]),
# 		        	bbox=dict(facecolor='blue', alpha=0.5),
# 		        	fontsize=14, color='white')

# 	plt.axis('off')
# 	plt.tight_layout()

# 	if filename is not None:
# 		plt.savefig(filename)

def draw_boxes(ax, im, boxes, labels, scores, thres=0.9, filename=None):

	ax.imshow(im.astype(np.uint8))
	cls_color = ['gray', 'yellow','cyan','red','green','blue','white', 'yellow','cyan','red','green','blue','white']

	# inds = np.where(boxes[:,4]>thres)[0]
	# if len(inds) == 0:
	# 	return

	for box, label, score in zip(boxes, labels, scores):

		# label = box[5] if len(box) > 5 else box[4]		# len(box) > 5, then box is prediction
		# score = box[4] if len(box) > 5 else None

		if score < thres:
			continue
		# label = int(label)
		
		ax.add_patch(
		    plt.Rectangle((box[0], box[1]),
		                  box[2] - box[0],
		                  box[3] - box[1], fill=False,
		                  edgecolor=cls_color[label], linewidth=2.5)
		    )
		if label:
			ax.text(box[0], box[1] - 2,
		    	    '{:s} {:.2f}'.format( OBJ_CLASSES[label], score) if score else '{:s}'.format(OBJ_CLASSES[label]),
		        	bbox=dict(facecolor='blue', alpha=0.5),
		        	fontsize=10, color='white')

	plt.axis('off')
	plt.tight_layout()

	if filename is not None:
		plt.savefig(filename)



if __name__ == '__main__':

	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	from torchcv.datasets import UnNormalize, Compose, ToTensor, ToPILImage, Normalize, Resize, RandomHorizontalFlip, RandomResizedCrop, ColorJitter
	from torchcv.models.ssd import SSD300, SSD512, SSDBoxCoder

	img_size = 512
	net = SSD512(num_classes=11)
	preprocess = Compose([  ColorJitter(0.5, 0.5, 0.3)])
	transforms = Compose([  RandomHorizontalFlip(), \
	                        RandomResizedCrop( (img_size,img_size), scale=(0.5, 2.0), ratio=(0.8, 1.2)), \
	                        ToTensor(), \
	                        Normalize((0.3465,0.3219,0.2842), (0.2358,0.2265,0.2274)), \
	                        SSDBoxCoder(net)
	                        ])

	trainset = BDD100kDataset('train', img_transform=preprocess, co_transform=transforms)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=32)

	ori_size = (720, 1280)
	tensor2image = Compose( [UnNormalize((0.3465,0.3219,0.2842), (0.2358,0.2265,0.2274)), ToPILImage('RGB'), Resize(ori_size)])
	coder = SSDBoxCoder(net)


	fig, ax = plt.subplots(figsize=(12,7))

	
	std_loc = torch.zeros( (0, 4) )
	# for ii, blob in enumerate(trainset):
	for ii, blob in enumerate(trainloader):
		# print('ii: {}'.format(ii))
		
		# ax.cla()

		image, loc_gt, cls_gt = blob

		std_loc = torch.cat( (std_loc, loc_gt[ cls_gt > 0 ].view(-1,4)) )
		# std_loc[ii] = (loc_gt * loc_gt ).mean(dim=0).mean(dim=0)

		if ii and ii % 10 == 0:
			print('ii: {} / {}'.format(ii, len(trainloader)))


		if ii and ii % 1000 == 0:
			pdb.set_trace()

		# # sz = image.size(1)

		# cls_pred_gt = torch.zeros( (cls_gt.size(0), 11), dtype=torch.uint8 )
		# cls_pred_gt.scatter_(1, cls_gt.unsqueeze(1), 1)
		# boxes, labels, scores = coder.decode(loc_gt, cls_pred_gt)

		# # boxes *= sz
		# boxes[:,(0,2)] *= ori_size[1]
		# boxes[:,(1,3)] *= ori_size[0]

		# draw_boxes( ax, np.array( tensor2image(image.clone())[0] ), boxes, labels+1, scores )

		# plt.savefig('test.jpg')

	pdb.set_trace()
				
	std_loc.mean(dim=0) - mean_loc*mean_loc

	

	# # ### Load all boxes, cluster boxes to determine prior boxes
	# # with open( 'BDD100k/labels/bdd100k_labels_images_train.json', 'r') as f:
	# # 	frames = json.load(f)

	# # boxes = [ [] for _ in range(len(OBJ_CLASSES)-1) ]
	
	# # for frame in frames:
	# # 	for label in frame['labels']:
	# # 		if label['category'] in OBJ_CLASSES:
	# # 			box = label['box2d']
	# # 			cls = OBJ_CLS_TO_IDX[ label['category'] ]
	# # 			bb = np.array( [[ box['x1'], box['y1'], box['x2'], box['y2'] ]] )
	# # 			boxes[cls-1].append( bb )

	# # from sklearn.cluster import KMeans
	# # # kmeans = KMeans(init='k-means++', n_clusters=9, random_state=170)
	# # kmeans = KMeans(init='k-means++', n_clusters=12, random_state=170)

	# # ## Select ratios	
	# # for ii, bbs in enumerate(boxes):
	# # 	bbs = np.concatenate( bbs, axis=0 )
	# # 	bbs[:,(2,3)] = bbs[:,(2,3)] - bbs[:,(0,1)]

	# # 	plt.clf()
	# # 	plt.plot( bbs[:,2], bbs[:,3], 'r.' )
	# # 	plt.savefig('analysis/wh-distribution_{:s}.png'.format(OBJ_CLASSES[ii+1].replace(' ', '_')))

	# # 	plt.clf()
	# # 	nums, ctrs, _ = plt.hist( bbs[:,2] / bbs[:,3], np.linspace(0, 4, 50) )
	# # 	plt.savefig('analysis/ratio-histogram_{:s}.png'.format(OBJ_CLASSES[ii+1].replace(' ', '_')))

	# # 	print( 'CLS: {:s}, RATIO (>20\%): {:}'.format(OBJ_CLASSES[ii+1], ctrs[ list(nums / sum(nums) > 0.05) + [False] ]) )

	# # 	# area_norm = np.sqrt( bbs[:,2] * bbs[:,3] / 100 )
	# # 	# area_norm = np.power( 2, np.floor(np.log(np.sqrt(bbs[:,2]*bbs[:,3]))/np.log(2) + 0.5) )
	# # 	area_norm = np.power( 2, np.clip( np.floor(np.log(np.sqrt(bbs[:,2]*bbs[:,3]))/np.log(2) + 0.5), a_min=3, a_max=6 ) )
	# # 	log_width = np.log( bbs[:,2] / area_norm )
	# # 	log_height = np.log( bbs[:,3] / area_norm )

		
	# # 	y_pred = kmeans.fit_predict( np.stack([log_width, log_height], axis=1) )
	# # 	# plt.clf()
	# # 	# plt.scatter(log_width, log_height, c=y_pred, cmap=plt.cm.Paired)
	# # 	# # plt.scatter( bbs[:,2], bbs[:,3], c=y_pred, cmap=plt.cm.Paired)
		
	# # 	# for cc, step in enumerate([8, 16, 32, 64]):
	# # 	# 	centers = np.exp( kmeans.cluster_centers_.copy() ) * step
	# # 	# 	plt.scatter( centers[:,0], centers[:,1], marker='x', s=169, c=cc*np.ones(centers.shape[0]), cmap=plt.cm.Set1, zorder=10)
	# # 	# # plt.scatter( kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='x', s=169, color='w', zorder=10 )
	# # 	# plt.ylabel('height')
	# # 	# plt.xlabel('width')
	# # 	# plt.savefig('analysis/kmeans-log-dist_{:s}.png'.format(OBJ_CLASSES[ii+1]))

	# # 	print( 'cluster centers: \n' )
	# # 	for jj, prior in enumerate( np.exp( kmeans.cluster_centers_.copy() ) ):
	# # 		start = '[' if jj == 0 else ' '
	# # 		end = ']' if jj == kmeans.cluster_centers_.shape[0] else ' '
	# # 		print('{:s}[ 0., 0., {:.4f}, {:.4f} ]{:}'.format(start, prior[0], prior[1], end))


	# # 	plt.clf()
	# # 	# plt.scatter(log_width, log_height, c=y_pred, cmap=plt.cm.Paired)
	# # 	plt.scatter( bbs[:,2], bbs[:,3], c=y_pred, cmap=plt.cm.Paired)
		
	# # 	# for cc, step in enumerate([8, 16, 32, 64]):
	# # 	clrs = np.zeros((0), dtype=np.int32)
	# # 	centers = np.zeros((0,2), dtype=np.float64)
	# # 	for cc, step in enumerate([8, 32, 48, 64, 80, 96]):
	# # 		priors = np.exp( kmeans.cluster_centers_.copy() ) * step
	# # 		centers = np.concatenate( (centers, priors), axis=0 )
	# # 		clrs = np.concatenate( (clrs, cc*np.ones(priors.shape[0], dtype=np.int32)), axis=0 )

	# # 	plt.scatter( centers[:,0], centers[:,1], marker='x', s=169, c=clrs, cmap=plt.cm.Set1, zorder=20)
	# # 	# plt.scatter( centers[:,0], centers[:,1], marker='x', s=169, c=(cc)*np.ones(centers.shape[0], dtype=np.int32), cmap=plt.cm.Set1, zorder=cc)
	# # 	# plt.scatter( kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='x', s=169, color='w', zorder=10 )
	# # 	plt.ylabel('height')
	# # 	plt.xlabel('width')
	# # 	plt.savefig('analysis/kmeans-real-scale-dist_{:s}.png'.format(OBJ_CLASSES[ii+1]))

	# # 	# pdb.set_trace()


	# # 	# ## Prepare prior boxes
	# # 	# priorboxes = []
	# # 	# image_size = [720, 1280]
	# # 	# steps = [8, 16, 32, 64, 64]
	# # 	# ratios = ctrs[ list(nums / sum(nums) > 0.05) + [False] ]
	# # 	# for ii, ss in enumerate(steps):        
	# # 	# 	f_k_h = image_size[0] / ss
	# # 	# 	f_k_w = image_size[1] / ss

	# # 	# 	# unit center x,y
	# # 	# 	# cx = (ss*0.5) / f_k_w
	# # 	# 	# cy = (ss*0.5) / f_k_h
	# # 	# 	box = np.zeros( (1,4) )

	# # 	# 	for ww, hh in [ (f_k_w, f_k_w), (f_k_w, f_k_h), (f_k_h, f_k_h)]:
	# # 	# 		box[0,2] = ww
	# # 	# 		box[0,3] = hh
	# # 	# 		priorboxes.append(box.copy())

	# # 	# 		for rr in ratios:
	# # 	# 			box[0,2] = ww * np.sqrt(rr)
	# # 	# 			box[0,3] = hh / np.sqrt(rr)
	# # 	# 			priorboxes.append(box.copy())

	# # 	# 			box[0,2] = ww / np.sqrt(rr)
	# # 	# 			box[0,3] = hh * np.sqrt(rr)
	# # 	# 			priorboxes.append(box.copy())

	# # 	# pdb.set_trace()

	
	# # pdb.set_trace()

    
	# from transforms import UnNormalize, Compose, ToPILImage, ColorJitter, RandomHorizontalFlip, ToTensor, Normalize, RandomResizedCrop

	

	# tensor2image = Compose( [UnNormalize(IMAGE_MEAN, IMAGE_STD), ToPILImage('RGB')])
	# tensor2target = ToPILImage()

	# dataset = BDD100kDataset( 'val',
	#                         img_transform=Compose([ColorJitter(0.5, 0.5, 0.3)]),
	#                         co_transform=Compose([ \
	#                             RandomHorizontalFlip(), \
	#                             RandomResizedCrop((540,960), scale=(0.5, 2.0), ratio=(0.8, 1.2)), \
	#                             ToTensor(), \
	#                             Normalize(IMAGE_MEAN, IMAGE_STD)]))

	# fig, ax = plt.subplots(figsize=(12, 12))

	# for ii in range(len(dataset)):            
	# 	# image, lane, boxes, weather, scene, time, info, index = dataset[ii]

	# 	pdb.set_trace()

	# 	image, lane, boxes, weather, scene, time, info, index = multitask_collate_fn( [dataset[ii]] )

	# 	image = np.array( tensor2image( image )[0] )        
	# 	lane = np.array( tensor2target( lane.byte() )[0] )
	# 	boxes = boxes.numpy().copy()

	# 	height, width = image.shape[:2]

	# 	if len(boxes):
	# 		ax.cla()

	# 		print('name: {}, image size: {}x{}x{}'.format(info['filename'], *image.shape))            
	# 		boxes[:,(0,2)] *= width            
	# 		boxes[:,(1,3)] *= height            
	# 		# boxes[:,4] = 1.0			
	# 		draw_boxes(ax, image, boxes, filename=os.path.basename(info['filename']))

	# 	if ii and ii % 20 == 0 :
	# 	    pdb.set_trace()

            
