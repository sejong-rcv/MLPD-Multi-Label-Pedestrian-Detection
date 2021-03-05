import os

from easydict import EasyDict as edict

import torch
import numpy as np

from utils.transforms import *

os.environ["CUDA_VISIBLE_DEVICES"]="7"

#### General
IMAGE_MEAN = [0.3465,  0.3219,  0.2842]
IMAGE_STD = [0.2358, 0.2265, 0.2274]

LWIR_MEAN = [0.1598]
LWIR_STD = [0.0813]


## path
PATH = edict()

PATH.DB_ROOT = './datasets/kaist-rgbt/'
PATH.JSON_GT_FILE = os.path.join( PATH.DB_ROOT, 'kaist_annotations_test20.json' )

## train
train = edict()

train.day = "all"
train.img_set = f"train-{train.day}-20.txt"

# Learning parameters
# checkpoint = './jobs/2021-01-28_07h54m_SSD_KAIST_LF_Multi_Label/checkpoint_ssd300.pth.tar003'
train.checkpoint = None
train.checkpoint = './jobs/2021-03-05_11h38m_/checkpoint_ssd300.pth.tar003'
train.batch_size = 4 # batch size
train.start_epoch = 3  # start at this epoch
train.epochs = 40  # number of epochs to run without early-stopping
train.epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
train.best_loss = 100.  # assume a high loss at first

train.lr = 1e-4   # learning rate
train.momentum = 0.9  # momentum
train.weight_decay = 5e-4  # weight decay
train.grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

train.print_freq=10

train.annotation = "AR-CNN" # AR-CNN, Sanitize, Original 





## test
test = edict()

test.day = "all"
test.img_set = f"test-{test.day}-20.txt"

### coco tool용.
test.result_path = './result'

test.annotation = "AR-CNN"

### test model
test.checkpoint = "./jobs/checkpoint_ssd300.pth.tar025"

test.input_size = [512., 640.]

### test ~~ datasets config
test.batch_size = 32
test.eval_batch_size = 1

                    
### dataset
dataset = edict()
dataset.workers = 0
dataset.OBJ_LOAD_CONDITIONS = {    
                                  'train': {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                                  'test': {'hRng': (-np.inf, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                              }
### Fusion Dead Zone
FDZ_case = edict()

FDZ_case.original = ["None", "None"]

FDZ_case.blackout_R = ["blackout", "None"]
FDZ_case.blackout_T = ["None", "blackout"]

FDZ_case.SidesBlackout_a = ["SidesBlackout_R", "SidesBlackout_L"]
FDZ_case.SidesBlackout_b = ["SidesBlackout_L", "SidesBlackout_R"]
FDZ_case.SurroundingBlackout = ["None", "SurroundingBlackout"]




# main
args = edict(path=PATH,
             train=train,
             test=test,
             dataset=dataset)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.exp_time = None
args.exp_name = None

args.n_classes = 3 # BG, 

args.upaired_augmentation = ["TT_RandomHorizontalFlip",
                             "TT_FixedHorizontalFlip",
                             "TT_RandomResizedCrop"]

args["test"].img_transform = Compose([Resize(test.input_size)])    
args["test"].co_transform = Compose([ToTensor(), \
                                     Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                     Normalize(LWIR_MEAN, LWIR_STD, 'T')                        
                                    ])
### FDZ_case : original, blackout_R, blackout_T, SidesBlackout_a, SidesBlackout_b, SurroundingBlackout
FDZ = [FusionDeadZone(FDZ_case.original, tuple(test.input_size)) ]
args["test"].img_transform.add(FDZ)



args["train"].img_transform = Compose( [ColorJitter(0.3, 0.3, 0.3), 
                                        ColorJitterLWIR(contrast=0.3)])
args["train"].co_transform = Compose([TT_RandomHorizontalFlip(p=0.5, flip=0.5), 
                                    TT_RandomResizedCrop([512,640], \
                                                          scale=(0.25, 4.0), \
                                                          ratio=(0.8, 1.2)), 
                                    ToTensor(), \
                                    Normalize( [0.3465,0.3219,0.2842], 
                                                [0.2358,0.2265,0.2274], 'R'),
                                    Normalize([0.1598], [0.0813], 'T')],
                                    args=args)