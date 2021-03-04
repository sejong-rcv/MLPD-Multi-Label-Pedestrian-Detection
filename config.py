import os

from easydict import EasyDict as edict

import torch
import numpy as np


path = edict()

path.DB_ROOT = './datasets/kaist-rgbt/'
path.JSON_GT_FILE = os.path.join( path.DB_ROOT, 'kaist_annotations_test20.json' )


## eval
eval = edict()
# Parameters
#eval.data_folder = './datasets/kaist-rgbt/' -> path.DB_ROOT

eval.keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
# eval.batch_size = 64 -> dataset.batch_size
# eval.workers = 4 -> ..
# eval.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") -> ..
eval.checkpoint = '/raid/tjkim/workspace/journal/code/MLPD-Multi-Label-Pedestrian-Detection-Backup/jobs/checkpoint_ssd300.pth.tar024' 


# # Load test data
# eval.preprocess1 = Compose([ Resize([300, 300])])    
# eval.transforms1 = Compose([ ToTensor(), \
#                         Normalize([0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
#                         Normalize([0.1598], [0.0813], 'T')                        
#                         ])


### dataset
dataset = edict()

dataset.DAY_NIGHT_CLS = {
                            'set00': 1, 'set01': 1, 'set02': 1,
                            'set03': 0, 'set04': 0, 'set05': 0,
                            'set06': 1, 'set07': 1, 'set08': 1,
                            'set09': 0, 'set10': 0, 'set11': 0,
                        }

dataset.OBJ_CLASSES = [ '__ignore__',   # Object with __backgroun__ label will be ignored.
                      'person', 'cyclist', 'people', 'person?', 'unpaired']
dataset.OBJ_IGNORE_CLASSES = [ 'cyclist', 'people', 'person?' , 'unpaired']

# OBJ_CLS_TO_IDX = { cls:1 if cls =='person' or cls == 'cyclist' or cls == 'people' \
#                     or cls == 'unpaired' else -1 for num, cls in enumerate(OBJ_CLASSES)}
# dataset.OBJ_CLS_TO_IDX = { cls:1 if cls =='person' else -1 for num, cls in enumerate(OBJ_CLASSES)}

dataset.OBJ_LOAD_CONDITIONS = {    
                                  'train': {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                                  'test': {'hRng': (-np.inf, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                              }

#### General
dataset.IMAGE_MEAN = (0.3465,  0.3219,  0.2842)
dataset.IMAGE_STD = (0.2358, 0.2265, 0.2274)

dataset.LWIR_MEAN = (0.1598)
dataset.LWIR_STD = (0.0813)

# dataset.classInfo = namedtuple('TASK', 'detection')

# dataset.tensor2image = Compose( [UnNormalize((0.3465,0.3219,0.2842), (0.2358,0.2265,0.2274)), ToPILImage('RGB'), Resize([512,640])])
# dataset.tensor2lwir = Compose( [UnNormalize([0.1598], [0.0813]), ToPILImage('L'), Resize([512,640])])


args = edict(dataset=dataset, path=path, eval=eval)
