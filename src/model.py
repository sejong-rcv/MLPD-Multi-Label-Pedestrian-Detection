import importlib, pdb
from math import sqrt
from itertools import product as product

from torch import nn
import torch.nn.functional as F
import torchvision


from utils.utils import *

args = importlib.import_module('config').args
device = args.device


class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    def __init__(self):
        super(VGGBase, self).__init__()

        #################################### RGB ####################################
 
        # Standard convolutional layers in VGG16
        self.conv1_1_vis = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_vis = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_vis = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_vis = nn.BatchNorm2d(64, affine=True)        
        self.pool1_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_vis = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_vis = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.pool2_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_vis = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.pool3_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.conv4_1_vis = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_1_bn_vis = nn.BatchNorm2d(512, affine=True)
        self.conv4_2_vis = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2_bn_vis = nn.BatchNorm2d(512, affine=True)
        self.conv4_3_vis = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3_bn_vis = nn.BatchNorm2d(512, affine=True)

        #############################################################################

        #################################### Thermal ####################################
 
        # Standard convolutional layers in VGG16
        self.conv1_1_lwir = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_lwir = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_lwir = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_lwir = nn.BatchNorm2d(64, affine=True)
        
        self.pool1_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_lwir = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_lwir = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_lwir = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_lwir = nn.BatchNorm2d(128, affine=True)

        self.pool2_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_lwir = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.pool3_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.conv4_1_lwir = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_1_bn_lwir = nn.BatchNorm2d(512, affine=True)
        self.conv4_2_lwir = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2_bn_lwir = nn.BatchNorm2d(512, affine=True)
        self.conv4_3_lwir = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3_bn_lwir = nn.BatchNorm2d(512, affine=True)
        
        #############################################################################

        self.pool4 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        # Replacements for FC6 and FC7 in VGG16
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True) # atrous convolution
        self.conv6_1_bn = nn.BatchNorm2d(512, affine=True)  
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1)

        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        self.conv7_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv7_2_bn = nn.BatchNorm2d(512, affine=True)  

        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        nn.init.xavier_uniform_(self.conv8_1.weight)
        nn.init.constant_(self.conv8_1.bias, 0.)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)  # stride = 1, by default
        nn.init.xavier_uniform_(self.conv8_2.weight)
        nn.init.constant_(self.conv8_2.bias, 0.)

        self.conv9_1 = nn.Conv2d(512, 256, kernel_size=1)
        nn.init.xavier_uniform_(self.conv9_1.weight)
        nn.init.constant_(self.conv9_1.bias, 0.)
        self.conv9_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv9_2.weight)
        nn.init.constant_(self.conv9_2.bias, 0.)

        self.conv10_1 = nn.Conv2d(512, 256, kernel_size=1)
        nn.init.xavier_uniform_(self.conv10_1.weight)
        nn.init.constant_(self.conv10_1.bias, 0.)
        self.conv10_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv10_2.weight)
        nn.init.constant_(self.conv10_2.bias, 0.)

        self.feat_1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_1_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_2_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_3 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_3_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_4 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_4_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_5 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_5_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_6 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_6_bn = nn.BatchNorm2d(512, momentum=0.01)
        
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image_vis, image_lwir):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
 
        ############################ RGB #####################################

        out_vis = F.relu(self.conv1_1_bn_vis(self.conv1_1_vis(image_vis)))  
        out_vis = F.relu(self.conv1_2_bn_vis(self.conv1_2_vis(out_vis))) 
        out_vis = self.pool1_vis(out_vis)  

        out_vis = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis)))
        out_vis = F.relu(self.conv2_2_bn_vis(self.conv2_2_vis(out_vis))) 
        out_vis = self.pool2_vis(out_vis) 

        out_vis = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis))) 
        out_vis = F.relu(self.conv3_2_bn_vis(self.conv3_2_vis(out_vis))) 
        out_vis = F.relu(self.conv3_3_bn_vis(self.conv3_3_vis(out_vis)))
        out_vis = self.pool3_vis(out_vis)
        
        out_vis = F.relu(self.conv4_1_bn_vis(self.conv4_1_vis(out_vis))) 
        out_vis = F.relu(self.conv4_2_bn_vis(self.conv4_2_vis(out_vis))) 
        out_vis = F.relu(self.conv4_3_bn_vis(self.conv4_3_vis(out_vis))) 
        out_vis = self.pool4(out_vis)
        ##########################################################################

        ############################ Thermal #####################################

        out_lwir = F.relu(self.conv1_1_bn_lwir(self.conv1_1_lwir(image_lwir)))  
        out_lwir = F.relu(self.conv1_2_bn_lwir(self.conv1_2_lwir(out_lwir))) 
        out_lwir = self.pool1_lwir(out_lwir)  

        out_lwir = F.relu(self.conv2_1_bn_lwir(self.conv2_1_lwir(out_lwir)))
        out_lwir = F.relu(self.conv2_2_bn_lwir(self.conv2_2_lwir(out_lwir))) 
        out_lwir = self.pool2_lwir(out_lwir) 

        out_lwir = F.relu(self.conv3_1_bn_lwir(self.conv3_1_lwir(out_lwir))) 
        out_lwir = F.relu(self.conv3_2_bn_lwir(self.conv3_2_lwir(out_lwir))) 
        out_lwir = F.relu(self.conv3_3_bn_lwir(self.conv3_3_lwir(out_lwir))) 
        out_lwir = self.pool3_lwir(out_lwir)

        out_lwir = F.relu(self.conv4_1_bn_lwir(self.conv4_1_lwir(out_lwir))) 
        out_lwir = F.relu(self.conv4_2_bn_lwir(self.conv4_2_lwir(out_lwir))) 
        out_lwir = F.relu(self.conv4_3_bn_lwir(self.conv4_3_lwir(out_lwir))) 
        out_lwir = self.pool4(out_lwir)

        #########################################################################

        conv4_3_feats = torch.cat([out_vis, out_lwir], dim=1)
        conv4_3_feats = F.relu(self.feat_1_bn(self.feat_1(conv4_3_feats)))

        #########################################################################

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors
        
        out_vis = F.relu(self.conv5_1_bn(self.conv5_1(out_vis))) 
        out_vis = F.relu(self.conv5_2_bn(self.conv5_2(out_vis))) 
        out_vis = F.relu(self.conv5_3_bn(self.conv5_3(out_vis)))
        out_vis = self.pool5(out_vis)
        out_lwir = F.relu(self.conv5_1_bn(self.conv5_1(out_lwir))) 
        out_lwir = F.relu(self.conv5_2_bn(self.conv5_2(out_lwir))) 
        out_lwir = F.relu(self.conv5_3_bn(self.conv5_3(out_lwir))) 
        out_lwir = self.pool5(out_lwir)
        
        out_vis = F.relu(self.conv6_1_bn(self.conv6_1(out_vis))) 
        out_vis = F.relu(self.conv6_2(out_vis))
        out_lwir = F.relu(self.conv6_1_bn(self.conv6_1(out_lwir))) 
        out_lwir = F.relu(self.conv6_2(out_lwir))

        #########################################################################

        conv6_feats = torch.cat([out_vis, out_lwir], dim=1)
        conv6_feats = F.relu(self.feat_2_bn(self.feat_2(conv6_feats)))

        #########################################################################

        out_vis = F.relu(self.conv7_1(out_vis))
        out_vis = F.relu(self.conv7_2_bn(self.conv7_2(out_vis))) 
        out_lwir = F.relu(self.conv7_1(out_lwir))
        out_lwir = F.relu(self.conv7_2_bn(self.conv7_2(out_lwir)))

        #########################################################################

        conv7_feats = torch.cat([out_vis, out_lwir], dim=1)
        conv7_feats = F.relu(self.feat_3_bn(self.feat_3(conv7_feats)))

        ######################################################################### 

        out_vis = F.relu(self.conv8_1(out_vis))
        out_vis = F.relu(self.conv8_2(out_vis)) 
        out_lwir = F.relu(self.conv8_1(out_lwir))
        out_lwir = F.relu(self.conv8_2(out_lwir)) 

        #########################################################################

        conv8_feats = torch.cat([out_vis, out_lwir], dim=1)
        conv8_feats = F.relu(self.feat_4_bn(self.feat_4(conv8_feats)))

        ######################################################################### 

        out_vis = F.relu(self.conv9_1(out_vis))
        out_vis = F.relu(self.conv9_2(out_vis))
        out_lwir = F.relu(self.conv9_1(out_lwir))
        out_lwir = F.relu(self.conv9_2(out_lwir))

        #########################################################################

        conv9_feats = torch.cat([out_vis, out_lwir], dim=1)
        conv9_feats = F.relu(self.feat_5_bn(self.feat_5(conv9_feats)))

        ######################################################################### 


        out_vis = F.relu(self.conv10_1(out_vis)) 
        out_vis = F.relu(self.conv10_2(out_vis)) 
        out_lwir = F.relu(self.conv10_1(out_lwir)) 
        out_lwir = F.relu(self.conv10_2(out_lwir))

        #########################################################################

        conv10_feats = torch.cat([out_vis, out_lwir], dim=1)
        conv10_feats = F.relu(self.feat_6_bn(self.feat_6(conv10_feats)))

        ######################################################################### 

        # Lower-level feature maps
        return conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats


    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """

        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[1:71]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        for i, param in enumerate(param_names[71:141]):    
   
            if param == 'conv1_1_lwir.weight':
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]][:, 0:1, :, :]              
            else:
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        for i, param in enumerate(param_names[141:162]):
               state_dict[param] = pretrained_state_dict[pretrained_param_names[i+70]]

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        n_boxes = {'conv4_3': 6,
                    'conv6': 6,
                    'conv7': 6,
                    'conv8': 6,
                    'conv9': 6,
                    'conv10': 6,}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv6 = nn.Conv2d(512, n_boxes['conv6'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(512, n_boxes['conv6'] * 4, kernel_size=3, padding=1)
        self.loc_conv8 = nn.Conv2d(512, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv9 = nn.Conv2d(512, n_boxes['conv8'] * 4, kernel_size=3, padding=1)
        self.loc_conv10 = nn.Conv2d(512, n_boxes['conv9'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv6 = nn.Conv2d(512, n_boxes['conv6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(512, n_boxes['conv6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8 = nn.Conv2d(512, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9 = nn.Conv2d(512, n_boxes['conv8'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10 = nn.Conv2d(512, n_boxes['conv9'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)


    def forward(self, conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats):

        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv6 = self.loc_conv6(conv6_feats)
        l_conv6 = l_conv6.permute(0, 2, 3, 1).contiguous()
        l_conv6 = l_conv6.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8 = self.loc_conv8(conv8_feats)
        l_conv8 = l_conv8.permute(0, 2, 3, 1).contiguous()
        l_conv8 = l_conv8.view(batch_size, -1, 4)

        l_conv9 = self.loc_conv9(conv9_feats)
        l_conv9 = l_conv9.permute(0, 2, 3, 1).contiguous()
        l_conv9 = l_conv9.view(batch_size, -1, 4)

        l_conv10 = self.loc_conv10(conv10_feats)
        l_conv10 = l_conv10.permute(0, 2, 3, 1).contiguous()
        l_conv10 = l_conv10.view(batch_size, -1, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats) 
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)


        c_conv6 = self.cl_conv6(conv6_feats)
        c_conv6 = c_conv6.permute(0, 2, 3, 1).contiguous()
        c_conv6 = c_conv6.view(batch_size, -1, self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8 = self.cl_conv8(conv8_feats)
        c_conv8 = c_conv8.permute(0, 2, 3, 1).contiguous()
        c_conv8 = c_conv8.view(batch_size, -1, self.n_classes)

        c_conv9 = self.cl_conv9(conv9_feats)
        c_conv9 = c_conv9.permute(0, 2, 3, 1).contiguous()
        c_conv9 = c_conv9.view(batch_size, -1, self.n_classes)

        c_conv10 = self.cl_conv10(conv10_feats)
        c_conv10 = c_conv10.permute(0, 2, 3, 1).contiguous()
        c_conv10 = c_conv10.view(batch_size, -1, self.n_classes)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)

        locs = torch.cat([l_conv4_3, l_conv6, l_conv7, l_conv8, l_conv9, l_conv10], dim=1)
        classes_scores = torch.cat([c_conv4_3, c_conv6, c_conv7, c_conv8, c_conv9, c_conv10],
                                   dim=1)

        return locs, classes_scores


class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image_vis, image_lwir):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv6_feats , conv7_feats, conv8_feats, conv9_feats, conv10_feats = self.base(image_vis, image_lwir)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats)  # (N, 8732, 4), (N, 8732, n_classes)
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """

        fmap_dims = {'conv4_3': [80,64],
                     'conv6': [40,32],
                     'conv7': [20,16],
                     'conv8': [10,8],
                     'conv9': [10,8],
                     'conv10': [10,8]}

        scale_ratios = {'conv4_3': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv6': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv7': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv8': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv9': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv10': [1., pow(2,1/3.), pow(2,2/3.)]}


        aspect_ratios = {'conv4_3': [1/2., 1/1.],
                         'conv6': [1/2., 1/1.],
                         'conv7': [1/2., 1/1.],
                         'conv8': [1/2., 1/1.],
                         'conv9': [1/2., 1/1.],
                         'conv10': [1/2., 1/1.]}


        anchor_areas = {'conv4_3': [40*40.],
                         'conv6': [80*80.],
                         'conv7': [160*160.],
                         'conv8': [200*200.],
                         'conv9': [280*280.],
                         'conv10': [360*360.]} 


        # fmaps = list(fmap_dims.keys())
        fmaps = ['conv4_3', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10']

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap][1]):
                for j in range(fmap_dims[fmap][0]):
                    cx = (j + 0.5) / fmap_dims[fmap][0]
                    cy = (i + 0.5) / fmap_dims[fmap][1]
                    for s in anchor_areas[fmap]:
                        for ar in aspect_ratios[fmap]: 
                            h = sqrt(s/ar)                
                            w = ar * h
                            for sr in scale_ratios[fmap]: # scale
                                anchor_h = h*sr/512.
                                anchor_w = w*sr/640.
                                prior_boxes.append([cx, cy, anchor_w, anchor_h])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        # predicted_scores = F.softmax(predicted_scores, dim=2)
        predicted_scores = torch.sigmoid(predicted_scores)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        all_images_bg_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to

            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)) 

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            image_bf_scores = list()
            
            predicted_fg_scores = predicted_scores[i][:,1:].mean(dim=1)
            score_above_min_score = predicted_fg_scores > min_score
            n_above_min_score = score_above_min_score.sum().item()

            class_scores = predicted_scores[i][:,1:][score_above_min_score]
            bg_scores = predicted_scores[i][:,0][score_above_min_score]
            class_decoded_locs = decoded_locs[score_above_min_score]
            
            # Sort predicted boxes and scores by scores\
            _, sort_ind = class_scores.mean(dim=1).sort(dim=0, descending=True)
            class_scores = class_scores[sort_ind]
            bg_scores = bg_scores[sort_ind]
            class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

            # Non-Maximum Suppression (NMS)
            suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device)  # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if suppress[box] == 1:
                    continue
                suppress = torch.max(suppress, overlap[box] > max_overlap)
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[~suppress])
            image_labels.append(torch.ones((~suppress).sum().item()).to(device))
            image_scores.append(class_scores[~suppress])
            image_bf_scores.append(bg_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))
                image_bf_scores.append(orch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_bg_scores = torch.cat(image_bf_scores, dim=0)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                _, sort_ind = image_scores.mean(dim=1).sort(dim=0, descending=True)
                image_scores = image_scores[sort_ind][:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)
                image_bg_scores = image_bg_scores[sort_ind][:top_k]

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
            all_images_bg_scores.append(image_bg_scores)

        return all_images_boxes, all_images_labels, all_images_scores, all_images_bg_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False, ignore_index=-1)
        self.loss_fn = nn.BCEWithLogitsLoss(reduce=False, reduction ='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
    
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float, device=device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long, device=device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],self.priors_xy)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            # Store
            true_classes[i] = label_for_each_prior
            
            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes > 0  # (N, 8732)

        # LOCALIZATION LOSS
        # Localization loss is computed only over positive (non-background) priors
        if true_locs[positive_priors].shape[0] == 0:
            loc_loss = torch.tensor([0], device=device)
        else:
            loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        ignore_index = torch.nonzero((true_classes.view(-1) == -1), as_tuple=False)
        pair_index = torch.nonzero((true_classes.view(-1) == 3), as_tuple=False)
        true_classes = (true_classes.view(-1)+1)
        true_classes = _to_one_hot(true_classes,n_dims=5)[:,1:4]

        if len(pair_index) != 0: 
            true_classes[pair_index,1] = 1
            true_classes[pair_index,2] = 1

        # First, find the loss for all priors
        conf_loss_all = self.loss_fn(predicted_scores.view(-1, n_classes), true_classes).sum(dim=1)
        conf_loss_all[ignore_index] = 0
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  

        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / ( 1e-10 + n_positives.sum().float() )  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss , conf_loss , loc_loss, n_positives

def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).type(dtype)        
    return zeros.scatter(scatter_dim, y_tensor, 1)
