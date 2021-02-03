'''SSD model with VGG16 as feature extractor.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchcv.utils import meshgrid
#from torchcv.models import StaticFusion, AdaptiveFusion

import math
import pdb

import numpy as np

BOXES = np.array([ [ [ 0.    ,    0.    ,   30.0,       30.0],
                   [   0.    ,    0.    ,   42.0,       42.0],
                   [   0.    ,    0.    ,   21.0,       42.0],      # 30, 1/2
                   [   0.    ,    0.    ,   15.0,       60.0],
                   [   0.    ,    0.    ,   28.3,       56.6],      # 40, 1/2
                   [   0.    ,    0.    ,   35.4,       70.7] ],    # 50, 1/2
                                            
                 [ [   0.    ,    0.    ,   60.0,       60.0],      
                   [   0.    ,    0.    ,   81.0,       81.0],
                   [   0.    ,    0.    ,   42.0,       84.0],      # 60, 1/2
                   [   0.    ,    0.    ,   30.0,      120.0],
                   [   0.    ,    0.    ,   56.6,      113.1],      # 80, 1/2
                   [   0.    ,    0.    ,   70.7,      141.4] ],    # 100, 1/2

                 [ [   0.    ,    0.    ,  111.0,      111.0],
                   [   0.    ,    0.    ,  134.0,      134.0],
                   [   0.    ,    0.    ,   78.5,      157.0],      # 111, 1/2
                   [   0.    ,    0.    ,   55.0,      222.0],
                   [   0.    ,    0.    ,   88.4,      176.8],      # 125, 1/2
                   [   0.    ,    0.    ,  102.5,      205.1] ],    # 145, 1/2
                   
                 [ [   0.    ,    0.    ,  162.0,      162.0],
                   [   0.    ,    0.    ,  185.0,      185.0],
                   [   0.    ,    0.    ,  114.6,      229.1],      # 162, 1/2
                   [   0.    ,    0.    ,   81.0,      324.0],
                   [   0.    ,    0.    ,  127.3,      254.6],      # 180, 1/2
                   [   0.    ,    0.    ,  141.4,      282.8] ],    # 200, 1/2

                 [ [   0.    ,    0.    ,  213.0,      213.0],
                   [   0.    ,    0.    ,  237.0,      237.0],
                   [   0.    ,    0.    ,  152.0,      298.0],      # 213, 1/2
                   [   0.    ,    0.    ,  106.0,      426.0],
                   [   0.    ,    0.    ,  162.6,      325.3],      # 230, 1/2
                   [   0.    ,    0.    ,  176.8,      353.6] ],    # 250, 1/2

                 [ [   0.    ,    0.    ,  264.0,      264.0],
                   [   0.    ,    0.    ,  288.0,      288.0],
                   [   0.    ,    0.    ,  188.0,      369.0],      # 264, 1/2
                   [   0.    ,    0.    ,  132.0,      528.0],
                   [   0.    ,    0.    ,  212.1,      424.3],      # 300, 1/2
                   [   0.    ,    0.    ,  247.5,      495.0] ],    # 350, 1/2

                 [ [   0.    ,    0.    ,  315.0,      315.0],
                   [   0.    ,    0.    ,  500.0,      500.0],
                   [   0.    ,    0.    ,  222.7,      445.5],      # 315, 1/2
                   [   0.    ,    0.    ,  157.5,      630.0],
                   [   0.    ,    0.    ,  282.8,      565.7],      # 400, 1/2
                   [   0.    ,    0.    ,  318.2,      636.4] ] ])   # 450, 1/2

BOXES[:,:,(0,2)] /= 960.
BOXES[:,:,(1,3)] /= 768.

BOXES[:,:,(0,2)] *= 320.
BOXES[:,:,(1,3)] *= 256.



class MSSDPed(nn.Module):
    
    # input_size = [512, 640]    
    # fm_sizes= [[64, 80], [32, 40], [16, 20], [8, 10], [8, 10], [8, 10]]
    # steps = (8, 16, 32, 64, 64, 64, 64)

    # anchor_areas = (20*20., 40*40., 80*80, 100*100., 140*140., 180*180.)  # Like SM Anchor
    # aspect_ratios = (1/2., 1/1.)
    # scale_ratios = (1., pow(2,1/3.), pow(2,2/3.))

    # num_anchors = (6, 6, 6, 6, 6, 6)    
    # in_channels = (512, 512, 512, 512, 512, 512)



    ### 18.10.20 half resolution
    input_size = [256, 320]
    # anchor_ref_size = [512, 640]
    fm_sizes= [[32, 40], [16, 20], [8, 10], [4, 5], [4, 5], [4, 5]]
    steps = (8, 16, 32, 64, 64, 64, 64)

    # anchor_areas = (10*10., 20*20., 40*40, 50*50., 70*70., 90*90.)  # Like SM Anchor    
    # anchor_areas = (20*20., 40*40., 80*80, 100*100., 140*140., 180*180.)  # Like SM Anchor
    aspect_ratios = (1/2., 1/1.)
    scale_ratios = (1., pow(2,1/3.), pow(2,2/3.))

    num_anchors = (6, 6, 6, 6, 6, 6)    
    in_channels = (512, 512, 512, 512, 512, 512)
    

    def __init__(self, num_classes, adaptiveFusion):
        super(MSSDPed, self).__init__()
        self.num_classes = num_classes       

        self.extractor = VGG16ExtractorPed(adaptiveFusion)
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*4, kernel_size=3, padding=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]*self.num_classes, kernel_size=3, padding=1)]                            

    def forward(self, x, y):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x, y)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            # loc_pred = self.loc_layers(x)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0),-1,4))

            cls_pred = self.cls_layers[i](x)
            # cls_pred = self.cls_layers(x)
            cls_pred = cls_pred.permute(0,2,3,1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0),-1,self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds

    def _get_anchor_index(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_idx = []
        idx = 0
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar                
                for sr in self.scale_ratios:  # scale                    
                    anchor_idx.append(idx)
                    idx += 1

        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_idx).view(num_fms, -1, 1)

    def _get_manual_anchor_wh(self):
        num_fms = len(self.fm_sizes)
        return torch.from_numpy( BOXES[:num_fms,:,2:] ).view( num_fms, -1, 2 ).float()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                # w = ar * h
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])

        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.fm_sizes)
        anchor_wh = self._get_anchor_wh()
        # anchor_wh = self._get_manual_anchor_wh()
        
        # fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes
        fm_sizes = self.fm_sizes
        input_size = self.input_size
            
        boxes = []
        for i in range(num_fms):
            num_anchor = self.num_anchors[i]

            fm_size = fm_sizes[i]

            grid_size = [ input_size[0] / fm_size[0], input_size[1] / fm_size[1] ]

            assert len( set(grid_size) ) == 1
            grid_size = grid_size[0]

            fm_w, fm_h = int(fm_size[1]), int(fm_size[0])
            xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,num_anchor,2)
            wh = anchor_wh[i].view(1,1,num_anchor,2).expand(fm_h,fm_w,num_anchor,2)

            xy = xy.float()            
            
            box = torch.cat([xy, wh], 3)  # [x,y,x,y]
            # box = torch.cat([xy-wh/2.,xy+wh/2.], 3)  # [x,y,x,y]
            # box = torch.cat([xy-wh/2.,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))

            # pdb.set_trace()

        aboxes = torch.cat(boxes, 0)
        # aboxes[:,0] /= input_size[1]
        # aboxes[:,1] /= input_size[0]
        # aboxes[:,2] /= self.anchor_ref_size[1]
        # aboxes[:,3] /= self.anchor_ref_size[0]
        aboxes[:,0] /= input_size[1]
        aboxes[:,1] /= input_size[0]
        aboxes[:,2] /= input_size[1]
        aboxes[:,3] /= input_size[0]
        return aboxes

    def __str__(self):
        return '{:s}_{:d}x{:d}'.format(self.__class__.__name__, self.input_size[0], self.input_size[1])



class VGG16ExtractorPed(nn.Module):
    def __init__(self, adaptiveFusion):
        super(VGG16ExtractorPed, self).__init__()
                
        self.adaptiveFusion = adaptiveFusion
        ######## Feature map 1. ########
        ### ~ Conv3_3        
        # vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        
        self.feat_rgb = nn.ModuleList([
            CBRBlock(3, use_bn=True, cfg=[64, 64], remove_act=True),
            CBRBlock(64, use_bn=True, cfg=[128, 128, 'M', 256, 256, 256, 'M'])
            ])
        self.feat_lwir = nn.ModuleList([
            CBRBlock(1, use_bn=True, cfg=[64, 64], remove_act=True),
            CBRBlock(64, use_bn=True, cfg=[128, 128, 'M', 256, 256, 256, 'M'])
            ])

        #fusion_module = []
        #if adaptiveFusion[0]:        
        #    fusion_module.append( AdaptiveFusion(64, 256, 256, 1, 'Conv') )
        #else:
        #    fusion_module.append( StaticFusion(256, 256, 1, 'Conv') )

        #if adaptiveFusion[1]:
        #    fusion_module.append( AdaptiveFusion(64, 256, 256, 1, 'Conv') )
        #else:
        #    fusion_module.append( StaticFusion(256, 256, 1, 'Conv') )           

        #self.fusion = nn.ModuleList( fusion_module )

        ### Conv4
        self.conv4 = CBRBlock(256, use_bn=True, cfg=[512, 512, 512, 'M'])        
        self.conv4.layers[-1].kernel_size=3
        self.conv4.layers[-1].padding=1
        self.conv4.layers[-1].stride=1

        ######## Feature map 2. ########
        self.conv5 = CBRBlock(512, use_bn=True, cfg=[512, 512, 512, 'M'])
        self.conv6 = CBRBlock(512, cfg=[512, 'F512'])

        ######## Feature map 3. ########
        self.conv7 = CBRBlock(512, cfg=['F256', 512])
        self.conv7.layers[-2].stride=2

        ######## Feature map 4. ########
        # self.conv8 = CBRBlock(512, cfg=['F256', 256])
        self.conv8 = CBRBlock(512, cfg=['F256', 512])
        self.conv8.layers[-2].stride=2                

        ######## Feature map 5. ########
        # self.conv9 = CBRBlock(256, cfg=['F128', 256])
        self.conv9 = CBRBlock(512, cfg=['F256', 512])

        ######## Feature map 6. ########
        # self.conv10 = CBRBlock(256, cfg=['F128', 256])
        self.conv10 = CBRBlock(512, cfg=['F256', 512])

        # ######## Feature map 7. ########
        # self.conv11 = CBRBlock(256, cfg=['F128', 256])
        

    def forward(self, x, y):        
        hs = []
                
        # x = self.feat_rgb[0](x)
        # x0 = F.max_pool2d( F.relu(x), 2, 2, ceil_mode=True )
        # hx = self.feat_rgb[1](x0)

        # y = self.feat_lwir[0](y)
        # y0 = F.max_pool2d( F.relu(y), 2, 2, ceil_mode=True )
        # hy = self.feat_lwir[1](y0)

        #if isinstance(self.fusion[0], AdaptiveFusion):
        #    torch.set_grad_enabled(False)

        for ii in range( len(self.feat_rgb[0].layers) ):
            x = self.feat_rgb[0].layers[ii](x)
            y = self.feat_lwir[0].layers[ii](y)

            if ii == 3:
                x0, y0= x, y

        # x = self.feat_rgb[0](x)
        x = F.max_pool2d( F.relu(x), 2, 2, ceil_mode=True )
        hx = self.feat_rgb[1](x)

        # y = self.feat_lwir[0](y)
        y = F.max_pool2d( F.relu(y), 2, 2, ceil_mode=True )
        hy = self.feat_lwir[1](y)
    
        #if isinstance(self.fusion[0], AdaptiveFusion):
        #    torch.set_grad_enabled(True)

        #hx = self.fusion[0](x0, y0, hx)
        #hy = self.fusion[1](x0, y0, hy)        
        #h = hx + hy
        h = hx + hy

        h = self.conv4(h)
        hs.append(h)  # conv4_3
        
        h = self.conv5(h)       ## To use pre-trained weight for conv5
        h = self.conv6(h)        
        hs.append(h)  # conv6

        h = self.conv7(h)        
        hs.append(h)  # conv7

        h = self.conv8(h)
        hs.append(h)  # conv8

        h = self.conv9(h)
        hs.append(h)  # conv9

        h = self.conv10(h)
        hs.append(h)  # conv10

        # h = self.conv11(h)        
        # hs.append(h)

        return hs

##########################################################################################
##########################################################################################

class VGG16BN(nn.Module):
    default_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    def __init__(self, in_ch=3, cfg=None):
        super(VGG16BN, self).__init__()
        
        if cfg is None:
            cfg = self.default_cfg

        self.layers = self._make_layers(cfg, in_ch)

    def forward(self, x):
        y = self.layers(x)
        return y

    def _make_layers(self, cfg, in_ch=3):
        '''VGG16 layers.'''
        
        layers = []     
        in_channels = in_ch   
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]                
            else:
                layers += [ nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(x, affine=True),
                            nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)


class VGG16(nn.Module):
    default_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    def __init__(self, in_ch=3, cfg=None):
        super(VGG16, self).__init__()
        
        if cfg is None:
            cfg = self.default_cfg

        self.layers = self._make_layers(cfg, in_ch)


    def forward(self, x):
        y = self.layers(x)
        return y

    def _make_layers(self, cfg, in_ch=3):
        '''VGG16 layers.'''
        
        layers = []     
        in_channels = in_ch   
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)


class L2Norm(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
        
    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None,:,None,None]
        return scale * x

class CBRBlock(nn.Module):
    default_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]   ## Conv4
    def __init__(self, in_ch, use_bn=False, cfg=None, remove_act=False):
        super(CBRBlock, self).__init__()
        
        if cfg is None:
            cfg = self.default_cfg

        self.use_bn = use_bn
        # self.bias = False if use_bn else True
        self.bias = True
        self.remove_act = remove_act
        self.layers = self._make_layers(cfg, in_ch)
        
    def forward(self, x):
        y = self.layers(x)
        return y

    def _make_layers(self, cfg, in_ch):
        '''VGG16 layers.'''
        
        layers = []             
        in_channels = in_ch   
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            elif isinstance(x, str) and 'F' in x:
                x = int(x[1:])
                layers += [ nn.Conv2d(in_channels, x, kernel_size=1) ]
                layers += [ nn.ReLU(inplace=True) ]
                in_channels = x            
            else:                             
                layers += [ nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias) ]
                if self.use_bn:
                    layers += [ nn.BatchNorm2d(x, affine=True) ]
                layers += [ nn.ReLU(inplace=True) ]
                            
                in_channels = x

        if self.remove_act:
            layers = layers[:-1]
            return nn.ModuleList(layers)
        else:
            return nn.Sequential(*layers)


def weights_init(m):
    if isinstance(m, nn.Conv2d):        
        init.normal_(m.weight.data, std=0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

