import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont

import config
from datasets import KAISTPed
from utils.transforms import FusionDeadZone, Compose, Resize, ToTensor

# Label map
voc_labels = ('P', 'M', 'A', 'B', 'a')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#3cb44b', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

def _line(Draw, xy, dot=0, width=3, fill='#3cb44b', h=False):
    if dot != 0:
        if (xy[2] - xy[0]) > (xy[3] - xy[1]):
            h=True
        if h:
            for i in range(xy[0], xy[2]):
                if i%dot*2==0:
                    Draw.line((i, xy[1], i+4, xy[3]), width=width, fill=fill)
        else:
            for i in range(xy[1], xy[3]):
                if i%dot*2==0:
                    Draw.line((xy[0], i, xy[2], i+4), width=width, fill=fill)
    else:
        Draw.line(xy, width=width, fill=fill)

def rectangle(draw, rec, dot=0, width=2, fill='#3cb44b' ):
    rec = np.array(rec, dtype=np.int16)
    a = tuple(rec[0:2])
    b = tuple([rec[2], rec[1]])
    c = tuple([rec[0], rec[3]])
    d = tuple(rec[2:4])

    _line(draw, a+b, fill=fill, dot=dot, width=width)
    _line(draw, a+c, fill=fill, dot=dot, width=width)
    _line(draw, c+d, fill=fill, dot=dot, width=width)
    _line(draw, b+d, fill=fill, dot=dot, width=width)

def detect(original_image, original_lwir, detection, \
        min_score=0.5, max_overlap=0.425, top_k=200, \
        suppress=None, width=2):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    
    det_boxes = detection[:,1:5]

    small_object =  det_boxes[:, 3] < 55
  
    det_boxes[:,2] = det_boxes[:,0] + det_boxes[:,2]
    det_boxes[:,3] = det_boxes[:,1] + det_boxes[:,3] 
    det_scores = detection[:,5]
    det_labels = list()
    for i in range(len(detection)) : 
        det_labels.append(1.0)
    det_labels = np.array(det_labels)
    det_score_sup = det_scores < min_score    
    det_boxes = det_boxes[~det_score_sup]
    det_scores = det_scores[~det_score_sup]
    det_labels = det_labels[~det_score_sup]
    
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels]

    # PIL from Tensor
    original_image = original_image.squeeze().permute(1, 2, 0)
    original_image = original_image.numpy() * 255
    original_lwir = original_lwir.squeeze().numpy() * 255
    original_image = Image.fromarray(original_image.astype(np.uint8))
    original_lwir = Image.fromarray(original_lwir.astype(np.uint8))


    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        new_image = Image.new('RGB',(2*original_image.size[0], original_image.size[1]))
        new_image.paste(original_image,(0,0))
        new_image.paste(original_lwir,(original_image.size[0],0))
        return new_image

    # Annotate
    annotated_image = original_image
    annotated_image_lwir = original_lwir
    draw = ImageDraw.Draw(annotated_image)
    draw_lwir = ImageDraw.Draw(annotated_image_lwir)
    font = ImageFont.truetype("./utils/calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.shape[0]):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
                
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]], width=width)
        draw_lwir.rectangle(xy=box_location, outline=label_color_map[det_labels[i]], width=width)
        
        # Text       
        text_score_vis = str(det_scores[i].item())[:7]
        text_score_lwir = str(det_scores[i].item())[:7]
        
        text_size_vis = font.getsize(text_score_vis)
        text_size_lwir = font.getsize(text_score_lwir)

        text_location_vis = [box_location[0] + 2., box_location[1] - text_size_vis[1]]
        textbox_location_vis = [box_location[0], box_location[1] - text_size_vis[1], box_location[0] + text_size_vis[0] + 4.,box_location[1]]
        
        text_location_lwir = [box_location[0] + 2., box_location[1] - text_size_lwir[1]]
        textbox_location_lwir = [box_location[0], box_location[1] - text_size_lwir[1], box_location[0] + text_size_lwir[0] + 4.,box_location[1]]

        draw.rectangle(xy=textbox_location_vis, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location_vis, text='{:.4f}'.format(det_scores[i].item()), fill='white', font=font)
        
        draw_lwir.rectangle(xy=textbox_location_lwir, fill=label_color_map[det_labels[i]])
        draw_lwir.text(xy=text_location_lwir, text='{:.4f}'.format(det_scores[i].item()), fill='white', font=font)
    
    new_image = Image.new('RGB',(original_image.size[0], original_image.size[1]))
    new_image.paste(original_image,(0,0))
    new_image_lwir = Image.new('RGB',(original_image.size[0], original_image.size[1]))
    new_image_lwir.paste(original_lwir,(0,0))

    del draw
    del draw_lwir

    return new_image, new_image_lwir

def visualize(result_filename, vis_dir, fdz_case):

    data_list = list()
    for line in open(result_filename):
        data_list.append(line.strip().split(','))
    data_list = np.array(data_list)

    input_size = config.test.input_size
    
    # Load dataloader for Fusion Dead Zone experiment
    FDZ = [FusionDeadZone(config.FDZ_case[fdz_case], tuple(input_size))]
    config.args.test.img_transform = Compose(FDZ)
    config.args.test.co_transform = Compose([
                                             Resize(input_size), \
                                             ToTensor()
                                            ])

    test_dataset = KAISTPed(config.args, condition="test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=config.args.dataset.workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)

    for idx, blob in enumerate(tqdm(test_loader, desc='Visualizing')): 
        image_vis, image_lwir, _, _, _ = blob
        detection = data_list[data_list[:,0] == str(idx+1)].astype(float)
        vis, lwir = detect(image_vis, image_lwir, detection)

        if fdz_case=='sidesblackout_a':
            new_images = Image.blend(vis, lwir, 0.5)
        elif fdz_case=='sidesblackout_b':
            new_images = Image.blend(lwir, vis, 0.5)
        elif fdz_case=='surroundingblackout':
            ## We emptied the center considering the visualization.
            ## In reality, the original image is used as an input.
            x = 120
            y = 96
            vv = np.array(vis)
            vv[y:-y, x:-x] = 0
            ##
            new_images = np.array(lwir) + vv
            new_images = Image.fromarray(new_images.astype(np.uint8))
        elif fdz_case in ['blackout_r', 'blackout_t', 'original']:
            new_images = Image.new('RGB',(2*vis.size[0], vis.size[1]))
            new_images.paste(vis,(0,0))
            new_images.paste(lwir,(vis.size[0],0))

        new_images.save('./{}/{:06d}.jpg'.format(vis_dir, idx))