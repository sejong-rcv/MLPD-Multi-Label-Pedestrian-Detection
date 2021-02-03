from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import os
import os.path
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './jobs/2021-01-28_07h54m_SSD_KAIST_LF_Multi_Label/checkpoint_ssd300.pth.tar003'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d. \n' % (start_epoch))
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((512, 640))
to_tensor = transforms.ToTensor()
normalize_vis = transforms.Normalize(mean=[0.3465,0.3219,0.2842],std=[0.2358,0.2265,0.2274])
normalize_lwir = transforms.Normalize(mean=[0.1598],std=[0.0813])

#Data load
DB_ROOT = './datasets/kaist-rgbt/'
image_set = 'test-all-20.txt'
# {SET_ID}/{VID_ID}/{MODALITY}/{IMG_ID}.jpg
imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')  

ids = list()

for line in open(os.path.join(DB_ROOT, 'imageSets', image_set)):
    ids.append((DB_ROOT, line.strip().split('/')))

def detect(original_image, original_lwir, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    vis = normalize_vis(to_tensor(original_image))
    lwir = normalize_lwir(to_tensor(original_lwir))

    # Move to default device
    vis = vis.to(device)
    lwir = lwir.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(vis.unsqueeze(0),lwir.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k) 

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

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
    font = ImageFont.truetype("./calibril.ttf", 15)


    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        
        
        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])  
        draw_lwir.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw_lwir.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])  
        # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_score_vis = str(det_scores[0][i][0])[:7]
        text_score_lwir = str(det_scores[0][i][1])[:7]
        text_size_vis = font.getsize(text_score_vis)
        text_size_lwir = font.getsize(text_score_lwir)
        text_location_vis = [box_location[0] + 2., box_location[1] - text_size_vis[1]]
        textbox_location_vis = [box_location[0], box_location[1] - text_size_vis[1], box_location[0] + text_size_vis[0] + 4.,box_location[1]]
        text_location_lwir = [box_location[0] + 2., box_location[1] - text_size_lwir[1]]
        textbox_location_lwir = [box_location[0], box_location[1] - text_size_lwir[1], box_location[0] + text_size_lwir[0] + 4.,box_location[1]]

        draw.rectangle(xy=textbox_location_vis, fill=label_color_map[det_labels[i]])
        # draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
        draw.text(xy=text_location_vis, text='{:.4f}'.format(det_scores[0][i][0].item()), fill='white', font=font)

        draw_lwir.rectangle(xy=textbox_location_lwir, fill=label_color_map[det_labels[i]])
        # draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
        draw_lwir.text(xy=text_location_lwir, text='{:.4f}'.format(det_scores[0][i][1].item()), fill='white', font=font)
    # del draw
    
    new_image = Image.new('RGB',(2*original_image.size[0], original_image.size[1]))
    new_image.paste(original_image,(0,0))
    new_image.paste(original_lwir,(original_image.size[0],0))
    
    return new_image

if __name__ == '__main__':


    for ii in enumerate(ids):
        frame_id = ii[1][0]
        set_id = ii[1][1][0]
        vid_id = ii[1][1][1]
        img_id = ii[1][1][2]
    
        lwir = Image.open( imgpath % ( frame_id, set_id, vid_id, 'lwir', img_id ), mode='r' ).convert('L')
        vis = Image.open( imgpath % ( frame_id, set_id, vid_id, 'visible', img_id ), mode='r' )
        
        annotate_vis = detect(vis,lwir, min_score=0.3, max_overlap=0.425, top_k=200)
        annotate_vis.save('./Detection_visualization/I{:06d}.jpg'.format(ii[0]))
        print('%d.jpg saved' %ii[0])
  