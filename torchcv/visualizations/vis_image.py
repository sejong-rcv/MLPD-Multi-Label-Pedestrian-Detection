import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pdb

def draw_boxes(ax, im, boxes, labels, scores, classes=None, cls_color=np.random.rand(32,3), thres=0.6, filename=None):

    try:
        ax.cla()
        if len(im.shape) == 2:
            ax.imshow(im.astype(np.uint8), cmap='gray')
        else:
            ax.imshow(im.astype(np.uint8))
        # cls_color = ['gray', 'yellow','cyan','red','green','blue','white', 'yellow','cyan','red','green','blue','white']
        
        # labels = labels.squeeze(1)
        # cls_color = cm.tab20(labels).reshape(-1, 4)[:,:3]
        # if isinstance(labels, torch.tensor):
        #     labels = labels.numpy()

        for box, label, score in zip(boxes, labels, scores):

            if score < thres:   continue
            
            ax.add_patch(
                plt.Rectangle((box[0], box[1]),
                              box[2] - box[0],
                              box[3] - box[1], fill=False,                              
                              edgecolor=cls_color[int(label)], linewidth=1.5)
                )
            if classes is not None:
                ax.text(box[0], box[1] - 2,
                        '{:s} {:.2f}'.format( classes[int(label)], score),
                        bbox=dict(facecolor=cls_color[int(label)], alpha=0.5),
                        fontsize=10, color='white')

        plt.axis('off')
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename,transparent = True)
    except:
        import torchcv.utils.trace_error
        pdb.set_trace()

def vis_image(img, boxes=None, label_names=None, scores=None):
    '''Visualize a color image.

    Args:
      img: (PIL.Image/tensor) image to visualize.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      label_names: (list) label names.
      scores: (list) confidence scores.

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_bbox.py
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_image.py
    '''
    # Plot image
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if isinstance(img, torch.Tensor):
        img = torchvision.transforms.ToPILImage()(img)
    ax.imshow(img)

    # Plot boxes
    if boxes is not None:
        for i, bb in enumerate(boxes):
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0] + 1
            height = bb[3] - bb[1] + 1

            ax.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor='red', linewidth=2))

            caption = []
            if label_names is not None:
                caption.append(label_names[i])

            if scores is not None:
                caption.append('{:.2f}'.format(scores[i]))

            if len(caption) > 0:
                ax.text(bb[0], bb[1],
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    # Show
    plt.show()
