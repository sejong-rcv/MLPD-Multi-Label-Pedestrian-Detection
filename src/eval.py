from tqdm import tqdm
from typing import Dict
import argparse
import config
# import importlib
import numpy as np

import torch
import torch.nn as nn

from datasets import KAISTPed
from utils.transforms import FusionDeadZone

# TODO(sohwang): why do we need this?
# args = importlib.import_module('config').args


def run_inference(model_path: str, fdz_case: str) -> Dict:
    """Load model and run inference

    Load pretrained model and run inference on KAIST dataset with FDZ setting.

    Parameters
    ----------
    model_path: str
        Full path of pytorch model
    fdz_case: str
        Fusion dead zone case defined in utils/transforms.py:FusionDeadZone

    Returns
    -------
    Dict
        A Dict of numpy arrays (K x 5: xywh + score) for given image_id key
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)['model']
    model = model.to(device)

    model = nn.DataParallel(model)
    model.eval()

    input_size = config.test.input_size
    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)

    # Load dataloader for Fusion Dead Zone experiment
    FDZ = [FusionDeadZone(fdz_case, tuple(input_size))]
    config.test.img_transform.add(FDZ)

    args = config.args
    batch_size = config.test.batch_size * torch.cuda.device_count()
    test_dataset = KAISTPed(args, condition="test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.dataset.workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)
    print(test_loader)

    results = dict()
    with torch.no_grad():
        for i, blob in enumerate(tqdm(test_loader, desc='Evaluating')):
            image_vis, image_lwir, boxes, labels, indices = blob

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs, predicted_scores = model(image_vis, image_lwir)

            # Detect objects in SSD output
            detections = model.module.detect_objects(predicted_locs, predicted_scores,
                                                     min_score=0.1, max_overlap=0.45, top_k=200)

            det_boxes_batch, det_labels_batch, det_scores_batch = detections[:3]

            for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, indices):
                boxes_np = boxes_t.cpu().numpy().reshape(-1, 4)
                scores_np = scores_t.cpu().numpy().mean(axis=1).reshape(-1, 1)

                # TODO(sohwang): check if labels are required
                # labels_np = labels_t.cpu().numpy().reshape(-1, 1)

                xyxy_np = boxes_np * xyxy_scaler_np
                xywh_np = xyxy_np
                xywh_np[:, 2] -= xywh_np[:, 0]
                xywh_np[:, 3] -= xywh_np[:, 1]

                results[image_id.item() + 1] = np.hstack([xywh_np, scores_np])
    return results


def save_results(results: Dict, result_filename: str):
    """Save detections

    Write a result file (.txt) for detection results.
    The results are saved in the order of image index.

    Parameters
    ----------
    results: Dict
        Detection results for each image_id: {image_id: box_xywh + score}
    result_filename: str
        Full path of result file name

    """

    if not result_filename.endswith('.txt'):
        result_filename += '.txt'

    with open(result_filename, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x, y, w, h, score in detections:
                f.write(f'{image_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{score:.8f}\n')


if __name__ == '__main__':

    FDZ_list = FusionDeadZone._FDZ_list

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--FDZ', default='original', type=str, choices=FDZ_list,
                        help='Setting for the "Fusion Dead Zone" experiment. e.g. {}'.format(', '.join(FDZ_list)))
    parser.add_argument('--model-path', type=str,
                        help='Pretrained model for evaluation.')
    parser.add_argument('--result-file', type=str,
                        help='Detection result file for evaluation.')
    arguments = parser.parse_args()

    assert arguments.model_path or arguments.result_file, "Please specify '--model-path' or '--results-file'"
    if arguments.model_path and arguments.result_file:
        print('Both --model-path and --results-file are specified. Ignore --model-path.')
        arguments.model_path = None

    print(arguments)

    fdz_case = arguments.FDZ.lower()
    model_path = arguments.model_path
    result_filename = arguments.result_file

    # Run inference to get detection results
    if model_path:
        result_filename = model_path + f'.{fdz_case}_TEST_det'

        # Run inference
        results = run_inference(model_path, fdz_case)

        # Save results
        save_results(results, result_filename)

    # TODO(sohwang): Load results

    # TODO(sohwang): Evaluate performance
