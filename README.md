# MLPD: Multi-Label Pedestrian Detectorin Multispectral Domain
### RA-L with IROS 2021 accepted Paper
[MLPD: Multi-Label Pedestrian Detectorin Multispectral Domain](./MLPD/MLPD.pdf)

## Abstract     
Multispectral pedestrian detection has been actively studied as a promising multi-modality solution to handle illumination and weather changes. Most multi-modality approaches carry the assumption that all inputs are fully-overlapped. However, these kinds of data pairs are not common in practical applications due to the complexity of the existing sensor configuration. In this paper, we tackle multispectral pedestrian detection, where all input data are 
not paired. To this end, we propose a novel single-stage detection framework that leverages multi-label learning to learn input state-aware features by assigning a separate label according to the given state of the input image pair. We also present a novel augmentation strategy
by applying geometric transformations to synthesize the unpaired multispectral images. In extensive experiments, we demonstrate the efficacy of the proposed method under various real-world conditions, such as fully-overlapped images and partially-overlapped images, in stereo-vision.

![demo](./video.gif)

## Contents

- [Prerequisites](#Prerequisites)
- [Getting Started](#Getting-Started)
  - [Training and Evaluation](#Training-and-Evaluation)
  - [Pretrained Models](#Pretrained-Models)
- [Dataset](#Dataset)

## Prerequisites

- Ubuntu 18.04
- Python 3.7
- Pytorch 1.6.0
- Torchvision 0.7.0
- CUDA 10.1
- docker/requirements.txt

## Getting Started

### Git Clone

```
git clone https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.git
cd MLPD-Multi-Label-Pedestrian-Detection
```

### Docker

- Prerequisite
  - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

```
cd docker
make docker-make
```

#### Make Contianer

```
cd ..
nvidia-docker run -it --name mlpd -v $PWD:/workspace -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all --shm-size=8G mlpd /bin/bash
```


## Dataset

For multispectral pedestrian detection, we train and test the proposed model on the [KAIST dataset](https://github.com/SoonminHwang/rgbt-ped-detection), you should first download the dataset. By default, we assume the dataset is stored in `data/kaist-rgbt`. Please see more details below.

We train the proposed model with paired annotations(`annotations_paired`) provided by [AR-CNN](https://github.com/luzhang16/AR-CNN).
Download and place them in the directory `data/kaist-rgbt/`.


``` 
<DATA_PATH>

+-- docker
+-- src
+-- kaist_annotations_test20.json
+-- imageSets
|   +-- train-all-02.txt
|   +-- test-all-20.txt
+-- data
|   +-- kaist-rgbt
|   |   +-- annotations_paired
|   |   |   +-- set00
|   |   |   |   +-- V000
|   |   |   |   |   +-- lwir
|   |   |   |   |   |   +-- I00000.txt
|   |   |   |   |   +-- visible
|   |   |   |   |   |   +-- I00000.txt
|   |   +-- images
|   |   |   +-- set00
|   |   |   |   +-- V000
|   |   |   |   |   +-- lwir
|   |   |   |   |   |   +-- I00000.jpg
|   |   |   |   |   +-- visible
|   |   |   |   |   |   +-- I00000.jpg


```

## Training and Evaluation

If you want to change default parameters, you can modify them in the module `src/config.py`.

### Train
Please, refer to the following code to train and evaluate the proposed model.
```
cd src
python train_eval.py
```

### Pretrained Model
If you want to skip the training process, download the pre-trained model and place it in the directory `src/jobs/`.

- [Pretrained Model](https://drive.google.com/file/d/1smXP4xpSDYC8cL_bbT9-E2aywROLlC2v/view?usp=sharing)

### Evaluation
`python eval.py`

### Fusion Dead Zone Experiment
If you want to check the results of the 'FDZ' experiments, you can run the file

`sh FDZ_exp.sh`


## Additional Experiments
*These additional expriments are not handled in the submitted manuscript.*

### Experimental Results Compare to RGB-based SOTA object detection methods on the KAIST dataset
Test results on state-of-the-art models of RGB-based pedestrian detection benchmarks on the KAIST dataset.

The result of other studies (e.g. CSP, YOLO v3, YOLO v4, and YOLO-ACN) depending on train modality with respect to AP.

> | Methods | Modality |   AP  |
> |:-------:|:--------------:|:-----:|
> |   MLPD(ours)  |   RGB+Thermal  | **85.43** |
>|   [CSP](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.html)   |       RGB      | 65.14 |
> |   [CSP](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.html)   |     Thermal    | 70.77 |
> |  [YOLOv5](https://github.com/ultralytics/yolov5) |       RGB      | 70.18 |
> |  [YOLOv5](https://github.com/ultralytics/yolov5) |     Thermal    | 75.35 |
> |  [YOLOv4](https://arxiv.org/abs/2004.10934v1) |     Thermal    | 76.9 |
> |  [YOLOv3](https://arxiv.org/abs/1804.02767v1) |     Thermal    | 75.5  |
> |  [YOLO-ACN](https://ieeexplore.ieee.org/abstract/document/9303478) |     Thermal    | 76.2 |


### Ablation Study
Additional ablation experiments of the proposed method.

- Ablation Study on Fusion Methods.
>
>| Fusion Method   | SUA | MLL | SMF |  Miss Rate (all/day/night)  | 
>|:---------------:|:---:|:---:|:---:|:-----------:| 
>|  Early fusion   |  -  |  -  |  -  | 11.21/13.41/6.54     |
>|  Early fusion   |  v  |  -  |  -  | 8.69/9.78/6.42            |
>|  Early fusion   |  v  |  v  |  -  | 7.77/8.95/**5.47**            |
>|  **Halfway fusion** |  v  |  v  |  v  | **7.58**/**7.95**/6.95            |
>
> SUA : Semi-Unpaired Augmentation,
> MLL : Multi-Label Learning, 
> SMF : Shared Multi-Fusion

- Quantitative Result of the Proposed Method Depending on the Backbone Network.

> | Methods | Backbone |  Miss rate  |   AP  |
> |:-------:|:--------------:|:-----:|:-----:| 
> |   MLPD(ours)  |  VGG-16 | **7.58** | 85.43 |
> |   MLPD(ours)  |   Resnet-50  | 7.61 | **85.45** |
> |   MLPD(ours)  |   Resnet-101  | 9.10 | 84.11 |

## Acknowledgement
We appreciate the provider of SSD code [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) and [Soonmin Hwang](https://github.com/SoonminHwang) who contributed to the proposed architecture. This code is built mostly based on them.

## Citation

```
@INPROCEEDINGS{ IEEE RA-L with IROS2021
  author = {JIWON KIM*, HYEONGJUN KIM*, TAEJOO KIM*, NAMIL KIM, AND YUKYUNG CHOI†},
  title = {Multi-Label-Pedestrian-Detection},
  booktitle = {IEEE Robotics and Automation Letters (RA-L). (Accepted. To Appear.)},
  year = {2021}
}
```

