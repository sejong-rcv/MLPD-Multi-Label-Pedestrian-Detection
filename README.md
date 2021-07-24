# Multi-Label-Pedestrian-Detection

## Abstract     
Multispectral pedestrian detection has been actively studied as a promising multi-modality solution to handle illumination and weather changes. Most multi-modality approaches carry the assumption that all inputs are fully-overlapped. However, these kinds of data pairs are not common in practical applications due to the complexity of the existing sensor configuration. In this paper, we tackle multispectral pedestrian detection, where all input data are 
not paired. To this end, we propose a novel single-stage detection framework that leverages multi-label learning to learn input state-aware features by assigning a separate label according to the given state of the input image pair. We also present a novel augmentation strategy
by applying geometric transformations to synthesize the unpaired multispectral images. In extensive experiments, we demonstrate the efficacy of the proposed method under various real-world conditions, such as fully-overlapped images and partially-overlapped images, in stereo-vision.

### Paper : [Paper](./MLPD/MLPD.pdf)

![demo](./video.gif)

## Contents

- [Prerequisites](#Prerequisites)
- [Pretrained Models](#Pretrained-Models)
- [Getting Started](#Getting-Started)
- [Dataset](#Dataset)
- [Training and Evaluation](#Training-and-Evaluation)

## Prerequisites

- Ubuntu 18.04
- Python 3.7
- Pytorch 1.6.0
- Torchvision 0.7.0
- CUDA 10.1

## Pretrained Model
Download the pretrained model and place it in the directory `./src/result/`.

- [Pretraibed Model](https://drive.google.com/file/d/1smXP4xpSDYC8cL_bbT9-E2aywROLlC2v/view?usp=sharing)

## Getting Started

#### Git clone

```
git clone https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.git
cd Multi-Lable-Pedestrian-Detection/docker
```

#### Docker

- Prerequisite
  - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

```
make docker-make
```

#### Make Contianer (example)

```
nvidia-docker run -it --name mlpd -p 8810:8810 -w /home/jwkim/workspace -v /home/jwkim/workspace:/home/jwkim/workspace -v /data/:/raid -e NVIDIA_VISIBLE_DEVICES=ALL --shm-size=32G mlpd /bin/bash
```

## Dataset

For Multispectral pedestrian detection, we train and test our model on the [KAIST dataset](https://github.com/SoonminHwang/rgbt-ped-detection), you should first download the dataset. By default, we assume the dataset is stored in `./src/data/kaist-rgbt`. Please see more details below.

We trained the model with Paired Annotations provided by [AR-CNN](https://github.com/luzhang16/AR-CNN).
Download them and place them in the directory `./src/data/kaist-rgbt/`.

``` 
<DATA_PATH>

+-- data
|   +-- kaist-rgbt
|   |   +-- kaist_annotations_test20.json
|   |   +-- annotations-xml-15
|   |   |   +-- set00
|   |   |   |   +-- V000
|   |   |   |   |   +-- I00000.xml
|   |   +-- images
|   |   |   +-- set00
|   |   |   |   +-- V000
|   |   |   |   |   +-- lwir
|   |   |   |   |   |   +-- I00000.jpg
|   |   |   |   |   +-- visible
|   |   |   |   |   |   +-- I00000.jpg
|   |   +-- imageSets
|   |   |   +-- train-all-02.txt
|   |   |   +-- test-all-20.txt

```

## Training and Evaluation

### Train

`python train_eval.py`

### Evaluation

`python eval.py`


## Experiment

### SOTA RGB based object detections

| Methods | Train Modality |   AP  |
|:-------:|:--------------:|:-----:|
|   MLPD(ours)  |   RGB+Thermal  | 85.43 |
|   CSP   |       RGB      | 65.14 |
|   CSP   |     Thermal    | 70.77 |
|  YOLOv5 |       RGB      | 70.18 |
|  YOLOv5 |     Thermal    | 75.35 |
|  YOLOv3 |     Thermal    | 75.5  |
|  YOLOv4 |     Thermal    | 76.9 |
|  YOLO-ACN |     Thermal    | 76.2 |

The result of other studies (e.g. CSP, YOLO v3, YOLO v4, and YOLO-ACN) depending on
train modality with respect to AP.


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

