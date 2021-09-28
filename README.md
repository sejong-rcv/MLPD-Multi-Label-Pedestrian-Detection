## MLPD: Multi-Label Pedestrian Detectorin Multispectral Domain
[![stars](https://img.shields.io/github/stars/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection?style=social)](https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection)
[![IEEE Xplore RA-L 2021](https://img.shields.io/badge/-IEEE%20Xplore%20RA--L%202021-blue)](https://ieeexplore.ieee.org/document/9496129)
 
[MLPD: Multi-Label Pedestrian Detectorin Multispectral Domain](https://ieeexplore.ieee.org/document/9496129)
 
### 📢Notice : Leaderboard is available.
 ![Leaderboard](https://img.shields.io/badge/Leaderboard-Multispectral%20Pedestrian%20Detection-brightgreen)


## Abstract     
Multispectral pedestrian detection has been actively studied as a promising multi-modality solution to handle illumination and weather changes. Most multi-modality approaches carry the assumption that all inputs are fully-overlapped. However, these kinds of data pairs are not common in practical applications due to the complexity of the existing sensor configuration. In this paper, we tackle multispectral pedestrian detection, where all input data are 
not paired. To this end, we propose a novel single-stage detection framework that leverages multi-label learning to learn input state-aware features by assigning a separate label according to the given state of the input image pair. We also present a novel augmentation strategy
by applying geometric transformations to synthesize the unpaired multispectral images. In extensive experiments, we demonstrate the efficacy of the proposed method under various real-world conditions, such as fully-overlapped images and partially-overlapped images, in stereo-vision.

![demo](./video.gif)


### Results
|Methods|Backbone|All|Day|Night|
|:--:|:--:|:--:|:--:|:--:|
| ACF | - | 47.32 | 42.57 | 56.17 |
| Halfway Fusion | VGG16 | 25.75 | 24.88 | 26.59 | 
| Fusion RPN+BF | VGG16 | 18.29 | 19.57 | 16.27 |
| IAF R-CNN | VGG16 | 15.73 | 14.55 | 18.26 |
| IATDNN+IASS | VGG16 | 14.95 | 14.67 | 15.72 |
| CIAN | VGG16 | 14.12  | 14.77  | 11.13 |
| MSDS-RCNN | VGG16 | 11.34 | 10.53 | 12.94 |
| AR-CNN | VGG16 | 9.34 |  9.94 | 8.38 |
| MBNet | ResNet50 | 8.13 | 8.28 | 7.86 |
| MLPD (Ours) | VGG16 | 7.58 | 7.95 | 6.95 |
| MLPD (Ours) | ResNet50 | 7.61 | 8.36 | 6.35 |



## Contents

- [Prerequisites](#Prerequisites)
- [Getting Started](#Getting-Started)
  - [Docker](#Docker)
- [Dataset](#Dataset)
- [Training and Evaluation](#Training-and-Evaluation)
  - [Train](#Train)
  - [Pretrained Model](#Pretrained-Model)
  - [Evaluation](#Evaluation)
  - [Fusion Dead Zone Experiment](#Fusion-Dead-Zone-Experiment)


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
+-- doc
+-- docker
+-- src
|   +-- kaist_annotations_test20.json
|   +-- imageSets
|   |   +-- train-all-02.txt
|   |   +-- test-all-20.txt
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
If you want to adjust the number of GPUs, add 'CUDA_VISIBLE_DEVIECES'
(optional) e.g. `CUDA_VISIBLE_DEVIECES=0,1 python train_eval.py`

### Pretrained Model
If you want to skip the training process, download the pre-trained model and place it in the directory `src/jobs/`.

- [Pretrained Model](https://drive.google.com/file/d/1smXP4xpSDYC8cL_bbT9-E2aywROLlC2v/view?usp=sharing)

### Evaluation
`python eval.py`

### Fusion Dead Zone Experiment
If you want to check the results of the 'FDZ' experiments, you can run the file

`sh FDZ_exp.sh`


## Acknowledgement
We appreciate the provider of SSD code [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) and [Soonmin Hwang](https://github.com/SoonminHwang) who contributed to the SSD-like Halfway architecture. This code is built mostly based on them.

## Citation

```
@INPROCEEDINGS{ IEEE RA-L with IROS2021
  author = {JIWON KIM*, HYEONGJUN KIM*, TAEJOO KIM*, NAMIL KIM, AND YUKYUNG CHOI†},
  title = {Multi-Label-Pedestrian-Detection},
  booktitle = {IEEE Robotics and Automation Letters (RA-L)},
  year = {2021}
}
```

