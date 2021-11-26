## [MLPD: Multi-Label Pedestrian Detectorin Multispectral Domain](https://ieeexplore.ieee.org/document/9496129)

[![IEEE RA-L 2021](https://img.shields.io/badge/-IEEE%20RA--L%202021-blue)](https://ieeexplore.ieee.org/document/9496129) [![Star on GitHub](https://img.shields.io/github/stars/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.svg?style=social)](https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection/stargazers)

### 📢Notice : Leaderboard is available.
 [![Leaderboard](https://img.shields.io/badge/Leaderboard-Multispectral%20Pedestrian%20Detection-brightgreen)](https://eval.ai/web/challenges/challenge-page/1247/leaderboard/3137)


## Abstract     
Multispectral pedestrian detection has been actively studied as a promising multi-modality solution to handle illumination and weather changes. Most multi-modality approaches carry the assumption that all inputs are fully-overlapped. However, these kinds of data pairs are not common in practical applications due to the complexity of the existing sensor configuration. In this paper, we tackle multispectral pedestrian detection, where all input data are 
not paired. To this end, we propose a novel single-stage detection framework that leverages multi-label learning to learn input state-aware features by assigning a separate label according to the given state of the input image pair. We also present a novel augmentation strategy
by applying geometric transformations to synthesize the unpaired multispectral images. In extensive experiments, we demonstrate the efficacy of the proposed method under various real-world conditions, such as fully-overlapped images and partially-overlapped images, in stereo-vision.

![demo](./Doc/figure/video.gif)


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

![FPPI KIAS Benchmark](./Doc/figure/figure.jpg)


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
If you want to adjust the number of GPUs, add 'CUDA_VISIBLE_DEVICES'
(optional) e.g. `CUDA_VISIBLE_DEVICES=0,1 python train_eval.py`

### Pretrained Model
If you want to skip the training process, download the pre-trained model and place it in the directory `pretrained/`.

- [Pretrained Model](https://drive.google.com/file/d/1smXP4xpSDYC8cL_bbT9-E2aywROLlC2v/view?usp=sharing)

Or just run below command

```bash
$ ./script/download_pretrained_model.sh
```

### Inference

Try below command to get inference from pretrained model

```bash
$ cd src
$ python inference.py --FDZ original --model-path ../pretrained/best_checkpoint.pth.tar
```
++ If you want to visualize the results, try addding the `--vis` argument. Like below

```bash
$ python inference.py --FDZ original --model-path ../pretrained/best_checkpoint.pth.tar --vis
```
Visualization results are stored in 'result/vis'.


### Fusion Dead Zone Experiment
If you want to check the results of the 'FDZ' experiments, you can run the file

```bash
$ sh FDZ_exp.sh
```

### Evaluation Benchmark

You can evaluate the result files of the models with the evaluation script below and draw all the results of state-of-the-art methods in a single figure to make it easy to compare.

The figure represents the miss-rate against false positives per image. Please refer to the paper for more understanding of the metric and figure.

Annotation files only support a json format. For results files, json and txt formats are supported.
(multiple `--rstFiles` are supported)

```bash
cd ..
cd evaluation_script

$ python evaluation_script.py \
	--annFile KAIST_annotation.json \
	--rstFile state_of_arts/MLPD_result.txt \
			  state_of_arts/ARCNN_result.txt \
			  state_of_arts/CIAN_result.txt \
			  state_of_arts/MSDS-RCNN_result.txt \
			  state_of_arts/MBNet_result.txt \
	--evalFig figure.jpg
  
```
![result img](./Doc/figure/figure.jpg)




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
