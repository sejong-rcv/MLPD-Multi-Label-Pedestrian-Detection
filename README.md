# Multi-Label-Pedestrian-Detection

This code is based on [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection). 

#### STEREO 

![STEREO](https://user-images.githubusercontent.com/44772344/108985713-af3bac80-76d4-11eb-9a4c-e42f8c24620d.gif)

#### EOIR

![EOIR](https://user-images.githubusercontent.com/44772344/108986272-4e60a400-76d5-11eb-9ce4-d76dac5c3a3f.gif)




### Installation

#### Git clone

```
git clone https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.git
cd Multi-Lable-Pedestrian-Detection/docker
```

#### Build docker 

```
make docker-make
```

#### Make Contianer (example)

```
nvidia-docker run -it --name mlpd -p 8810:8810 -w /home/jwkim/workspace -v /home/jwkim/workspace:/home/jwkim/workspace -v /data/:/raid -e NVIDIA_VISIBLE_DEVICES=ALL --shm-size=32G mlpd /bin/bash
```

### Dataset

For Multispectral pedestrian detection, we train and test our model on [Multispectral Pedestrian Detection: Benchmark Dataset and Baselines](https://github.com/SoonminHwang/rgbt-ped-detection), you should firstly download the dataset. By default, we assume the dataset is stored in `./data/kaist-rgbt`. Please see details below

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


### Citation

```
@INPROCEEDINGS{ACCV2020,
  author = {JIWON KIM*, HYEONGJUN KIM*, TAEJOO KIM*, NAMIL KIM, AND YUKYUNG CHOI*},
  title = {Multi-Label-Pedestrian-Detection},
  booktitle = {-},
  year = {2021}
}
```
