
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
