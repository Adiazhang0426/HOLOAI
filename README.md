<!--
 * @Description: 
 * @Author: Adiazhang
 * @Date: 2024-05-29 16:45:10
 * @LastEditTime: 2024-06-02 11:27:40
 * @LastEditors: Adiazhang
-->
# YOLO detection + location + cluster segmentation for holographic particle field

This repository contains the code for the paper:
 **Adaptive in-focus particle detection and segmentation in holographic 3D image with mechanism-guided machine learning**
Hang Zhang, Boyi Wang, Letian Zhang, Yue Zhao, Yu Wang, Jianhan Feng, Wei Xiao, Gaofeng Wang, Yingchun Wu and Xuecheng Wu

# Abstract
In the domainofthree-dimensional (3D) particle holography, diverse imaging characteristics across particles of varying scales and positions pose a notable impediment to efficient infor mation extraction. This research introduces an adaptive detection and segmentation approach employing mechanism-guided machine learning techniques. Leveraging a detection network and a particle size distribution oriented cropping operation derived from the atomization mechanism, this method delineates precise regions of interest (ROIs) of particles. Noteworthy considerations encompass not just the characterization of blurred boundaries but also nuanced grayscale variances along reconstruction directions, offering crucial insights for cluster-based segmentation methods. Rigorous evaluation of different detection network settings and segmen tation methods, and detailed analysis are conducted on experimental data from an icing wind tunnel and calibration board underscores detailed analysis, including the ablation experiments of mechanism-guided opearions. The influence of the hyperparameter in segmentation is also illustrated. To showcase its robustness, the proposed method is applied to segment a swirl spray particle field, demonstrating its versatility and applicability across varied particle field scenarios.

<img src="src\algorithm.gif" alt="precess" width="950" height="500">      
<br /> If you have any question on the code, please contact the author: 22227138@zju.edu.cn.

## Requirements
- Python 3.8.0
- Pytorch 1.12.1 
- torchvision 0.13.1
- PyYAML 3.12
- scipy 1.7.3
- opencv-python 4.6.0.66
- numpy 1.22.4
- scikit-image 0.19.3
- scikit-learn 0.22.2

The versions of site-packages above are not the only suitable ones, but they are the ones that the author has tested.
## Usage 
### Data and weights acquirement 
- All data and weights are available in **https://drive.google.com/drive/folders/1C9QlaklXpApIpE8AukCNMTKUnZw_YcGJ?usp=drive_link**. Training dataset and test data are in corresponding folders with the zip file extension, kust download, unzip the files and put them in the ```data``` folder. The pre-trained weights are in the ```weights``` folder with the zip file extension. ```yolov8n.pt``` and ```traced_model.pt``` should be placed in the main directory, while ```YOLOv7e6e.pt``` and ```yolov10x.pt``` should be placed in the ```weights``` folder in the main directory.

### Preprocessing
### Train
- Create the ```runs``` folder in the main directory.
- For provided dataset, the pre-trained model is saved in ```weights/yolov7e6e.pt``` and ```weights/yolov10x.pt``` by default. If you want to apply to your own dataset, for the YOLOv7e6e network, just run the ```train_aux_v7.py```, adjust the input parameters in the parser and paths in ```YOLOv7\yolov7dataset.yaml```; for other detection networks, just run the  ```MAIN.py```, choose the model configurations in ```ultralytics/cfg/models``` and adjust the paths in ```ultralytics\cfg\datasets\mydataset.yaml```.
### Detection & Location & Segmentation
- The reconstructed holographic images of calibration dot board is saved in file ```data/cali_test```, and the experimenal data of icing wind tunnel is saved in file ```data/exp_test``` by default.
- For location and segmentation, create the ```results``` folder in the main directory and the hyperparameters can be adjusted in the parser of ```MAIN.py``` and the functions used are in ```main_utils.py```.
### Evaluation
- For evaluation, create the ```tests``` in the main directory and the ground truth files are available in ```data/ground_truth``` with the format of txt.

## Acknowledgement
The code is partly based on the work of **YOLOv10: Real-Time End-to-End Object Detection** and **YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**.
## Citation
If you find this work useful in your research, please consider citing:

